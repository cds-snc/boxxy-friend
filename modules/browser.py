
from playwright.async_api import async_playwright
import asyncio
import re
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
from playwright.sync_api import expect
from modules.objective import ProbeFailed
from modules.window_layout import WindowBounds

class Browser:
    # Short enough that a click blocked by an overlay fails fast instead of
    # waiting Playwright's default 30s for the element to become actionable.
    ACTION_TIMEOUT_MS = 5000

    def __init__(self, base_url, window_bounds: WindowBounds | None = None):
        self.base_url = base_url
        self.window_bounds = window_bounds
        self.browser = None
        self.playwright = None
        self.objective = None
        self.before_action = None

    def _guard_navigation(self, route):
        request = route.request
        if request.is_navigation_request() and not self.objective.allows(request.url):
            self.scope_error = f"Blocked out-of-scope navigation: {request.url}"
            route.abort()
        else:
            route.continue_()

    def configure_scope(self, objective, before_action):
        self.objective = objective
        self.before_action = before_action
        self.scope_error = None

    def check_scope(self):
        if self.objective is not None:
            if self.scope_error:
                raise ProbeFailed(self.scope_error)
            if not self.objective.allows(self.page.url):
                raise ProbeFailed(f"Page left approved scope: {self.page.url}")

    def _action(self):
        self.check_scope()
        if self.before_action is not None:
            self.before_action()

    def open(self):
        self.playwright = sync_playwright().start()
        if self.window_bounds is None:
            self.browser = self.playwright.chromium.launch(headless=False)
            self.page = self.browser.new_page()
        else:
            bounds = self.window_bounds
            self.browser = self.playwright.chromium.launch(
                headless=False,
                args=[
                    f"--window-position={bounds.x},{bounds.y}",
                    f"--window-size={bounds.width},{bounds.height}",
                ],
            )
            self.page = self.browser.new_page(no_viewport=True)
            session = self.page.context.new_cdp_session(self.page)
            try:
                window = session.send("Browser.getWindowForTarget")
                session.send("Browser.setWindowBounds", {
                    "windowId": window["windowId"],
                    "bounds": {
                        "left": bounds.x, "top": bounds.y,
                        "width": bounds.width, "height": bounds.height,
                        "windowState": "normal",
                    },
                })
            finally:
                session.detach()
        self.page.set_default_timeout(self.ACTION_TIMEOUT_MS)
        if self.objective is not None:
            self.page.context.route("**/*", self._guard_navigation)
        self.page.goto(self.base_url)

    def close(self):
        # Must be called from the same thread that called open() (Playwright sync API requirement).
        try:
            if self.browser is not None:
                self.browser.close()
        finally:
            self.browser = None
            if self.playwright is not None:
                self.playwright.stop()
                self.playwright = None

    def explore_view(self):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")
        
        self.page.wait_for_load_state()

        return self.page.aria_snapshot()

    def page_info(self):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")

        try:
            title = self.page.title()
        except Exception:
            title = ""  # title() can fail mid-navigation; the URL is still useful on its own.
        return {"url": self.page.url, "title": title}

    # Matches ARIA snapshot node lines like: button "Design a form Start with a blank form."
    _ROLE_NAME_RE = re.compile(r'^\s*([a-zA-Z]+)\s+"(.*)"\s*$')

    # Looser variant for comboboxes, whose snapshot lines may carry a leading "- ", a trailing ":" (native
    # <select> with nested options) or ": <current value>" (custom ARIA comboboxes).
    _LOOSE_ROLE_NAME_RE = re.compile(r'^\s*(?:-\s+)?([a-zA-Z]+)\s+"((?:[^"\\]|\\.)*)"')

    @staticmethod
    def _clean_element(element):
        # Only clean ARIA name delimiters, not backslashes or punctuation inside names.
        return re.sub(
            r'^(\s*(?:-\s+)?[a-zA-Z]+\s+)\\?"(.*?)\\?"(\s*(?::.*|\[.*)?)\s*,?\s*$',
            r'\1"\2"\3',
            element,
        )

    def _locate_by_role_or_text(self, element):
        element = self._clean_element(element)
        match = self._LOOSE_ROLE_NAME_RE.match(element)
        if not match:
            return self.page.get_by_text(element), f'element with text "{element}"'
        role, name = match.group(1), match.group(2)
        locator = self.page.get_by_role(role, name=name, exact=True)
        if locator.count() == 0:
            locator = self.page.get_by_role(role, name=name)
        return locator, f'role "{role}" with name "{name}"'

    @classmethod
    def _option_label(cls, value):
        # Accept the option copied verbatim from the snapshot, e.g. option "As entered" [selected].
        value = cls._clean_element(value)
        match = cls._LOOSE_ROLE_NAME_RE.match(value)
        if match and match.group(1) == "option":
            return match.group(2)
        return value.strip()

    def use_combobox(self, element, value, index=None):
        self._action()
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")

        label = self._option_label(value)
        combobox, description = self._locate_by_role_or_text(element)
        combobox = combobox.first if index is None else combobox.nth(index)

        if combobox.evaluate("e => e.tagName.toLowerCase()") == "select":
            selected = self._select_native_option(combobox, label)
        else:
            selected = self._select_custom_option(combobox, label)

        self.page.wait_for_timeout(1000) # let the selection event actually do something.
        return f'I selected option "{selected}" in {description}'

    def _select_native_option(self, combobox, label):
        try:
            combobox.select_option(label=label)
        except Exception:
            try:
                combobox.select_option(value=label)
            except Exception:
                available = combobox.evaluate("e => Array.from(e.options, o => o.label)")
                raise ValueError(
                    f'Option "{label}" not found. Available options: '
                    + ", ".join(f'"{o}"' for o in available)
                ) from None
        return combobox.evaluate("e => e.selectedOptions[0] ? e.selectedOptions[0].label : ''")

    def _select_custom_option(self, combobox, label):
        if combobox.get_attribute("aria-expanded") != "true":
            combobox.click()

        # Options usually live in a popup referenced by aria-controls (or aria-owns), not inside the combobox.
        popup_id = combobox.get_attribute("aria-controls") or combobox.get_attribute("aria-owns")
        scope = self.page.locator(f'[id="{popup_id.split()[0]}"]') if popup_id else self.page

        options = scope.get_by_role("option", name=label, exact=True)
        if options.count() == 0 and combobox.evaluate(
            "e => e.isContentEditable || ['input', 'textarea'].includes(e.tagName.toLowerCase())"
        ):
            # Autocomplete comboboxes only render matching options after typing.
            combobox.fill(label)
            try:
                scope.get_by_role("option").first.wait_for(state="visible")
            except Exception:
                pass
        if options.count() == 0:
            options = scope.get_by_role("option", name=label)

        if options.count() == 0:
            available = scope.get_by_role("option").all_inner_texts()
            raise ValueError(
                f'Option "{label}" not found. Available options: '
                + (", ".join(f'"{o.strip()}"' for o in available) or "none visible")
            )

        options.first.click()
        return label

    def type(self, element, text, index=None):
        self._action()
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")

        element = self._clean_element(element)
        match = self._ROLE_NAME_RE.match(element)
        if match:
            role, name = match.group(1), match.group(2)
            result = f'I typed into role "{role}" with name "{name}" : "{text}"'
            locator = self.page.get_by_role(role, name=name)
        else:
            result = f'I typed into element with text: "{element}" : "{text}"'
            locator = self.page.get_by_text(element)
        if index is not None:
            locator = locator.nth(index)
        locator.fill(text)

        self.page.wait_for_timeout(1000) # let the typing event actually do something.

        # Read the value back so the agent learns whether the field kept the text (masks, max lengths, resets).
        try:
            value = locator.input_value()
        except Exception:
            return result
        if value == text:
            return result + "; the field now contains exactly that text"
        return result + f'; the field now contains "{value}" (differs from what was typed)'
    
    def click(self, element, index=None):
        self._action()
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")

        element = self._clean_element(element)
        result = ""
        # `element` is typically a line from the ARIA tree/snapshot, e.g.:
        #   button "Design a form Start with a blank form."
        # The quoted text is the *accessible name*, which is computed by
        # concatenating descendant text nodes (often inserting spaces that
        # don't exist verbatim in the DOM). get_by_text() matches literal
        # rendered text, so it can fail on composite names like this.
        # get_by_role() matches on the accessible name using the same
        # algorithm the ARIA tree uses, so prefer it when we have a role.
        match = self._ROLE_NAME_RE.match(element)
        if match:
            role, name = match.group(1), match.group(2)
            result = f'I clicked on role "{role}" with name "{name}"'
            self.page.get_by_role(role, name=name).nth(0 if index is None else index).click()
        else:
            result = f'I clicked on element with text: "{element}"'
            self.page.get_by_text(element).nth(0 if index is None else index).click()

        self.page.wait_for_timeout(1000) # let the click event actually do something.
        return result

    def check_test_inputs(self, objective, valid):
        self.check_scope()
        if not self.page.url.split("?")[0].endswith("/preview"):
            raise ProbeFailed("Required checks must run in GC Forms Test (/preview) mode.")
        first = self.page.get_by_role("textbox", name=re.compile(r"^Boxxy question 1(?:\s|$)"))
        expect(first).to_be_visible()
        if valid:
            for field in objective.fields:
                self._action()
                label = re.compile("^" + re.escape(field["label"]) + r"(?:\s|$)")
                if field["type"] == "radio":
                    group = self.page.get_by_role("group", name=label)
                    if group.count() == 0:
                        group = self.page.get_by_role("radiogroup", name=label)
                    radio = group.get_by_role("radio", name="Option A", exact=True)
                    radio.check()
                    expect(radio).to_be_checked()
                else:
                    control = self.page.get_by_role("textbox", name=label)
                    answer = f"Synthetic answer for {field['label']}"
                    control.fill(answer)
                    expect(control).to_have_value(answer)
        else:
            if first.input_value():
                raise ProbeFailed("The required-empty check must run before entering answers.")
        submit = self.page.get_by_role("button", name="Submit", exact=True)
        self._action()
        submit.click()
        if valid:
            expect(first).not_to_have_attribute("aria-invalid", "true")
            if not first.evaluate("e => e.validity.valid"):
                raise ProbeFailed("Valid synthetic input did not satisfy browser validation.")
            if self.page.locator('[aria-invalid="true"]').count():
                raise ProbeFailed("Test mode still contains invalid fields after valid input.")
            return "All approved inputs exercised with synthetic values; no field validation errors after Submit."
        # A static confirmation-page preview is not submission evidence.
        try:
            expect(first).to_have_attribute("aria-invalid", "true")
        except AssertionError as error:
            raise ProbeFailed(
                "Test mode did not expose required-empty rejection after Submit. Signed-out preview "
                "may not support validation; its static confirmation preview is not success evidence."
            ) from error
        return "Empty required input rejected after Submit (aria-invalid=true)."

    def download_form(self, element, objective, folder):
        self._action()
        target, _ = self._locate_by_role_or_text(element)
        with self.page.expect_download(timeout=self.ACTION_TIMEOUT_MS) as pending:
            target.click()
        download = pending.value
        if download.failure():
            raise ProbeFailed(f"Download failed: {download.failure()}")
        path = Path(folder) / "form.json"
        download.save_as(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise ProbeFailed(f"Downloaded file is not valid JSON: {error}") from error
        objective.validate_export(data)
        return str(path)

    def test_browser(self):
        async def run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto("https://example.com")
                await page.screenshot(path="example.png")
                
                await browser.close()

        asyncio.run(run())
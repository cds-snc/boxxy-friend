
from playwright.async_api import async_playwright
import asyncio
import re
from playwright.sync_api import sync_playwright

class Browser:
    # Short enough that a click blocked by an overlay fails fast instead of
    # waiting Playwright's default 30s for the element to become actionable.
    ACTION_TIMEOUT_MS = 5000

    def __init__(self, base_url):
        self.base_url = base_url
        self.browser = None
        self.playwright = None

    def open(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page()
        self.page.set_default_timeout(self.ACTION_TIMEOUT_MS)
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

    def use_combobox(self, element, value):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")

        label = self._option_label(value)
        combobox, description = self._locate_by_role_or_text(element)
        combobox = combobox.first

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

    def type(self, element, text):
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
    
    def click(self, element):
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
            self.page.get_by_role(role, name=name).nth(0).click()
        else:
            result = f'I clicked on element with text: "{element}"'
            self.page.get_by_text(element).nth(0).click()

        self.page.wait_for_timeout(1000) # let the click event actually do something.
        return result

    def test_browser(self):
        async def run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto("https://example.com")
                await page.screenshot(path="example.png")
                
                await browser.close()

        asyncio.run(run())
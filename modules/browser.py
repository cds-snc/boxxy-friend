
from playwright.async_api import async_playwright
import asyncio
import re
from playwright.sync_api import sync_playwright

class Browser:
    def __init__(self, base_url):
        self.base_url = base_url
        self.browser = None
        self.playwright = None

    def open(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page()
        self.page.goto(self.base_url)

    def explore_view(self):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")
        

        self.page.wait_for_load_state()

        return self.page.aria_snapshot()

    # Matches ARIA snapshot node lines like: button "Design a form Start with a blank form."
    _ROLE_NAME_RE = re.compile(r'^\s*([a-zA-Z]+)\s+"(.*)"\s*$')

    def click(self, element):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")

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
            print(f'Clicking on role "{role}" with name "{name}"')
            self.page.get_by_role(role, name=name).click()
        else:
            print("Clicking on element with text:", element)
            self.page.get_by_text(element).click()

        self.page.wait_for_timeout(500) # let the click event actually do something.

    def test_browser(self):
        async def run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto("https://example.com")
                await page.screenshot(path="example.png")
                
                await browser.close()

        asyncio.run(run())
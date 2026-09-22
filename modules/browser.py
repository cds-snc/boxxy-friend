
from playwright.async_api import async_playwright
import asyncio
from playwright.sync_api import sync_playwright

class Browser:
    def __init__(self, base_url):
        self.base_url = base_url
        self.browser = None
        self.playwright = None

    def open(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()
        self.page.goto(self.base_url)

    def explore_view(self):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")
        self.page.wait_for_load_state('networkidle')
        return self.page.aria_snapshot()

    def click(self, element):
        if self.browser is None:
            raise Exception("Browser is not open. Call open() first.")
        self.page.click(element)

    def test_browser(self):
        async def run():
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto("https://example.com")
                await page.screenshot(path="example.png")
                
                await browser.close()

        asyncio.run(run())
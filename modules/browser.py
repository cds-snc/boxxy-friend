
from playwright.async_api import async_playwright
import asyncio

class Browser:
    def __init__(self):
        self.meow = "meow"

    def test_browser(self):
        async def run():
            async with async_playwright() as p:
                browser = await p.firefox.launch(headless=True)
                page = await browser.new_page()
                
                await page.goto("https://example.com")
                await page.screenshot(path="example.png")
                
                await browser.close()

        asyncio.run(run())
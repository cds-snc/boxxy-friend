
from playwright.async_api import async_playwright
import asyncio

local_path = "./models/gemma-4-E2B-it"


from modules.llm import LLM

llm = LLM(model=None, processor=None)
llm.load_model(local_path)


# Async

async def main():
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        page = await browser.new_page()
        
        await page.goto("https://example.com")
        await page.screenshot(path="example.png")
        
        await browser.close()

asyncio.run(main())

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a poem about filling forms."},
]

response = llm.gen_text(messages)

# Parse output
print(response)

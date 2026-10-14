
from pathlib import Path

from setup import DEFAULT_FILENAME, DEFAULT_OUTPUT_DIR

local_path = Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIR / DEFAULT_FILENAME

from modules.llm import LLM

llm = LLM()
llm.load_model(local_path)

# Prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a poem about filling forms."},
]

response = llm.gen_text(messages)

# Parse output
print(response)

# Test the Playwright Browser
from modules.browser import Browser

browser = Browser("https://example.com")
browser.test_browser()
print("Smoke test passed: model generated text and Chromium saved example.png.")
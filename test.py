
## Test the LLM
local_path = "./models/gemma-4-E2B-it"

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

browser = Browser()
browser.test_browser()
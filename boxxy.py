## Boxxy is your friend!
## Boxxy will imitate a user in interacting with web pages to perform a task.
from modules.gui import BoxxyGui

test_url = "https://forms-staging.cdssandbox.xyz/en/form-builder"
local_path = "./models/gemma-4-E2B-it"

if __name__ == "__main__":
    BoxxyGui(test_url=test_url, model_path=local_path).run()

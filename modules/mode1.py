import json

class Mode1:
    def __init__(self, test_url, local_path):
        self.test_url = test_url
        self.test_behavior = "You are an automated testing agent mimicking a user utilizing a screen reader. " \
        "Your goal is to test the whole application for accessibility and functionality." \
        "You are testing GC Forms, a product for creating and managing web forms." \
        "Simulate developing a complex web form with various input types and validation rules." \
        "Be curious, explore, and provide feedback on your findings." \
        "Always perform an action until your task is completed."

        # Initialize notes to keep track of observations during exploration.
        self.notes = []

        from modules.browser import Browser
        self.browser = Browser(self.test_url)

        from modules.llm import LLM
        self.llm = LLM()
        self.llm.load_model(local_path)

        self.tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "perform_click",
                    "description": "Perform a click action on the web page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "The element on the web page to be clicked."
                            }
                        },
                        "required": ["element"]
                    }
                }
            }
        ]

        self.test_behavior += f" You have access to the following tools: {json.dumps(self.tools_schema)}. If you need to call a function, respond strictly with a JSON object containing 'function' and 'args'."

    def launch(self):
        self.browser.open()

    def explore(self):
        content = self.browser.explore_view()
        print(content)

        messages = [
            {"role": "system", "content": self.test_behavior},
            {"role":"user", "content": "Please develop a complex form that includes various input types and validation rules on the quality of pastas."},
            {"role": "user", "content" : "The following is the ARIA snapshot of the page:"},
            {"role": "user", "content": content},
        ]

        response = self.llm.gen_text(messages, self.tools_schema)

        print(response)
    def perform_click(self, instructions):
        self.browser.click(instructions["element"])

    def report(self):
        print("Report:")
        for note in self.notes:
            print(note)
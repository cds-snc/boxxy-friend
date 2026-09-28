import json
from modules.logger import log

class Mode1:
    def __init__(self, test_url, local_path):
        self.complete_counter = 0
        self.test_url = test_url
        self.test_behavior = "You are an automated testing agent mimicking a user utilizing a screen reader. " \
        "Your goal is to test the whole application for accessibility and functionality." \
        "You are testing GC Forms, a product for creating and managing web forms." \
        "Simulate developing a complex web form with various input types and validation rules." \
        "Be curious, explore, and provide feedback on your findings." \
        "Always perform an action until your task is completed." \
        "When you call the perform_click tool, the 'element' argument MUST be copied verbatim from the ARIA " \
        "snapshot you were given, in the exact form <role> \"<accessible name>\" (for example: " \
        "button \"Design a form Start with a blank form.\"). Use the full accessible name exactly as it appears " \
        "in the snapshot, including any text contributed by child elements. Never paraphrase, shorten, or " \
        "reword the accessible name, and never invent an element that is not present in the snapshot."

        # Initialize notes to keep track of observations during exploration.
        self.messages = []

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
                                "description": "The exact line for the target element as it appears verbatim in the "
                                    "ARIA snapshot, in the form <role> \"<accessible name>\" (e.g. button \"Design a "
                                    "form Start with a blank form.\"). Copy the role and full quoted accessible name "
                                    "character-for-character from the snapshot, including any inner text from child "
                                    "elements. Do not paraphrase, summarize, truncate, or invent text that isn't in "
                                    "the snapshot."
                            }
                        },
                        "required": ["element"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "perform_typing",
                    "description": "Perform a typing action on the web page.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "element": {
                                "type": "string",
                                "description": "The exact line for the target element as it appears verbatim in the "
                                    "ARIA snapshot, in the form <role> \"<accessible name>\" (e.g. button \"Design a "
                                    "form Start with a blank form.\"). Copy the role and full quoted accessible name "
                                    "character-for-character from the snapshot, including any inner text from child "
                                    "elements. Do not paraphrase, summarize, truncate, or invent text that isn't in "
                                    "the snapshot."
                            },
                            "text": {
                                "type": "string",
                                "description": "The text to type into the target element."
                            }
                        },
                        "required": ["element", "text"]
                    }
                }
            }
        ]

        self.test_behavior += f" You have access to the following tools: {json.dumps(self.tools_schema)}. If you need to call a function, respond strictly with a JSON object containing 'function' and 'args'."

    def launch(self):
        self.browser.open()

    def llm_access_content(self):
        content = self.browser.explore_view()

        messages_with_snapshot = self.messages + [
            {"role": "user", "content" : "The following is the ARIA snapshot of the page:"},
            {"role": "user", "content": content},
        ] 

        log("Exploring content...", clear_screen=True)
        log(content)

        response = self.llm.gen_text(messages_with_snapshot, self.tools_schema)

        response_text = response['text']
        if response['thoughts']:
            response_text += response['thoughts']
        
        tools_used = False

        ## Do any tool calls.
        for tool_call in response['tool_calls']:
            tools_used = True
            self.complete_counter = 0 # reset the completion counter whenever a tool is used
            log(tool_call)
            if tool_call['function']['name'] == 'perform_click':
                response_text += self.perform_click(tool_call['function']['arguments'])

            elif tool_call['function']['name'] == 'perform_typing':
                response_text += self.perform_typing(tool_call['function']['arguments'])

        ## Append the latest user message to the conversation history
        self.messages.append({"role": "model", "content": response_text})
        log(response_text)
        
        if tools_used:
            self.continue_explore()
        else:
            self.validate_status()

    def explore(self):
        self.messages = [
            {"role": "system", "content": self.test_behavior},
            {"role":"user", "content": "Please develop a complex form that includes various input types and validation rules on the quality of pastas."},
        ]

        self.llm_access_content()

    def continue_explore(self):
        self.llm_access_content()

    def perform_click(self, instructions):
        return self.browser.click(instructions["element"])

    def perform_typing(self, instructions):
        return self.browser.type(instructions["element"], instructions["text"])

    def validate_status(self):
        self.complete_counter += 1

        if self.complete_counter >= 2: # if we haven't used a tool twice in a row, we're done.
            self.report()
        else:
            self.messages.append({"role": "user", "content": "Use a tool to continue, or report completion."})
            self.llm_access_content()

    def report(self):
        self.messages.append({"role": "user", "content": "Please provide a final report based on the exploration."})
        
        response = self.llm.gen_text(self.messages, self.tools_schema)
        log(f"{response['text']}\r", clear_screen=True)
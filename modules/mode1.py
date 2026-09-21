class Mode1:
    def __init__(self, test_url, local_path):
        self.test_url = test_url
        self.test_behavior = "You are an automated testing agent mimicking a user utilizing a screen reader. Your goal is to test the application for accessibility and functionality."

        # Initialize notes to keep track of observations during exploration.
        self.notes = []

        from modules.browser import Browser
        self.browser = Browser(self.test_url)

        from modules.llm import LLM
        self.llm = LLM()
        self.llm.load_model(local_path)

    def launch(self):
        self.browser.open()

    def explore(self):
        content = self.browser.explore_view()
        print(content)

        messages = [
            {"role": "system", "content": self.test_behavior},
            {"role": "user", "content" : "The following is the ARIA snapshot of the page:"},
            {"role": "user", "content": content},
        ]

        response = self.llm.gen_text(messages)
        self.notes.append(response)
    def report(self):
        print("Report:")
        for note in self.notes:
            print(note)
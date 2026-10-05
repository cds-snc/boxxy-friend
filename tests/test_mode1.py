import unittest
from unittest.mock import patch

from modules.mode1 import Mode1


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def gen_text(self, messages, tools_schema):
        self.calls.append(messages)
        return next(self.responses)


class FakeBrowser:
    def __init__(self, fail_first_click=False):
        self.fail_first_click = fail_first_click
        self.click_count = 0
        self.type_count = 0
        self.selections = []
        self.snapshot = 'button "Continue"'

    def explore_view(self):
        return self.snapshot

    def click(self, element):
        self.click_count += 1
        if self.fail_first_click and self.click_count == 1:
            raise LookupError("target is no longer available")
        if element == 'button "Add Element"':
            self.snapshot = 'dialog "Add element"'
        return f'Clicked {element}'

    def type(self, element, text):
        self.type_count += 1
        return f'Typed into {element}: {text}'

    def use_combobox(self, element, value):
        self.selections.append((element, value))
        return f'Selected {value} in {element}'


def response(tool_calls=None, text=""):
    return {
        "text": text,
        "thoughts": "",
        "tool_calls": tool_calls or [],
    }


def click_call():
    return {
        "function": {
            "name": "perform_click",
            "arguments": {"element": 'button "Continue"'},
        }
    }


def typing_call():
    return {
        "function": {
            "name": "perform_typing",
            "arguments": {"element": 'textbox "Form title"', "text": "Pasta quality"},
        }
    }


class Mode1RecoveryTests(unittest.TestCase):
    def make_mode(self, llm, browser):
        mode = Mode1("https://example.test", llm=llm)
        mode.browser = browser
        mode.MAX_NO_PROGRESS_TURNS = 2
        return mode

    def test_action_error_is_transient_and_model_can_recover(self):
        llm = FakeLLM([
            response([click_call()], text="I will click Continue."),
            response([typing_call()]),
            response(),
            response(),
            response(text="The form could not be verified."),
        ])
        mode = self.make_mode(llm, FakeBrowser(fail_first_click=True))

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertIn("LookupError: target is no longer available", str(llm.calls[1]))
        self.assertNotIn("target is no longer available", str(llm.calls[2]))
        self.assertNotIn("target is no longer available", str(mode.messages))
        self.assertNotIn("I will click Continue.", str(mode.messages))
        self.assertEqual(mode.browser.click_count, 1)
        self.assertEqual(mode.browser.type_count, 1)

    def test_tool_free_turn_is_retried_without_persisting_feedback(self):
        llm = FakeLLM([
            response(text="I need more guidance."),
            response(text="I still need more guidance."),
            response(text="The form could not be verified."),
        ])
        mode = self.make_mode(llm, FakeBrowser())

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertIn("No browser action was performed", str(llm.calls[1]))
        self.assertIn("No browser action was performed", str(llm.calls[2]))
        self.assertNotIn("No browser action was performed", str(mode.messages))
        self.assertNotIn("I need more guidance.", str(mode.messages))
        self.assertNotIn("I still need more guidance.", str(mode.messages))
        self.assertEqual(len(mode.messages), 2)

    def test_repeated_failed_action_is_not_executed_again(self):
        llm = FakeLLM([
            response([click_call()]),
            response([click_call()]),
            response(text="The page is blocked."),
            response(text="The page is blocked."),
            response(text="The page is blocked."),
        ])
        mode = self.make_mode(llm, FakeBrowser(fail_first_click=True))

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertEqual(mode.browser.click_count, 1)
        self.assertIn("already failed while the page was in this unchanged state", str(llm.calls[2]))

    def test_changed_snapshot_is_provided_for_next_action(self):
        llm = FakeLLM([
            response([{
                "function": {
                    "name": "perform_click",
                    "arguments": {"element": 'button "Add Element"'},
                }
            }]),
            response(text="The add-element dialog is open."),
            response(text="The dialog is available."),
            response(text="The dialog remains open."),
        ])
        mode = self.make_mode(llm, FakeBrowser())

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertIn('dialog "Add element"', str(llm.calls[1]))
        self.assertIn("ARIA snapshot changed", str(llm.calls[1]))
        self.assertEqual(mode.browser.click_count, 1)

    def test_guidance_lists_usable_elements_and_hides_blocked_ones(self):
        llm = FakeLLM([
            response([click_call()]),
            response(text="Trying something else."),
            response(text="Done."),
        ])
        browser = FakeBrowser(fail_first_click=True)
        browser.snapshot = (
            '- main:\n'
            '  - button "Continue"\n'
            '  - button "Save" [disabled]\n'
            '  - textbox "Form title"\n'
            '  - heading "Builder" [level=1]'
        )
        mode = self.make_mode(llm, browser)

        with patch("modules.mode1.log"):
            mode.explore()

        first_guidance = llm.calls[0][-1]["content"]
        self.assertIn('button "Continue"', first_guidance)
        self.assertIn('textbox "Form title"', first_guidance)
        self.assertNotIn('button "Save"', first_guidance)
        self.assertNotIn('heading "Builder"', first_guidance)

        retry_guidance = llm.calls[1][-1]["content"]
        usable, _, avoid = retry_guidance.partition("Do not use these elements")
        self.assertNotIn('button "Continue"', usable)
        self.assertIn('button "Continue"', avoid)
        self.assertIn('textbox "Form title"', usable)

    def test_select_option_tool_uses_combobox(self):
        llm = FakeLLM([
            response([{
                "function": {
                    "name": "perform_select_option",
                    "arguments": {"element": 'combobox "Order list:"', "option": "Alphabetically (A-Z)"},
                }
            }]),
            response(text="Done."),
            response(text="Done."),
            response(text="Done."),
        ])
        mode = self.make_mode(llm, FakeBrowser())

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertEqual(mode.browser.selections, [('combobox "Order list:"', "Alphabetically (A-Z)")])
        self.assertIn("Browser action succeeded: perform_select_option", str(mode.messages))

    def test_guidance_groups_combobox_options(self):
        llm = FakeLLM([response(text="Done."), response(text="Done."), response(text="Done.")])
        browser = FakeBrowser()
        browser.snapshot = (
            '- main:\n'
            '  - combobox "Order list:":\n'
            '    - option "As entered" [selected]\n'
            '    - option "Alphabetically (A-Z)"\n'
            '  - combobox "Sort by:": Newest\n'
            '  - listbox "Colours":\n'
            '    - option "Red"\n'
            '  - button "Continue"'
        )
        mode = self.make_mode(llm, browser)

        with patch("modules.mode1.log"):
            mode.explore()

        usable, _, dropdowns = llm.calls[0][-1]["content"].partition("Dropdowns")
        self.assertIn('combobox "Order list:"', usable)
        self.assertNotIn('option "As entered"', usable)
        self.assertIn('option "Red"', usable)
        self.assertIn('button "Continue"', usable)
        self.assertIn('combobox "Order list:" options: "As entered" [selected], "Alphabetically (A-Z)"', dropdowns)
        self.assertIn('combobox "Sort by:" options: (not listed)', dropdowns)


if __name__ == "__main__":
    unittest.main()

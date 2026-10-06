import unittest
from unittest.mock import MagicMock, patch

from modules.browser import Browser
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
        self.page = {"url": "https://example.test/", "title": "Example"}

    def explore_view(self):
        return self.snapshot

    def page_info(self):
        return self.page

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
        # The one-off retry feedback is transient, but the failure stays in the progress log as history.
        self.assertIn("The previous browser action did not succeed", str(llm.calls[1]))
        self.assertNotIn("The previous browser action did not succeed", str(llm.calls[2]))
        self.assertIn("Recent failed actions", llm.calls[2][2]["content"])
        self.assertIn("Step 1 on", llm.calls[2][2]["content"])
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
        self.assertEqual(mode.steps[0]["result"], 'Selected Alphabetically (A-Z) in combobox "Order list:"')
        self.assertIn('result: Selected Alphabetically (A-Z) in combobox "Order list:"', str(llm.calls[1]))

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


class Mode1ProgressTests(unittest.TestCase):
    def make_mode(self, llm, browser, **kwargs):
        mode = Mode1("https://example.test", llm=llm, **kwargs)
        mode.browser = browser
        mode.MAX_NO_PROGRESS_TURNS = 1
        return mode

    @staticmethod
    def progress(call):
        return next(m["content"] for m in call if m["content"].startswith("Progress so far."))

    def test_progress_report_shows_goal_page_intent_result_and_changes(self):
        llm = FakeLLM([
            response([{
                "function": {
                    "name": "perform_click",
                    "arguments": {"element": 'button "Add Element"', "intent": "Open the element picker"},
                }
            }]),
            response(text="Done."),
            response(text="Report."),
        ])
        browser = FakeBrowser()
        browser.snapshot = 'button "Add Element"'
        browser.page = {"url": "https://example.test/builder", "title": "Form builder"}
        mode = self.make_mode(llm, browser, goal="Build a pasta survey.")

        with patch("modules.mode1.log"):
            mode.explore()

        first = self.progress(llm.calls[0])
        self.assertIn("Goal: Build a pasta survey.", first)
        self.assertIn("Completed steps: none yet", first)
        self.assertIn('page "Form builder" (https://example.test/builder)', str(llm.calls[0]))

        second = self.progress(llm.calls[1])
        self.assertIn('Step 1 on "Form builder" (https://example.test/builder)', second)
        self.assertIn("intent: Open the element picker", second)
        self.assertIn('result: Clicked button "Add Element"', second)
        self.assertIn('appeared: dialog "Add element"', second)
        self.assertIn('disappeared: button "Add Element"', second)

        # The final report is grounded in the same progress log.
        self.assertIn("intent: Open the element picker", self.progress(llm.calls[2]))
        self.assertEqual(len(mode.messages), 2)

    def test_intent_does_not_bypass_repeated_failure_block(self):
        def failing_click(intent):
            return {"function": {"name": "perform_click",
                                 "arguments": {"element": 'button "Continue"', "intent": intent}}}

        llm = FakeLLM([
            response([failing_click("Move on")]),
            response([failing_click("Try moving on again")]),
            response(text="Report."),
        ])
        mode = self.make_mode(llm, FakeBrowser(fail_first_click=True))
        mode.MAX_NO_PROGRESS_TURNS = 2

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertEqual(mode.browser.click_count, 1)
        self.assertIn("already failed while the page was in this unchanged state", str(llm.calls[2]))
        self.assertIn("(intent: Move on)", self.progress(llm.calls[1]))

    def test_older_steps_are_summarized_and_failures_capped(self):
        mode = Mode1("https://example.test", llm=FakeLLM([]))
        mode.browser = FakeBrowser()
        with patch("modules.mode1.log"):
            for number in range(1, mode.MAX_RECENT_STEPS + 4):
                mode._record_step("perform_typing", {"intent": f"Fill field {number}"}, f"typed {number}",
                                  "page", "a", "a")
            for number in range(mode.MAX_RECORDED_FAILURES + 3):
                mode._record_failure("page", {}, f"failure {number}")
            report = mode._progress_report()

        earlier, _, recent = report.partition("Most recent steps:")
        self.assertIn("Step 1: Fill field 1 -> typed 1", earlier)
        self.assertNotIn("Step 1 on", recent)
        self.assertIn(f"Step {mode.MAX_RECENT_STEPS + 3} on page", recent)
        self.assertEqual(len(mode.failures), mode.MAX_RECORDED_FAILURES)
        self.assertNotIn("failure 0", report)
        self.assertIn(f"failure {mode.MAX_RECORDED_FAILURES + 2}", report)


def progress_call(**arguments):
    return {"function": {"name": "update_progress", "arguments": arguments}}


class Mode1NotesTests(unittest.TestCase):
    def make_mode(self, llm, browser=None):
        mode = Mode1("https://example.test", llm=llm)
        mode.browser = browser or FakeBrowser()
        mode.MAX_NO_PROGRESS_TURNS = 1
        return mode

    def test_plan_and_findings_are_shown_in_later_turns_and_report(self):
        llm = FakeLLM([
            # Notes listed after the browser action must still be applied.
            response([
                {"function": {"name": "perform_click",
                              "arguments": {"element": 'button "Continue"', "intent": "Start a form"}}},
                progress_call(plan=["Create a blank form", "Add a title", "Add a dropdown"]),
            ]),
            response([
                typing_call(),
                progress_call(done=["[ ] create a blank form"], finding="Continue button has no visible focus ring"),
            ]),
            response(text="Done."),
            response(text="Report."),
        ])
        mode = self.make_mode(llm)

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertEqual(mode.browser.click_count, 1)
        self.assertIn("Your plan: none yet", llm.calls[0][2]["content"])
        self.assertIn("[ ] Create a blank form\n[ ] Add a title\n[ ] Add a dropdown", llm.calls[1][2]["content"])
        later = llm.calls[2][2]["content"]
        self.assertIn("[x] Create a blank form\n[ ] Add a title", later)
        self.assertIn("- Continue button has no visible focus ring", later)
        self.assertIn("- Continue button has no visible focus ring", str(llm.calls[3]))
        self.assertEqual(len(mode.steps), 2)

    def test_notes_only_turn_is_not_progress_and_gets_specific_feedback(self):
        llm = FakeLLM([
            response([progress_call(plan=["Add a title"])]),
            response(text="Report."),
        ])
        mode = self.make_mode(llm)

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertEqual(mode.steps, [])
        self.assertEqual(mode.browser.click_count, 0)
        self.assertIn("Your progress notes were saved, but no browser action was performed", str(llm.calls[1]))

    def test_replacing_plan_keeps_done_items_and_caps_lists(self):
        mode = self.make_mode(FakeLLM([]))
        with patch("modules.mode1.log"):
            mode._update_progress({"plan": ["Add a title", "Add a dropdown"], "done": "Add a title"})
            mode._update_progress({"plan": ["Add a title.", "Add a checkbox", "Add a dropdown"]})
            mode._update_progress({"done": ["Publish the form"]})
            mode._update_progress({"plan": "not a list but a string"})
            for number in range(mode.MAX_FINDINGS + 2):
                mode._update_progress({"finding": f"finding {number}"})
            mode._update_progress({"finding": f"finding {mode.MAX_FINDINGS + 1}"})
            mode._update_progress("garbage")

        self.assertEqual(mode.plan, [{"text": "not a list but a string", "done": False}])
        self.assertEqual(len(mode.findings), mode.MAX_FINDINGS)
        self.assertEqual(mode.findings[-1], f"finding {mode.MAX_FINDINGS + 1}")

        mode.plan = []
        with patch("modules.mode1.log"):
            mode._update_progress({"plan": ["Add a title", "Add a dropdown"], "done": "Add a title"})
            mode._update_progress({"plan": ["Add a title.", "Add a checkbox", "Add a dropdown"]})
            mode._update_progress({"done": ["Publish the form"]})
        self.assertEqual(mode.plan, [
            {"text": "Add a title.", "done": True},
            {"text": "Add a checkbox", "done": False},
            {"text": "Add a dropdown", "done": False},
            {"text": "Publish the form", "done": True},
        ])

    def test_progress_and_prompt_callbacks_receive_what_is_sent(self):
        llm = FakeLLM([
            response([typing_call(), progress_call(plan=["Add a title"], done=["Add a title"])]),
            response(text="Done."),
            response(text="Report."),
        ])
        progress_updates, prompts = [], []
        mode = Mode1("https://example.test", llm=llm, on_progress=progress_updates.append, on_prompt=prompts.append)
        mode.browser = FakeBrowser()
        mode.MAX_NO_PROGRESS_TURNS = 1

        with patch("modules.mode1.log"):
            mode.explore()

        self.assertEqual(prompts, llm.calls)
        self.assertEqual(progress_updates[0], llm.calls[0][2]["content"])
        # The post-turn refresh shows the step and checklist before the next prompt is built.
        self.assertIn("[x] Add a title", progress_updates[1])
        self.assertIn("Step 1 on", progress_updates[1])
        self.assertEqual(progress_updates[-1], llm.calls[-1][2]["content"])


class BrowserTypeTests(unittest.TestCase):
    def make_browser(self, value):
        browser = Browser("https://example.test")
        browser.browser = MagicMock()
        browser.page = MagicMock()
        browser.page.get_by_role.return_value.input_value.return_value = value
        return browser

    def test_type_reports_matching_value(self):
        browser = self.make_browser("Pasta quality")
        result = browser.type('textbox "Form title"', "Pasta quality")
        browser.page.get_by_role.assert_called_with("textbox", name="Form title")
        self.assertTrue(result.endswith("the field now contains exactly that text"))

    def test_type_reports_value_that_differs(self):
        browser = self.make_browser("Pasta")
        result = browser.type('textbox "Form title"', "Pasta quality")
        self.assertIn('the field now contains "Pasta" (differs from what was typed)', result)


if __name__ == "__main__":
    unittest.main()

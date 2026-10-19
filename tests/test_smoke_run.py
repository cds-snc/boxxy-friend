import json
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.mode1 import Mode1, StopRequested
from modules.objective import Objective, ProbeFailed


def call(name, **arguments):
    return {"text": "", "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class SmokeRunTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("modules.mode1.log")
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_mode(self, responses, objective=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        objective = objective or Objective()
        with patch("modules.browser.Browser") as browser_class:
            mode = Mode1(objective.url, objective=objective, llm=MagicMock(),
                         output_dir=Path(temp.name) / "run")
        mode.llm.gen_text.side_effect = responses
        mode.browser = browser_class.return_value
        mode.browser.explore_view.return_value = '- button "Continue"'
        mode.browser.page_info.return_value = {"url": objective.url, "title": "GC Forms"}
        mode.browser.check_test_inputs.return_value = "Observed check"
        mode.browser.download_form.return_value = str(mode.output_dir / "form.json")
        mode.launch()
        return mode

    def test_completion_requires_ordered_checks_and_download(self):
        mode = self.make_mode([
            call("check_required_empty"), call("check_valid_input"),
            call("download_form", element='button "Download form file"', intent="Save JSON"),
        ])
        mode.explore()
        self.assertEqual(mode.outcome, "completed")
        self.assertEqual([item["check"] for item in mode.checkpoints], ["required_empty", "valid_input"])
        saved = json.loads((mode.output_dir / "result.json").read_text())
        self.assertEqual(saved["outcome"], "completed")
        self.assertEqual(saved["objective"]["element_count"], 3)
        self.assertEqual(saved["export"], mode.export_path)

    def test_export_before_checks_and_check_failure_halt_immediately(self):
        for responses in (
            [call("download_form", element="Download")],
            [call("check_valid_input")],
            [call("check_required_empty")],
        ):
            mode = self.make_mode(responses)
            mode.browser.check_test_inputs.side_effect = ProbeFailed("Required check failed")
            mode.explore()
            self.assertEqual(mode.outcome, "blocked")
            self.assertEqual(mode.llm.gen_text.call_count, 1)
            mode.browser.download_form.assert_not_called()

    def test_model_notes_are_not_success_evidence(self):
        mode = self.make_mode([
            call("update_progress", plan=["Export"], done=["Export"]),
            {"text": "Success!", "tool_calls": []},
        ], replace(Objective(), no_progress_turns=2))
        mode.explore()
        self.assertEqual(mode.outcome, "blocked")
        self.assertEqual(mode.checkpoints, [])

    def test_action_limit_counts_failed_attempts_and_stops_before_next_action(self):
        mode = self.make_mode([
            call("perform_click", element='button "Continue"'),
            call("perform_click", element='button "Other"'),
        ], replace(Objective(), max_actions=1))
        def fail_click(*_):
            mode._before_action()
            raise LookupError("Target unavailable")

        mode.browser.click.side_effect = fail_click
        mode.explore()
        self.assertEqual(mode.action_count, 1)
        self.assertEqual(mode.outcome, "budget_exhausted")
        self.assertEqual(mode.browser.click.call_count, 1)
        self.assertEqual(mode.llm.gen_text.call_count, 1)
        self.assertIn("Target unavailable", mode.failures[0])

    def test_elapsed_budget_and_user_stop_are_persisted(self):
        mode = self.make_mode([])
        with patch("modules.mode1.time.monotonic", return_value=mode.started_at + 900):
            mode.explore()
        self.assertEqual(mode.outcome, "budget_exhausted")
        mode = self.make_mode([])
        mode.stop_event = threading.Event()
        mode.stop_event.set()
        with self.assertRaises(StopRequested):
            mode.explore()
        self.assertEqual(mode.outcome, "stopped_by_user")

    def test_scope_violation_and_explicit_blocker_are_not_retried(self):
        for scope_error in (False, True):
            mode = self.make_mode([call("halt_blocked", reason="No Test mode available")])
            if scope_error:
                mode.browser.check_scope.side_effect = ProbeFailed("Scope violation")
            mode.explore()
            self.assertEqual(mode.outcome, "blocked")
            self.assertLessEqual(mode.llm.gen_text.call_count, 1)

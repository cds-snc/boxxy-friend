import queue
import unittest
from unittest.mock import MagicMock, patch

from modules.gui import BoxxyGui, HAIKU_PROMPT


class ModelReadinessTests(unittest.TestCase):
    def make_gui(self):
        gui = BoxxyGui.__new__(BoxxyGui)
        gui.root = MagicMock()
        gui.closed = False
        gui.main_open = False
        gui.llm = None
        gui.model_path = "/models/selected.gguf"
        gui.test_url = "https://example.test"
        gui.events = queue.Queue()
        gui.model_loading = True
        gui.loading_ready = False
        gui.loading_start = gui.loading_phase_start = 100
        gui.loading_phase = "load"
        for name in (
            "loading_screen", "loading_progress", "loading_status_var", "loading_time_var",
            "loading_results", "retry_loading_button", "choose_model_button", "loading_ok_button",
        ):
            setattr(gui, name, MagicMock())
        return gui

    def test_load_completion_starts_haiku_phase_without_enabling_main(self):
        gui = self.make_gui()
        with patch("modules.gui.time.perf_counter", return_value=110):
            gui._handle_event("model_loaded", (10.0,))
        self.assertEqual(gui.loading_phase, "haiku")
        self.assertEqual(gui.loading_phase_start, 110)
        self.assertTrue(gui.model_loading)
        self.assertFalse(gui.loading_ready)
        self.assertIsNone(gui.llm)
        gui.loading_ok_button.configure.assert_not_called()
        gui._enter_main()
        gui.loading_screen.destroy.assert_not_called()

    def test_results_wait_for_okay_before_building_main_ui(self):
        gui = self.make_gui()
        llm = MagicMock()
        with (
            patch.object(gui, "_position_window"),
            patch.object(gui, "_build_toolbar") as toolbar,
            patch.object(gui, "_build_error_bar"),
            patch.object(gui, "_build_status_bar"),
            patch.object(gui, "_build_main_panes") as panes,
            patch.object(gui, "_refresh_buttons") as refresh,
            patch.object(gui, "_set_text") as text,
            patch("modules.gui.logger.add_listener"),
        ):
            gui._handle_event("model_ready", (llm, "A test haiku", 12.5, 3.25))
            self.assertIs(gui.llm, llm)
            self.assertFalse(gui.main_open)
            self.assertFalse(gui.model_loading)
            self.assertTrue(gui.loading_ready)
            toolbar.assert_not_called()
            panes.assert_not_called()
            gui.loading_ok_button.configure.assert_called_once_with(state="normal")
            gui.loading_progress.configure.assert_called_once_with(value=100)
            result = text.call_args.args[1]
            for expected in ("A test haiku", "Model load: 12.50s", "Haiku generation: 3.25s", "Total: 15.75s"):
                self.assertIn(expected, result)
            screen = gui.loading_screen
            gui._enter_main()
            gui._enter_main()
        screen.destroy.assert_called_once()
        self.assertIsNone(gui.loading_screen)
        self.assertTrue(gui.main_open)
        toolbar.assert_called_once_with(gui.test_url)
        panes.assert_called_once()
        refresh.assert_called_once()

    def test_estimated_progress_is_capped_until_each_phase_finishes(self):
        for phase, elapsed, expected_progress, remaining in (
            ("load", 0, 0, "~90s"),
            ("load", 30, 25, "~60s"),
            ("load", 120, 45, "Taking longer than estimated"),
            ("haiku", 0, 50, "~30s"),
            ("haiku", 15, 75, "~15s"),
            ("haiku", 120, 95, "Taking longer than estimated"),
        ):
            with self.subTest(phase=phase, elapsed=elapsed):
                gui = self.make_gui()
                gui.loading_phase = phase
                with patch("modules.gui.time.perf_counter", return_value=100 + elapsed):
                    gui._update_loading_progress()
                gui.loading_progress.configure.assert_called_once_with(value=expected_progress)
                label = gui.loading_time_var.set.call_args.args[0]
                self.assertIn("Estimated progress", label)
                self.assertIn(f"Elapsed: {elapsed:.1f}s", label)
                self.assertIn(remaining, label)

    def test_loading_cannot_be_started_twice_or_after_readiness(self):
        for loading, ready, screen, closed in (
            (True, False, MagicMock(), False),
            (False, True, MagicMock(), False),
            (False, False, None, False),
            (False, False, MagicMock(), True),
        ):
            with self.subTest(loading=loading, ready=ready, closed=closed):
                gui = self.make_gui()
                gui.model_loading, gui.loading_ready = loading, ready
                gui.loading_screen, gui.closed = screen, closed
                with patch("modules.gui.threading.Thread") as thread:
                    gui.load_model_async()
                thread.assert_not_called()

    def test_retry_resets_progress_errors_and_buttons_before_starting_worker(self):
        gui = self.make_gui()
        gui.model_loading = False
        with (
            patch("modules.gui.time.perf_counter", return_value=200),
            patch("modules.gui.threading.Thread") as thread,
            patch.object(gui, "_set_text") as text,
        ):
            gui.load_model_async()
        self.assertTrue(gui.model_loading)
        self.assertEqual(gui.loading_phase, "load")
        self.assertEqual(gui.loading_start, 200)
        text.assert_called_once_with(gui.loading_results, "")
        for button in (gui.loading_ok_button, gui.retry_loading_button, gui.choose_model_button):
            button.configure.assert_called_once_with(state="disabled")
        thread.assert_called_once_with(
            target=gui._load_model_worker, name="boxxy-model-loader", daemon=True
        )
        thread.return_value.start.assert_called_once()

    def test_worker_posts_measured_results_without_calling_tk(self):
        gui = self.make_gui()
        with (
            patch("modules.llm.LLM") as llm,
            patch("modules.gui.time.perf_counter", side_effect=[100, 110, 111, 114]),
        ):
            llm.return_value.gen_text.return_value = {"text": "  A test haiku  "}
            gui._load_model_worker()
        llm.return_value.load_model.assert_called_once_with(gui.model_path)
        llm.return_value.gen_text.assert_called_once_with(HAIKU_PROMPT)
        self.assertEqual(gui.events.get_nowait(), ("model_loaded", (10,)))
        self.assertEqual(
            gui.events.get_nowait(), ("model_ready", (llm.return_value, "A test haiku", 10, 3))
        )
        self.assertTrue(gui.events.empty())
        self.assertEqual(gui.root.mock_calls, [])
        self.assertEqual(gui.loading_results.mock_calls, [])

    def test_load_and_haiku_failures_offer_retry_without_enabling_okay(self):
        for failure in ("load", "haiku", "empty"):
            with self.subTest(failure=failure):
                gui = self.make_gui()
                with (
                    patch("modules.llm.LLM") as llm,
                    patch.object(gui, "_set_text") as text,
                ):
                    if failure == "load":
                        llm.return_value.load_model.side_effect = RuntimeError("load failed")
                    elif failure == "haiku":
                        llm.return_value.gen_text.side_effect = RuntimeError("generation failed")
                    else:
                        llm.return_value.gen_text.return_value = {"text": " "}
                    gui._load_model_worker()
                    gui._poll_events()
                self.assertFalse(gui.loading_ready)
                self.assertFalse(gui.model_loading)
                self.assertIsNone(gui.llm)
                self.assertFalse(gui.main_open)
                gui.loading_ok_button.configure.assert_not_called()
                gui.retry_loading_button.configure.assert_called_once_with(state="normal")
                gui.choose_model_button.configure.assert_called_once_with(state="normal")
                self.assertIn("failed", text.call_args.args[1].lower())

    def test_can_return_to_selector_only_after_failure(self):
        gui = self.make_gui()
        with patch.object(gui, "_build_model_selector") as selector:
            gui._return_to_selector()
            selector.assert_not_called()
            gui.model_loading = False
            screen = gui.loading_screen
            gui._return_to_selector()
        screen.destroy.assert_called_once()
        selector.assert_called_once()
        self.assertIsNone(gui.loading_screen)

    def test_close_stops_polling_without_touching_destroyed_widgets(self):
        gui = self.make_gui()
        gui.closed = True
        gui.post("model_ready", MagicMock(), "haiku", 10, 3)
        gui._poll_events()
        self.assertEqual(gui.root.mock_calls, [])
        self.assertEqual(gui.loading_ok_button.mock_calls, [])

    def test_start_does_nothing_before_main_ui_opens(self):
        gui = self.make_gui()
        gui.llm = MagicMock()
        gui.start_run()
        self.assertEqual(gui.root.mock_calls, [])


if __name__ == "__main__":
    unittest.main()

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from modules.gui import BoxxyGui
from modules.mode1 import Mode1
from modules.window_layout import WindowBounds, desktop_bounds


class WindowLayoutTests(unittest.TestCase):
    def test_halves_cover_desktop_without_overlap_or_gaps(self):
        for desktop in (
            WindowBounds(0, 33, 1512, 862),
            WindowBounds(40, 24, 1365, 744),
            WindowBounds(0, 0, 2, 1),
        ):
            with self.subTest(desktop=desktop):
                left, right = desktop.split()
                self.assertEqual(left.x, desktop.x)
                self.assertEqual(right.x, left.x + left.width)
                self.assertEqual(right.x + right.width, desktop.x + desktop.width)
                self.assertEqual(left.y, right.y)
                self.assertEqual(left.y, desktop.y)
                self.assertEqual(left.height, desktop.height)
                self.assertEqual(right.height, desktop.height)
                self.assertLessEqual(abs(left.width - right.width), 1)

    def test_invalid_dimensions_are_rejected(self):
        for width, height in ((0, 800), (800, 0), (-1, 800)):
            with self.assertRaises(ValueError):
                WindowBounds(0, 0, width, height)
        with self.assertRaises(ValueError):
            WindowBounds(0, 0, 1, 800).split()

    @patch("modules.window_layout.sys.platform", "darwin")
    @patch("modules.window_layout.subprocess.run")
    def test_macos_uses_visible_frame(self, run):
        run.return_value.stdout = '{"x":0,"y":33,"width":1512,"height":862}'
        self.assertEqual(desktop_bounds(1512, 982), WindowBounds(0, 33, 1512, 862))
        self.assertEqual(run.call_args.kwargs["timeout"], 5)
        self.assertTrue(run.call_args.kwargs["check"])

    @patch("modules.window_layout.sys.platform", "darwin")
    @patch("modules.window_layout.subprocess.run")
    def test_macos_errors_are_not_silently_ignored(self, run):
        for result in ('not json', '{"x":0}', '{"x":0,"y":0,"width":0,"height":800}'):
            run.return_value.stdout = result
            with self.assertRaisesRegex(RuntimeError, "usable desktop area"):
                desktop_bounds(1512, 982)
        run.side_effect = subprocess.TimeoutExpired("osascript", 5)
        with self.assertRaisesRegex(RuntimeError, "usable desktop area"):
            desktop_bounds(1512, 982)

    @patch("modules.window_layout.sys.platform", "linux")
    @patch("modules.window_layout.subprocess.run")
    def test_other_platforms_use_tk_screen_dimensions(self, run):
        self.assertEqual(desktop_bounds(1920, 1080), WindowBounds(0, 0, 1920, 1080))
        run.assert_not_called()

    def test_gui_accounts_for_title_bar_and_clamps_minimum_size(self):
        gui = BoxxyGui.__new__(BoxxyGui)
        gui.gui_bounds = WindowBounds(0, 33, 512, 400)
        gui.root = MagicMock()
        gui.root.winfo_rooty.return_value = 65
        gui.root.winfo_y.return_value = 33
        gui._position_window()
        self.assertEqual(gui.root.geometry.call_args.args, ("512x368+0+33",))
        gui.root.minsize.assert_called_once_with(512, 350)

    @patch("modules.browser.Browser")
    def test_mode_passes_browser_bounds_through(self, browser):
        bounds = WindowBounds(756, 33, 756, 862)
        Mode1("https://example.test", llm=MagicMock(), browser_bounds=bounds)
        browser.assert_called_once_with("https://example.test", window_bounds=bounds)

    @patch("modules.mode1.Mode1")
    @patch("modules.gui.logger.log")
    def test_gui_worker_passes_cached_bounds_without_calling_tk(self, log, mode):
        gui = BoxxyGui.__new__(BoxxyGui)
        gui.browser_bounds = WindowBounds(756, 33, 756, 862)
        gui.llm = MagicMock()
        gui.stop_event = MagicMock()
        gui.post = MagicMock()
        gui.root = MagicMock()
        gui._run_worker("https://example.test")
        self.assertEqual(mode.call_args.kwargs["browser_bounds"], gui.browser_bounds)
        gui.root.assert_not_called()
        self.assertEqual(gui.root.mock_calls, [])
        mode.return_value.launch.assert_called_once_with()
        mode.return_value.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

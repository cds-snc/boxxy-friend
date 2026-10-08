import unittest
from unittest.mock import MagicMock, patch

from modules.browser import Browser
from modules.window_layout import WindowBounds


class BrowserWindowTests(unittest.TestCase):
    @patch("modules.browser.sync_playwright")
    def test_default_launch_is_unchanged(self, playwright):
        browser = Browser("https://example.test")
        browser.open()
        chromium = playwright.return_value.start.return_value.chromium
        chromium.launch.assert_called_once_with(headless=False)
        chromium.launch.return_value.new_page.assert_called_once_with()
        browser.page.goto.assert_called_once_with("https://example.test")

    @patch("modules.browser.sync_playwright")
    def test_positioned_launch_uses_native_window_and_responsive_viewport(self, playwright):
        browser = Browser("https://example.test", WindowBounds(756, 33, 756, 862))
        browser.open()
        chromium = playwright.return_value.start.return_value.chromium
        chromium.launch.assert_called_once_with(
            headless=False, args=["--window-position=756,33", "--window-size=756,862"]
        )
        chromium.launch.return_value.new_page.assert_called_once_with(no_viewport=True)
        session = browser.page.context.new_cdp_session.return_value
        session.send.assert_any_call("Browser.setWindowBounds", {
            "windowId": session.send.return_value["windowId"],
            "bounds": {
                "left": 756, "top": 33, "width": 756, "height": 862, "windowState": "normal",
            },
        })
        session.detach.assert_called_once_with()
        browser.page.set_default_timeout.assert_called_once_with(Browser.ACTION_TIMEOUT_MS)
        browser.page.goto.assert_called_once_with("https://example.test")

    @patch("modules.browser.sync_playwright")
    def test_positioning_failure_is_propagated_and_session_detached(self, playwright):
        browser = Browser("https://example.test", WindowBounds(756, 33, 756, 862))
        page = playwright.return_value.start.return_value.chromium.launch.return_value.new_page.return_value
        session = page.context.new_cdp_session.return_value
        session.send.side_effect = RuntimeError("Cannot position window")
        with self.assertRaisesRegex(RuntimeError, "Cannot position window"):
            browser.open()
        session.detach.assert_called_once_with()
        page.goto.assert_not_called()


class BrowserElementCleanupTests(unittest.TestCase):
    def make_browser(self):
        browser = Browser("https://example.test")
        browser.browser = MagicMock()
        browser.page = MagicMock()
        return browser

    def test_click_uses_role_for_escaped_or_comma_suffixed_elements(self):
        for element in (
            'button "Design a form Start with a blank form."',
            'button "Design a form Start with a blank form.",',
            r'button \"Design a form Start with a blank form.\"',
            r'button \"Design a form Start with a blank form.\",',
        ):
            with self.subTest(element=element):
                browser = self.make_browser()
                browser.click(element)
                browser.page.get_by_role.assert_called_once_with(
                    "button", name="Design a form Start with a blank form."
                )
                browser.page.get_by_role.return_value.click.assert_called_once_with()
                browser.page.get_by_text.assert_not_called()

    def test_typing_cleans_target_but_preserves_typed_text(self):
        browser = self.make_browser()
        text = r'Keep \"quotes\", C:\pasta, and commas,'
        browser.type(r'textbox \"Form title\",', text)
        browser.page.get_by_role.assert_called_once_with("textbox", name="Form title")
        browser.page.get_by_role.return_value.fill.assert_called_once_with(text)

    def test_combobox_cleans_target_and_snapshot_option(self):
        browser = self.make_browser()
        browser.page.get_by_role.return_value.first.evaluate.return_value = "select"
        browser._select_native_option = MagicMock(return_value="As entered")
        browser.use_combobox(r'combobox \"Format\": current value,',
                             r'option \"As entered\" [selected],')
        browser.page.get_by_role.assert_called_once_with("combobox", name="Format", exact=True)
        browser._select_native_option.assert_called_once_with(
            browser.page.get_by_role.return_value.first, "As entered"
        )

    def test_cleanup_preserves_names_and_plain_text(self):
        for element in (
            r'button "C:\pasta, please"',
            r'button "Say \"hello\""',
            r'C:\pasta, please,',
            r'\"Plain text\",',
            'button "Comma,"',
            '- combobox "Format": As entered',
            'option "As entered" [selected]',
        ):
            with self.subTest(element=element):
                self.assertEqual(Browser._clean_element(element), element)


if __name__ == "__main__":
    unittest.main()

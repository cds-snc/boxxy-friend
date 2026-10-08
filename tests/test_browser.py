import unittest
from unittest.mock import MagicMock

from modules.browser import Browser


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

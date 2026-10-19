import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from playwright.sync_api import Error, sync_playwright

from modules.browser import Browser
from modules.objective import Objective, ProbeFailed
from test_objective import exported_form


class SmokeBrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.chromium = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.chromium.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.chromium.new_context()
        self.addCleanup(self.context.close)
        self.objective = Objective(url="https://example.test/en/form-builder")
        self.browser = Browser(self.objective.url)
        self.browser.browser = self.chromium
        self.browser.page = self.context.new_page()
        self.browser.page.set_default_timeout(500)
        self.action = MagicMock()
        self.browser.configure_scope(self.objective, self.action)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def load(self, export=None, validation=True):
        data = json.dumps(export if export is not None else exported_form(self.objective))
        html = """
        <form onsubmit="event.preventDefault(); const first=this.querySelector('input');
            first.setAttribute('aria-invalid', String(!first.value));">
          <label>Boxxy question 1<input></label>
          <label>Boxxy question 2<textarea></textarea></label>
          <fieldset><legend>Boxxy question 3</legend>
            <label><input type="radio" name="choice" value="a">Option A</label>
            <label><input type="radio" name="choice" value="b">Option B</label>
          </fieldset>
          <button>Submit</button>
        </form>
        <button id="download">Download form file</button>
        <h1>Your form has been submitted</h1>
        <script>
          const data = EXPORT_DATA;
          document.getElementById('download').onclick = () => {
            const link = document.createElement('a');
            link.href = URL.createObjectURL(new Blob([JSON.stringify(data)], {type:'application/json'}));
            link.download = 'form.json'; link.click();
          };
        </script>
        """.replace("EXPORT_DATA", data)
        if not validation:
            html = html.replace(
                "first.setAttribute('aria-invalid', String(!first.value));", ""
            )
        self.browser.page.route("**/*", lambda route: route.fulfill(body=html, content_type="text/html"))
        self.browser.page.goto(self.objective.url + "/0000/preview")
        self.browser.page.unroute("**/*")

    def test_real_browser_verifies_checks_and_saves_validated_download(self):
        self.load()
        evidence = self.browser.check_test_inputs(self.objective, valid=False)
        self.assertIn("rejected", evidence)
        evidence = self.browser.check_test_inputs(self.objective, valid=True)
        self.assertIn("no field validation errors", evidence)
        path = self.browser.download_form(
            'button "Download form file"', self.objective, self.temp.name
        )
        data = json.loads(Path(path).read_text())
        self.objective.validate_export(data)
        self.assertEqual(self.action.call_count, 6)

    def test_static_confirmation_does_not_pass_required_check(self):
        self.load(validation=False)
        with self.assertRaisesRegex(ProbeFailed, "static confirmation preview is not success"):
            self.browser.check_test_inputs(self.objective, valid=False)

    def test_wrong_export_fails_after_real_download(self):
        self.load(export={"elements": []})
        with self.assertRaises(ProbeFailed):
            self.browser.download_form('button "Download form file"', self.objective, self.temp.name)
        self.assertTrue((Path(self.temp.name) / "form.json").is_file())

    def test_scope_guard_blocks_navigation_before_network_request(self):
        self.load()
        self.context.route("**/*", self.browser._guard_navigation)
        with self.assertRaises(Error):
            self.browser.page.goto("https://outside.test/en/form-builder")
        with self.assertRaisesRegex(ProbeFailed, "Blocked out-of-scope"):
            self.browser.check_scope()

    def test_duplicate_question_inputs_can_be_targeted_by_index(self):
        self.load()
        self.browser.page.set_content('<label>Question<input></label><label>Question<input></label>')
        self.browser.type('textbox "Question"', "Second question", index=1)
        inputs = self.browser.page.get_by_role("textbox", name="Question")
        self.assertEqual(inputs.nth(0).input_value(), "")
        self.assertEqual(inputs.nth(1).input_value(), "Second question")


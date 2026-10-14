import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import setup


class SetupTests(unittest.TestCase):
    def test_defaults_download_only_ornith_q4(self):
        with patch.object(sys, "argv", ["setup.py"]):
            args = setup.parse_args()
        self.assertEqual(args.repo_id, "ornith-ai/Ornith-1.5-9B-GGUF")
        self.assertEqual(args.output_dir, "models")
        self.assertEqual(args.allow_patterns, ["Ornith-1.5-9B-Q4_K_M.gguf"])

    def test_download_options_are_forwarded(self):
        hub = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model"
            argv = [
                "setup.py", "--repo-id", "example/model", "--output-dir", str(output),
                "--revision", "v1", "--token", "test-token", "--allow-patterns", "*",
            ]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertEqual(setup.main(), 0)
            self.assertTrue(output.is_dir())
            hub.snapshot_download.assert_called_once_with(
                repo_id="example/model",
                repo_type="model",
                revision="v1",
                local_dir=str(output),
                allow_patterns=["*"],
                token="test-token",
            )

    def test_default_download_is_filtered(self):
        hub = MagicMock()
        with tempfile.TemporaryDirectory() as directory:
            argv = ["setup.py", "--output-dir", directory]
            with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"huggingface_hub": hub}):
                self.assertEqual(setup.main(), 0)
        options = hub.snapshot_download.call_args.kwargs
        self.assertEqual(options["repo_id"], setup.DEFAULT_REPO_ID)
        self.assertEqual(options["allow_patterns"], [setup.DEFAULT_FILENAME])

    def test_smoke_script_uses_default_model_and_browser_url(self):
        llm_module = MagicMock()
        browser_module = MagicMock()
        root = Path(__file__).resolve().parent.parent
        with patch.dict(sys.modules, {
            "modules.llm": llm_module,
            "modules.browser": browser_module,
        }):
            runpy.run_path(str(root / "test.py"), run_name="__main__")
        llm_module.LLM.return_value.load_model.assert_called_once_with(
            root / setup.DEFAULT_OUTPUT_DIR / setup.DEFAULT_FILENAME
        )
        llm_module.LLM.return_value.gen_text.assert_called_once()
        browser_module.Browser.assert_called_once_with("https://example.com")
        browser_module.Browser.return_value.test_browser.assert_called_once_with()

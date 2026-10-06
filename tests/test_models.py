import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.gui import BoxxyGui
from modules.models import discover_models, validate_model


class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models_dir = Path(self.temp.name)

    def make_model(self, name="model"):
        path = self.models_dir / name
        path.mkdir()
        for filename, data in (
            ("config.json", {"model_type": "gemma4"}),
            ("processor_config.json", {"processor_class": "Gemma4Processor"}),
            ("tokenizer_config.json", {"tokenizer_class": "TokenizersBackend"}),
        ):
            (path / filename).write_text(json.dumps(data), encoding="utf-8")
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")
        return path

    def test_lists_complete_models_sorted_and_reports_incomplete_folders(self):
        second = self.make_model("Zulu")
        first = self.make_model("alpha")
        self.make_model(".hidden")
        (self.models_dir / "incomplete").mkdir()
        (self.models_dir / "not-a-model.txt").write_text("ignored", encoding="utf-8")
        models, rejected = discover_models(self.models_dir)
        self.assertEqual(models, [first, second])
        self.assertEqual(len(rejected), 1)
        self.assertIn("incomplete:", rejected[0])

    def test_missing_directory_reports_an_error(self):
        with self.assertRaises(FileNotFoundError):
            discover_models(self.models_dir / "missing")

    def test_empty_directory_has_no_models(self):
        self.assertEqual(discover_models(self.models_dir), ([], []))

    def test_invalid_configs_are_rejected(self):
        path = self.make_model()
        for data in ("{", "[]", "{}", '{"architectures": []}'):
            with self.subTest(data=data):
                (path / "config.json").write_text(data, encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_model(path)

    def test_missing_tokenizer_and_empty_weights_are_rejected(self):
        path = self.make_model()
        (path / "model.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Missing model weights"):
            validate_model(path)
        (path / "model.safetensors").write_bytes(b"weights")
        (path / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing tokenizer"):
            validate_model(path)

    def test_pytorch_weights_are_supported(self):
        path = self.make_model()
        (path / "model.safetensors").rename(path / "pytorch_model.bin")
        validate_model(path)

    def test_sharded_weights_require_every_shard_inside_model_folder(self):
        path = self.make_model()
        (path / "model.safetensors").unlink()
        index = path / "model.safetensors.index.json"
        for shards, expected in (
            ({}, "weight_map"),
            ({"a": "missing.safetensors"}, "Missing or empty weight shard"),
            ({"a": "../outside.safetensors"}, "outside the model folder"),
            ({"a": 123}, "invalid shard name"),
        ):
            with self.subTest(shards=shards):
                index.write_text(json.dumps({"weight_map": shards}), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    validate_model(path)
        (path / "part.safetensors").write_bytes(b"weights")
        index.write_text(json.dumps({"weight_map": {"a": "part.safetensors"}}), encoding="utf-8")
        validate_model(path)


class ModelSelectorTests(unittest.TestCase):
    def test_startup_only_builds_selector_without_loading_model(self):
        with (
            patch("modules.gui.tk.Tk"),
            patch.object(BoxxyGui, "_build_model_selector") as selector,
            patch.object(BoxxyGui, "_build_toolbar") as toolbar,
            patch.object(BoxxyGui, "load_model_async") as load,
        ):
            gui = BoxxyGui()
        selector.assert_called_once()
        toolbar.assert_not_called()
        load.assert_not_called()
        self.assertFalse(gui.main_open)
        self.assertIsNone(gui.model_path)
        self.assertIsNone(gui.llm)

    def make_gui(self):
        with patch("modules.gui.tk.Tk"), patch.object(BoxxyGui, "_build_model_selector"):
            gui = BoxxyGui()
        gui.selector = MagicMock()
        gui.model_var = MagicMock()
        gui.model_var.get.return_value = "selected"
        gui.available_models = {"selected": Path("/models/selected")}
        return gui

    def test_selected_path_is_used_and_loading_is_scheduled_after_main_build(self):
        gui = self.make_gui()
        with (
            patch("modules.gui.validate_model") as validate,
            patch("modules.gui.logger.add_listener"),
            patch.object(gui, "_build_toolbar") as toolbar,
            patch.object(gui, "_build_error_bar"),
            patch.object(gui, "_build_status_bar"),
            patch.object(gui, "_build_main_panes") as panes,
            patch.object(gui, "_install_exception_hooks"),
        ):
            gui._open_main()
            gui._open_main()
        validate.assert_called_once_with(Path("/models/selected"))
        toolbar.assert_called_once_with(gui.test_url)
        panes.assert_called_once()
        gui.selector.destroy.assert_called_once()
        self.assertTrue(gui.main_open)
        self.assertEqual(gui.model_path, "/models/selected")
        gui.root.after.assert_any_call(50, gui.load_model_async)

    def test_model_removed_before_open_keeps_selector_and_reports_error(self):
        gui = self.make_gui()
        with (
            patch("modules.gui.validate_model", side_effect=FileNotFoundError("model removed")),
            patch.object(gui, "_refresh_models") as refresh,
            patch.object(gui, "_show_error") as error,
        ):
            gui._open_main()
        refresh.assert_called_once()
        error.assert_called_once_with("Invalid model", "model removed")
        gui.selector.destroy.assert_not_called()
        self.assertFalse(gui.main_open)

    def test_cancel_closes_without_starting_workers(self):
        gui = self.make_gui()
        with patch("modules.gui.logger.remove_listener"), patch.object(gui, "load_model_async") as load:
            gui.on_close()
        self.assertTrue(gui.closed)
        gui.root.destroy.assert_called_once()
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()

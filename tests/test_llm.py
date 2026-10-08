import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from modules.llm import LLM


TOOLS = [{
    "type": "function",
    "function": {
        "name": "perform_click",
        "description": "Click an element.",
        "parameters": {
            "type": "object",
            "properties": {"element": {"type": "string"}},
            "required": ["element"],
        },
    },
}]
MESSAGES = [{"role": "user", "content": "Click Continue."}]


def completion(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class GGUFLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "model.gguf"
        self.path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 1, 1) + b"data")

    def test_loads_file_or_single_model_folder_without_transformers(self):
        for path in (self.path, self.folder):
            with self.subTest(path=path):
                backend = MagicMock()
                llm = LLM()
                with (
                    patch.dict(sys.modules, {"llama_cpp": backend}),
                    patch("modules.llm.AutoProcessor") as processor,
                    patch("modules.llm.AutoModelForMultimodalLM") as model,
                ):
                    llm.load_model(str(path))
                backend.Llama.assert_called_once_with(
                    model_path=str(self.path), n_ctx=32000, n_gpu_layers=-1, verbose=False
                )
                self.assertIs(llm.gguf_model, backend.Llama.return_value)
                processor.from_pretrained.assert_not_called()
                model.from_pretrained.assert_not_called()

    def test_missing_backend_has_actionable_error(self):
        with patch.dict(sys.modules, {"llama_cpp": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install llama-cpp-python"):
                LLM().load_model(str(self.path))

    def test_native_load_errors_are_not_hidden(self):
        backend = MagicMock()
        backend.Llama.side_effect = ValueError("unsupported architecture")
        with patch.dict(sys.modules, {"llama_cpp": backend}):
            with self.assertRaisesRegex(ValueError, "unsupported architecture"):
                LLM().load_model(str(self.path))

    def test_transformers_loading_is_unchanged_and_clears_gguf_backend(self):
        llm = LLM()
        llm.gguf_model = MagicMock()
        with (
            patch("modules.llm.AutoProcessor") as processor,
            patch("modules.llm.AutoModelForMultimodalLM") as model,
        ):
            llm.load_model(str(self.folder / "transformers"))
        processor.from_pretrained.assert_called_once_with(str(self.folder / "transformers"))
        model.from_pretrained.assert_called_once_with(str(self.folder / "transformers"), dtype="auto")
        model.from_pretrained.return_value.to.assert_called_once_with("mps")
        self.assertIsNone(llm.gguf_model)


class GGUFGenerationTests(unittest.TestCase):
    def setUp(self):
        self.llm = LLM()
        self.llm.gguf_model = MagicMock()

    def test_plain_generation_preserves_response_contract_and_normalizes_role(self):
        messages = [{"role": "model", "content": "Earlier answer"}, *MESSAGES]
        self.llm.gguf_model.create_chat_completion.return_value = completion("A haiku")
        self.assertEqual(self.llm.gen_text(messages), {
            "raw": "A haiku", "thoughts": "", "text": "A haiku", "tool_calls": [],
        })
        self.llm.gguf_model.create_chat_completion.assert_called_once_with(
            messages=[{"role": "assistant", "content": "Earlier answer"}, *MESSAGES], max_tokens=2048
        )
        self.assertEqual(messages[0]["role"], "model")

    def test_tools_use_schema_and_return_argument_objects(self):
        result = {
            "thoughts": "Continue the form",
            "text": "",
            "tool_calls": [{
                "function": {"name": "perform_click", "arguments": {"element": 'button "Continue"'}}
            }],
        }
        raw = json.dumps(result)
        self.llm.gguf_model.create_chat_completion.return_value = completion(raw)
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS), {**result, "raw": raw})
        options = self.llm.gguf_model.create_chat_completion.call_args.kwargs
        schema = options["response_format"]["schema"]
        function = schema["properties"]["tool_calls"]["items"]["oneOf"][0]["properties"]["function"]
        self.assertEqual(function["properties"]["name"], {"const": "perform_click"})
        self.assertEqual(function["properties"]["arguments"], TOOLS[0]["function"]["parameters"])
        self.assertIn("Click an element", options["messages"][-1]["content"])
        self.assertEqual(MESSAGES, [{"role": "user", "content": "Click Continue."}])

    def test_plain_generation_separates_reasoning_from_visible_text(self):
        for raw in ("<think>Reasoning</think>\nAnswer", "Reasoning</think>\nAnswer"):
            with self.subTest(raw=raw):
                self.llm.gguf_model.create_chat_completion.return_value = completion(raw)
                self.assertEqual(self.llm.gen_text(MESSAGES), {
                    "raw": raw, "thoughts": "Reasoning", "text": "Answer", "tool_calls": [],
                })

    def test_invalid_json_retries_then_returns_valid_response(self):
        result = {"thoughts": "", "text": "Done", "tool_calls": []}
        self.llm.gguf_model.create_chat_completion.side_effect = [
            completion("{"), completion(json.dumps(result)),
        ]
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS)["text"], "Done")
        self.assertEqual(self.llm.gguf_model.create_chat_completion.call_count, 2)

    def test_invalid_response_shapes_and_calls_fail_after_bounded_retries(self):
        for result in (
            [], {}, {"thoughts": "", "text": "", "tool_calls": "invalid"},
            {"thoughts": "", "text": "", "tool_calls": [None]},
            {"thoughts": "", "text": "", "tool_calls": [
                {"function": {"name": "unknown_tool", "arguments": {}}}
            ]},
            {"thoughts": "", "text": "", "tool_calls": [
                {"function": {"name": "perform_click", "arguments": "{}"}}
            ]},
        ):
            with self.subTest(result=result):
                self.llm.gguf_model.reset_mock()
                self.llm.gguf_model.create_chat_completion.return_value = completion(json.dumps(result))
                with self.assertRaises(ValueError):
                    self.llm.gen_text(MESSAGES, TOOLS)
                self.assertEqual(self.llm.gguf_model.create_chat_completion.call_count, 2)

    def test_missing_content_and_generation_errors_are_explicit(self):
        self.llm.gguf_model.create_chat_completion.return_value = completion(None)
        with self.assertRaisesRegex(ValueError, "no text content"):
            self.llm.gen_text(MESSAGES)
        self.llm.gguf_model.create_chat_completion.side_effect = RuntimeError("context too long")
        with self.assertRaisesRegex(RuntimeError, "context too long"):
            self.llm.gen_text(MESSAGES)


if __name__ == "__main__":
    unittest.main()

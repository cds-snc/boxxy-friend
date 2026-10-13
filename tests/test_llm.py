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
PROGRESS_TOOL = {
    "type": "function",
    "function": {
        "name": "update_progress",
        "description": "Update progress notes.",
        "parameters": {
            "type": "object",
            "properties": {"finding": {"type": "string"}},
        },
    },
}


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
        log_patch = patch("modules.llm.log")
        self.log = log_patch.start()
        self.addCleanup(log_patch.stop)

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
        self.assertEqual(schema["properties"]["tool_calls"]["maxItems"], 1)
        self.assertEqual(function["properties"]["name"], {"const": "perform_click"})
        self.assertEqual(function["properties"]["arguments"], TOOLS[0]["function"]["parameters"])
        self.assertIn("Click an element", options["messages"][-1]["content"])
        self.assertIn("at most ONE browser action", options["messages"][-1]["content"])
        self.assertIn("one short sentence", options["messages"][-1]["content"])
        self.assertEqual(MESSAGES, [{"role": "user", "content": "Click Continue."}])
        self.log.assert_not_called()

    def test_one_action_and_optional_progress_are_preserved_in_either_order(self):
        action = {"function": {"name": "perform_click", "arguments": {"element": 'button "Continue"'}}}
        progress = {"function": {"name": "update_progress", "arguments": {"finding": "Continue is available"}}}
        for calls in ([], [action], [progress], [progress, action], [action, progress]):
            with self.subTest(calls=calls):
                result = {"thoughts": "", "text": "", "tool_calls": calls}
                raw = json.dumps(result)
                self.llm.gguf_model.create_chat_completion.return_value = completion(raw)
                self.assertEqual(
                    self.llm.gen_text(MESSAGES, [*TOOLS, PROGRESS_TOOL]), {**result, "raw": raw}
                )
                schema = self.llm.gguf_model.create_chat_completion.call_args.kwargs["response_format"]["schema"]
                self.assertEqual(schema["properties"]["tool_calls"]["maxItems"], 2)
        self.log.assert_not_called()

    def test_excess_actions_or_progress_updates_are_rejected_not_silently_dropped(self):
        action = {"function": {"name": "perform_click", "arguments": {"element": 'button "Continue"'}}}
        progress = {"function": {"name": "update_progress", "arguments": {}}}
        for tools, calls in (
            (TOOLS, [action, action]),
            ([*TOOLS, PROGRESS_TOOL], [action, action]),
            ([*TOOLS, PROGRESS_TOOL], [progress, progress]),
            ([*TOOLS, PROGRESS_TOOL], [progress, action, action]),
        ):
            with self.subTest(calls=calls, tools=tools):
                self.llm.gguf_model.reset_mock()
                self.log.reset_mock()
                raw = json.dumps({"thoughts": "", "text": "", "tool_calls": calls})
                self.llm.gguf_model.create_chat_completion.return_value = completion(raw)
                with self.assertRaisesRegex(ValueError, "at most one browser action"):
                    self.llm.gen_text(MESSAGES, tools)
                self.assertEqual(self.llm.gguf_model.create_chat_completion.call_count, 2)
                self.assertEqual(self.log.call_count, 2)

    def test_repeated_actions_retry_with_a_single_action(self):
        action = {"function": {"name": "perform_click", "arguments": {"element": 'button "Continue"'}}}
        invalid = {"thoughts": "", "text": "", "tool_calls": [action] * 20}
        valid = {**invalid, "tool_calls": [action]}
        self.llm.gguf_model.create_chat_completion.side_effect = [
            completion(json.dumps(invalid)), completion(json.dumps(valid)),
        ]
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS)["tool_calls"], [action])
        messages = self.llm.gguf_model.create_chat_completion.call_args.kwargs["messages"]
        self.assertIn("at most one browser action", messages[-1]["content"])
        self.assertIn("much shorter", messages[-1]["content"])
        self.assertNotIn(json.dumps(invalid), [message["content"] for message in messages])

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
        self.log.assert_called_once()
        message = self.log.call_args.args[0]
        self.assertIn("attempt 1/2", message)
        self.assertIn("finish_reason='unknown'", message)
        self.assertIn("usage=None", message)

    def test_truncated_json_logs_diagnostics_on_both_attempts_and_still_raises(self):
        raw = '{"thoughts": "' + "x" * 600 + "\\nunfinished"
        response = completion(raw)
        response["choices"][0]["finish_reason"] = "length"
        response["usage"] = {
            "prompt_tokens": 3000, "completion_tokens": 2048, "total_tokens": 5048,
        }
        self.llm.gguf_model.create_chat_completion.return_value = response

        with self.assertRaises(json.JSONDecodeError):
            self.llm.gen_text(MESSAGES, TOOLS)

        self.assertEqual(self.llm.gguf_model.create_chat_completion.call_count, 2)
        self.assertEqual(self.log.call_count, 2)
        for attempt, call in enumerate(self.log.call_args_list, start=1):
            message = call.args[0]
            self.assertIn(f"attempt {attempt}/2", message)
            self.assertIn("Unterminated string", message)
            self.assertIn("finish_reason='length'", message)
            self.assertIn(f"usage={response['usage']!r}", message)
            self.assertIn(f"max_tokens={self.llm.MAX_TOKENS}", message)
            self.assertIn(f"response_chars={len(raw)}", message)
            self.assertIn(f"response_tail (last 500 characters)={raw[-500:]!r}", message)
            self.assertNotIn(raw[:600], message)

    def test_truncation_retry_requests_shorter_output_without_echoing_invalid_response(self):
        raw = '{"thoughts": "' + "repeated action " * 600
        truncated = completion(raw)
        truncated["choices"][0]["finish_reason"] = "length"
        result = {"thoughts": "Click Continue.", "text": "", "tool_calls": []}
        self.llm.gguf_model.create_chat_completion.side_effect = [
            truncated, completion(json.dumps(result)),
        ]
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS), {**result, "raw": json.dumps(result)})
        first, retry = self.llm.gguf_model.create_chat_completion.call_args_list
        self.assertEqual(len(retry.kwargs["messages"]), len(first.kwargs["messages"]) + 1)
        correction = retry.kwargs["messages"][-1]["content"]
        self.assertIn("output token limit", correction)
        self.assertIn("much shorter", correction)
        self.assertIn("at most ONE browser action", correction)
        self.assertNotIn(raw, [message["content"] for message in retry.kwargs["messages"]])
        self.assertEqual(retry.kwargs["max_tokens"], 2048)
        self.assertEqual(MESSAGES, [{"role": "user", "content": "Click Continue."}])

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

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules.llm import LLM
from modules.objective import Objective, refine_objective
from dataclasses import asdict


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
                llm.mlx_model = MagicMock()
                llm.mlx_tokenizer = MagicMock()
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
                self.assertIsNone(llm.mlx_model)
                self.assertIsNone(llm.mlx_tokenizer)
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
        llm.mlx_model = MagicMock()
        llm.mlx_tokenizer = MagicMock()
        with (
            patch("modules.llm.AutoProcessor") as processor,
            patch("modules.llm.AutoModelForMultimodalLM") as model,
        ):
            llm.load_model(str(self.folder / "transformers"))
        processor.from_pretrained.assert_called_once_with(str(self.folder / "transformers"))
        model.from_pretrained.assert_called_once_with(str(self.folder / "transformers"), dtype="auto")
        model.from_pretrained.return_value.to.assert_called_once_with("mps")
        self.assertIsNone(llm.gguf_model)
        self.assertIsNone(llm.mlx_model)
        self.assertIsNone(llm.mlx_tokenizer)


class MLXLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ornith"
        self.path.mkdir()
        for filename, data in (
            ("config.json", {
                "model_type": "qwen3_5", "quantization": {"bits": 6, "group_size": 64},
            }),
            ("tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"}),
            ("tokenizer.json", {}),
        ):
            (self.path / filename).write_text(json.dumps(data), encoding="utf-8")
        (self.path / "model.safetensors").write_bytes(b"weights")
        self.backend = MagicMock()
        self.backend.load.return_value = (MagicMock(), MagicMock())
        for target, value in (("system", "Darwin"), ("machine", "arm64")):
            patcher = patch(f"modules.llm.platform.{target}", return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_loads_mlx_without_transformers_and_clears_other_backends(self):
        llm = LLM()
        llm.model = llm.processor = llm.gguf_model = MagicMock()
        with (
            patch.dict(sys.modules, {"mlx_lm": self.backend}),
            patch("modules.llm.AutoProcessor") as processor,
            patch("modules.llm.AutoModelForMultimodalLM") as model,
        ):
            llm.load_model(self.path)
        self.backend.load.assert_called_once_with(str(self.path.resolve()))
        self.assertIs(llm.mlx_model, self.backend.load.return_value[0])
        self.assertIs(llm.mlx_tokenizer, self.backend.load.return_value[1])
        self.assertIsNone(llm.model)
        self.assertIsNone(llm.processor)
        self.assertIsNone(llm.gguf_model)
        processor.from_pretrained.assert_not_called()
        model.from_pretrained.assert_not_called()

    def test_missing_backend_has_actionable_error(self):
        with patch.dict(sys.modules, {"mlx_lm": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
                LLM().load_model(self.path)

    def test_unsupported_platform_has_explicit_error(self):
        for system, machine in (("Darwin", "x86_64"), ("Linux", "aarch64")):
            with (
                self.subTest(system=system, machine=machine),
                patch("modules.llm.platform.system", return_value=system),
                patch("modules.llm.platform.machine", return_value=machine),
                patch.dict(sys.modules, {"mlx_lm": self.backend}),
            ):
                with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
                    LLM().load_model(self.path)
        self.backend.load.assert_not_called()

    def test_load_errors_propagate_and_preserve_previous_backend(self):
        self.backend.load.side_effect = ValueError("unsupported architecture")
        llm = LLM()
        previous = llm.gguf_model = MagicMock()
        with patch.dict(sys.modules, {"mlx_lm": self.backend}):
            with self.assertRaisesRegex(ValueError, "unsupported architecture"):
                llm.load_model(self.path)
        self.assertIs(llm.gguf_model, previous)
        self.assertIsNone(llm.mlx_model)

    def test_incomplete_snapshot_does_not_reach_backend(self):
        (self.path / "tokenizer.json").unlink()
        with patch.dict(sys.modules, {"mlx_lm": self.backend}):
            with self.assertRaisesRegex(ValueError, "Missing tokenizer"):
                LLM().load_model(self.path)
        self.backend.load.assert_not_called()


class MLXConstraintSetupTests(unittest.TestCase):
    def test_reuses_tokenizer_preprocessing_with_fresh_processor_per_attempt(self):
        llm = LLM()
        llm.mlx_tokenizer = MagicMock()
        constraints_module = MagicMock()
        first, retry = object(), object()
        constraints_module.MLXJSONSchema.return_value.processor.side_effect = [first, retry]
        schema = {"type": "object"}
        with patch.dict(sys.modules, {"modules.mlx_json": constraints_module}):
            self.assertEqual(llm._mlx_logits_processors(schema), [first])
            self.assertEqual(llm._mlx_logits_processors(schema), [retry])
        constraints_module.MLXJSONSchema.assert_called_once_with(llm.mlx_tokenizer)
        self.assertEqual(
            constraints_module.MLXJSONSchema.return_value.processor.call_args_list,
            [unittest.mock.call(schema), unittest.mock.call(schema)],
        )


class MLXGenerationTests(unittest.TestCase):
    @staticmethod
    def chunks(raw, finish_reason="stop", generation_tokens=80):
        return [SimpleNamespace(
            text=raw, finish_reason=finish_reason,
            prompt_tokens=100, generation_tokens=generation_tokens,
        )]

    def setUp(self):
        self.llm = LLM()
        self.llm.mlx_model = MagicMock()
        self.llm.mlx_tokenizer = MagicMock()
        self.llm.mlx_tokenizer.apply_chat_template.return_value = "formatted prompt"
        self.backend = MagicMock()
        self.sampling = MagicMock()
        patcher = patch.dict(sys.modules, {"mlx_lm": self.backend, "mlx_lm.sample_utils": self.sampling})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("modules.llm.log")
        self.log = patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(self.llm, "_mlx_logits_processors", side_effect=lambda schema: [MagicMock()])
        self.processors = patcher.start()
        self.addCleanup(patcher.stop)

    def test_plain_generation_uses_template_and_preserves_response_contract(self):
        messages = [{"role": "model", "content": "Earlier answer"}, *MESSAGES]
        self.backend.generate.return_value = "A haiku"
        self.assertEqual(self.llm.gen_text(messages), {
            "raw": "A haiku", "thoughts": "", "text": "A haiku", "tool_calls": [],
        })
        self.llm.mlx_tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "assistant", "content": "Earlier answer"}, *MESSAGES],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        self.backend.generate.assert_called_once_with(
            self.llm.mlx_model, self.llm.mlx_tokenizer,
            prompt="formatted prompt", max_tokens=2048, verbose=False,
        )
        self.assertEqual(messages[0]["role"], "model")

    def test_plain_generation_separates_thinking(self):
        self.backend.generate.return_value = "<think>Reasoning</think>\nAnswer"
        self.assertEqual(self.llm.gen_text(MESSAGES), {
            "raw": "<think>Reasoning</think>\nAnswer",
            "thoughts": "Reasoning", "text": "Answer", "tool_calls": [],
        })

    def test_tools_and_progress_use_existing_json_contract(self):
        result = {
            "thoughts": "Continue", "text": "", "tool_calls": [
                {"function": {"name": "perform_click", "arguments": {"element": "Continue"}}},
                {"function": {"name": "update_progress", "arguments": {"finding": "Found button"}}},
            ],
        }
        raw = json.dumps(result)
        self.backend.stream_generate.return_value = self.chunks(raw)
        self.assertEqual(self.llm.gen_text(MESSAGES, [*TOOLS, PROGRESS_TOOL]), {**result, "raw": raw})
        prompt_messages = self.llm.mlx_tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("Available tools:", prompt_messages[-1]["content"])
        self.assertIn("at most ONE browser action", prompt_messages[-1]["content"])
        self.assertEqual(MESSAGES, [{"role": "user", "content": "Click Continue."}])
        schema = self.processors.call_args.args[0]
        self.assertEqual(schema["properties"]["tool_calls"]["maxItems"], 2)
        self.assertEqual(
            schema["properties"]["tool_calls"]["items"]["oneOf"][0]["properties"]["function"]
            ["properties"]["arguments"], TOOLS[0]["function"]["parameters"],
        )
        self.assertEqual(len(self.backend.stream_generate.call_args.kwargs["logits_processors"]), 1)
        self.assertEqual(self.backend.stream_generate.call_args.kwargs["max_tokens"], 8192)
        self.sampling.make_sampler.assert_called_once_with(temp=0.2)
        self.log.assert_not_called()

    def test_objective_refinement_uses_proposal_instructions(self):
        self.backend.stream_generate.return_value = self.chunks(json.dumps({
            "thoughts": "", "text": "Use two inputs.", "tool_calls": [{
                "function": {"name": "propose_objective", "arguments": asdict(Objective(element_count=2))}
            }],
        }))
        _, proposal = refine_objective(self.llm, Objective(), [], "Use two inputs")
        self.assertEqual(proposal.element_count, 2)
        messages = self.llm.mlx_tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("ONE propose_objective", messages[-1]["content"])
        self.assertNotIn("current ARIA snapshot", messages[-1]["content"])

    def test_invalid_output_retries_then_returns_valid_response(self):
        valid = {"thoughts": "", "text": "Done", "tool_calls": []}
        self.backend.stream_generate.side_effect = [self.chunks("{"), self.chunks(json.dumps(valid))]
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS), {**valid, "raw": json.dumps(valid)})
        self.assertEqual(self.backend.stream_generate.call_count, 2)
        self.log.assert_called_once()
        self.assertIn("MLX response parsing failed", self.log.call_args.args[0])
        messages = self.llm.mlx_tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("much shorter", messages[-1]["content"])
        self.assertEqual(self.processors.call_count, 2)
        first, retry = self.backend.stream_generate.call_args_list
        self.assertIsNot(first.kwargs["logits_processors"][0], retry.kwargs["logits_processors"][0])

    def test_invalid_shapes_unknown_tools_and_excess_actions_fail_explicitly(self):
        action = {"function": {"name": "perform_click", "arguments": {}}}
        for result in (
            [], {}, {"thoughts": "", "text": "", "tool_calls": [None]},
            {"thoughts": "", "text": "", "tool_calls": [
                {"function": {"name": "unknown_tool", "arguments": {}}},
            ]},
            {"thoughts": "", "text": "", "tool_calls": [
                {"function": {"name": "perform_click", "arguments": "{}"}},
            ]},
            {"thoughts": "", "text": "", "tool_calls": [action, action]},
        ):
            with self.subTest(result=result):
                self.backend.stream_generate.reset_mock(side_effect=True)
                self.backend.stream_generate.return_value = self.chunks(json.dumps(result))
                with self.assertRaises(ValueError):
                    self.llm.gen_text(MESSAGES, TOOLS)
                self.assertEqual(self.backend.stream_generate.call_count, 2)

    def test_truncation_retries_with_real_token_diagnostics_and_never_returns_partial_tools(self):
        valid = {"thoughts": "", "text": "", "tool_calls": []}
        self.backend.stream_generate.side_effect = [
            self.chunks('{"thoughts":"","text":"' + "\\n" * 2048, "length", 8192),
            self.chunks(json.dumps(valid)),
        ]
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS), {**valid, "raw": json.dumps(valid)})
        self.log.assert_called_once()
        diagnostic = self.log.call_args.args[0]
        self.assertIn("finish_reason='length'", diagnostic)
        self.assertIn("'completion_tokens': 8192", diagnostic)
        self.assertIn("max_tokens=8192", diagnostic)
        self.assertIn("incomplete tool calls will not be executed", diagnostic)
        messages = self.llm.mlx_tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("output token limit", messages[-1]["content"])

    def test_repeated_truncation_raises_actionable_error_even_if_partial_json_is_parseable(self):
        raw = json.dumps({"thoughts": "", "text": "", "tool_calls": []})
        self.backend.stream_generate.return_value = self.chunks(raw, "length", 8192)
        with self.assertRaisesRegex(ValueError, "8192-token output limit"):
            self.llm.gen_text(MESSAGES, TOOLS)
        self.assertEqual(self.backend.stream_generate.call_count, 2)

    def test_empty_stream_and_generation_errors_are_explicit(self):
        self.backend.stream_generate.return_value = []
        with self.assertRaisesRegex(ValueError, "no output chunks"):
            self.llm.gen_text(MESSAGES, TOOLS)
        self.backend.stream_generate.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.llm.gen_text(MESSAGES, TOOLS)

    def test_stream_chunks_are_joined_without_repeating_output(self):
        raw = json.dumps({"thoughts": "", "text": "Done", "tool_calls": []})
        first = self.chunks(raw[:20], None)[0]
        last = self.chunks(raw[20:])[0]
        self.backend.stream_generate.return_value = [first, last]
        self.assertEqual(self.llm.gen_text(MESSAGES, TOOLS)["raw"], raw)

    def test_missing_content_and_generation_errors_are_explicit(self):
        self.backend.generate.return_value = None
        with self.assertRaisesRegex(ValueError, "MLX model returned no text content"):
            self.llm.gen_text(MESSAGES)
        self.backend.generate.side_effect = RuntimeError("out of memory")
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.llm.gen_text(MESSAGES)


class GGUFGenerationTests(unittest.TestCase):
    def test_refinement_uses_proposal_instructions_not_browser_instructions(self):
        self.llm.gguf_model.create_chat_completion.return_value = completion(json.dumps({
            "thoughts": "", "text": "Use two inputs.", "tool_calls": [{
                "function": {"name": "propose_objective",
                             "arguments": asdict(Objective(element_count=2))}
            }],
        }))
        _, proposal = refine_objective(self.llm, Objective(), [], "Use two inputs")
        self.assertEqual(proposal.element_count, 2)
        messages = self.llm.gguf_model.create_chat_completion.call_args.kwargs["messages"]
        self.assertIn("ONE propose_objective", messages[-1]["content"])
        self.assertNotIn("current ARIA snapshot", messages[-1]["content"])

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

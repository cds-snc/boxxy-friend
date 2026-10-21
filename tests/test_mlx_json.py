import importlib.util
import json
import platform
import unittest
from unittest.mock import MagicMock

from modules.mlx_json import bounded_schema


class BoundedSchemaTests(unittest.TestCase):
    def test_bounds_nested_strings_arrays_and_properties_without_mutating_input(self):
        schema = {
            "type": "object", "properties": {
                "thoughts": {"type": "string"}, "text": {"type": "string"},
                "tool_calls": {
                    "type": "array", "maxItems": 2,
                    "items": {"oneOf": [{
                        "type": "object", "properties": {
                            "plan": {"type": "array", "items": {"type": "string"}},
                            "done": {"type": "array", "items": {"type": "string"}},
                            "element": {"type": "string"}, "intent": {"type": "string"},
                        },
                    }]},
                },
            },
        }
        before = json.dumps(schema)
        bounded = bounded_schema(schema)
        self.assertEqual(json.dumps(schema), before)
        self.assertEqual(bounded["properties"]["thoughts"]["maxLength"], 120)
        self.assertEqual(bounded["properties"]["text"]["maxLength"], 1024)
        calls = bounded["properties"]["tool_calls"]
        self.assertEqual(calls["maxItems"], 2)
        arguments = calls["items"]["oneOf"][0]
        self.assertFalse(arguments["additionalProperties"])
        for field in ("plan", "done"):
            self.assertEqual(arguments["properties"][field]["maxItems"], 15)
            self.assertEqual(arguments["properties"][field]["items"]["maxLength"], 120)
        self.assertEqual(arguments["properties"]["element"]["maxLength"], 1024)
        self.assertEqual(arguments["properties"]["intent"]["maxLength"], 120)

    def test_preserves_explicit_limits_required_fields_and_literal_values(self):
        schema = {
            "type": "object",
            "properties": {
                "url": {"type": "string", "maxLength": 2048},
                "title": {"type": "string", "minLength": 1, "maxLength": 120},
                "text": {"type": "string", "minLength": 2000},
                "name": {"const": "perform_click"},
                "choice": {"type": "string", "enum": ["Option A", "Option B"]},
                "items": {"type": "array", "minItems": 20, "items": {"type": "string"}},
            },
            "required": ["url", "title"], "additionalProperties": False,
        }
        bounded = bounded_schema(schema)
        for field in ("url", "title", "name", "choice"):
            self.assertEqual(bounded["properties"][field], schema["properties"][field])
        self.assertEqual(bounded["required"], schema["required"])
        self.assertEqual(bounded["properties"]["text"]["maxLength"], 2000)
        self.assertEqual(bounded["properties"]["items"]["maxItems"], 20)

MLX_AVAILABLE = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("lmformatenforcer") is not None
)


@unittest.skipUnless(MLX_AVAILABLE, "Requires the Apple Silicon MLX dependencies")
class MLXJSONSchemaTests(unittest.TestCase):
    def setUp(self):
        import mlx.core as mx
        from modules.mlx_json import MLXJSONSchema

        self.mx = mx
        characters = [chr(value) for value in range(32, 127)] + ["\n", "\t", "é"]
        self.vocabulary = {char: index for index, char in enumerate(characters)}
        self.eos = len(self.vocabulary)
        self.tokenizer = MagicMock()
        self.tokenizer.get_vocab.return_value = {**self.vocabulary, "<eos>": self.eos}
        self.tokenizer.all_special_ids = [self.eos]
        self.tokenizer.eos_token_ids = {self.eos}
        self.tokenizer.encode.side_effect = lambda text, **_kwargs: [self.vocabulary[c] for c in text]
        self.tokenizer.decode.side_effect = lambda tokens, **_kwargs: "".join(characters[t] for t in tokens)
        self.constraints = MLXJSONSchema(self.tokenizer)
        self.schema = {
            "type": "object",
            "properties": {
                "thoughts": {"type": "string", "maxLength": 8},
                "tool_calls": {
                    "type": "array", "maxItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "function": {
                                "type": "object",
                                "properties": {
                                    "name": {"const": "perform_click"},
                                    "arguments": {
                                        "type": "object",
                                        "properties": {"element": {"type": "string"}},
                                        "required": ["element"],
                                        "additionalProperties": False,
                                    },
                                },
                                "required": ["name", "arguments"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["function"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["thoughts", "tool_calls"],
            "additionalProperties": False,
        }

    def replay(self, processor, text, tokens=None):
        tokens = [self.eos] if tokens is None else tokens
        for char in text:
            logits = processor(self.mx.array(tokens), self.mx.zeros((1, self.eos + 1)))
            token = self.vocabulary[char]
            self.assertTrue(self.mx.isfinite(logits[0, token]).item(), (char, text))
            tokens.append(token)
        return tokens

    def test_masks_extra_closing_brace_from_reported_failure_and_preserves_quotes(self):
        processor = self.constraints.processor(self.schema)
        prefix = '{"thoughts":"Click","tool_calls":[{"function":{"name":"perform_click","arguments":{"element":"button \\"Continue\\""}}}'
        tokens = self.replay(processor, prefix)
        logits = processor(self.mx.array(tokens), self.mx.zeros((1, self.eos + 1)))
        self.assertTrue(self.mx.isneginf(logits[0, self.vocabulary["}"]]).item())
        self.assertTrue(self.mx.isneginf(logits[0, self.eos]).item())
        tokens = self.replay(processor, "]}", tokens)
        logits = processor(self.mx.array(tokens), self.mx.zeros((1, self.eos + 1)))
        self.assertTrue(self.mx.isfinite(logits[0, self.eos]).item())
        self.assertEqual(json.loads(prefix + "]}")["tool_calls"][0]["function"]["arguments"],
                         {"element": 'button "Continue"'})

    def test_bounds_thoughts_to_leave_budget_for_tool_calls(self):
        processor = self.constraints.processor(self.schema)
        tokens = self.replay(processor, '{"thoughts":"12345678')
        logits = processor(self.mx.array(tokens), self.mx.zeros(self.eos + 1))
        self.assertTrue(self.mx.isneginf(logits[self.vocabulary["9"]]).item())
        self.assertTrue(self.mx.isfinite(logits[self.vocabulary['"']]).item())

    def test_bounds_repeated_escapes_in_action_argument_before_token_limit(self):
        from modules.llm import LLM

        processor = self.constraints.processor(self.schema)
        prefix = '{"thoughts":"","tool_calls":[{"function":{"name":"perform_click","arguments":{"element":"'
        tokens = self.replay(processor, prefix + "\\n" * 1024)
        logits = processor(self.mx.array(tokens), self.mx.zeros(self.eos + 1))
        self.assertTrue(self.mx.isneginf(logits[self.vocabulary["\\"]]).item())
        tokens = self.replay(processor, '"}}}]}', tokens)
        logits = processor(self.mx.array(tokens), self.mx.zeros(self.eos + 1))
        self.assertTrue(self.mx.isfinite(logits[self.eos]).item())
        self.assertGreater(len(tokens), LLM.MAX_TOKENS)
        self.assertLess(len(tokens), LLM.MLX_TOOL_MAX_TOKENS)

    def test_progress_arrays_cannot_repeat_forever(self):
        schema = {"type": "array", "items": {"type": "string"}}
        processor = self.constraints.processor(schema)
        prefix = "[" + ",".join(['"step"'] * 15)
        tokens = self.replay(processor, prefix)
        logits = processor(self.mx.array(tokens), self.mx.zeros(self.eos + 1))
        self.assertTrue(self.mx.isneginf(logits[self.vocabulary[","]]).item())
        self.assertTrue(self.mx.isfinite(logits[self.vocabulary["]"]]).item())

    def test_whitespace_is_bounded_outside_strings(self):
        processor = self.constraints.processor(self.schema)
        tokens = self.replay(processor, " ")
        logits = processor(self.mx.array(tokens), self.mx.zeros(self.eos + 1))
        self.assertTrue(self.mx.isneginf(logits[self.vocabulary[" "]]).item())
        self.assertTrue(self.mx.isfinite(logits[self.vocabulary["{"]]).item())

    def test_enforces_tool_name_arguments_and_call_count(self):
        for prefix, invalid in (
            ('{"thoughts":"","tool_calls":[{"function":{"name":"perform_', "x"),
            ('{"thoughts":"","tool_calls":[{"function":{"name":"perform_click","arguments":', '"'),
            ('{"thoughts":"","tool_calls":[{"function":{"name":"perform_click","arguments":{"element":"Continue"}}}', ","),
        ):
            with self.subTest(prefix=prefix):
                processor = self.constraints.processor(self.schema)
                tokens = self.replay(processor, prefix)
                logits = processor(self.mx.array(tokens), self.mx.zeros(self.eos + 1))
                self.assertTrue(self.mx.isneginf(logits[self.vocabulary[invalid]]).item())

    def test_reuses_tokenizer_data_but_resets_parser_for_each_generation(self):
        for _ in range(2):
            processor = self.constraints.processor(self.schema)
            self.replay(processor, '{"thoughts":"café","tool_calls":[]}')
        self.tokenizer.get_vocab.assert_called_once()


if __name__ == "__main__":
    unittest.main()

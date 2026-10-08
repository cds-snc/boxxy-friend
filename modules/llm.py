import json
import re
from pathlib import Path

from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils.chat_parsing_utils import recursive_parse

from modules.models import resolve_gguf, validate_gguf


class LLM:
    MAX_PARSE_ATTEMPTS = 2
    MAX_TOKENS=2048

    def __init__(self):
        self.model = None
        self.processor = None
        self.gguf_model = None

    def load_model(self, local_path):
        gguf = resolve_gguf(Path(local_path))
        if gguf is not None:
            validate_gguf(gguf)
            try:
                from llama_cpp import Llama
            except ModuleNotFoundError as error:
                if error.name != "llama_cpp":
                    raise
                raise RuntimeError(
                    "GGUF models require llama-cpp-python. Install it with: pip install llama-cpp-python"
                ) from error
            model = Llama(model_path=str(gguf), n_ctx=8192, n_gpu_layers=-1, verbose=False)
            self.gguf_model = model
            self.model = None
            self.processor = None
            return

        # Load model
        processor = AutoProcessor.from_pretrained(local_path)
        model = AutoModelForMultimodalLM.from_pretrained(
            local_path,
            dtype="auto"
        )

        model = model.to("mps")
        self.model = model
        self.processor = processor
        self.gguf_model = None

    def gen_text(self, input_text, tools_schema=None):
        if self.gguf_model is not None:
            return self._gen_gguf_text(input_text, tools_schema)
        messages = input_text

        for attempt in range(self.MAX_PARSE_ATTEMPTS):
            response = self._generate_response(messages, tools_schema)

            try:
                return self._parse_response(response)
            except ValueError:
                if attempt == self.MAX_PARSE_ATTEMPTS - 1:
                    raise

                messages = [
                    *input_text,
                    {"role": "model", "content": response},
                    {
                        "role": "user",
                        "content": (
                            "Your previous response could not be parsed. Respond again with the same intended "
                            "answer, but ensure every tool call is valid JSON. JSON object keys and string values "
                            "must use double quotes, not single quotes. Do not include the invalid response in your "
                            "answer."
                        )
                    }
                ]

        raise RuntimeError("LLM response generation exhausted without returning or raising")

    def _gen_gguf_text(self, input_text, tools_schema):
        messages = [
            {**message, "role": "assistant" if message["role"] == "model" else message["role"]}
            for message in input_text
        ]
        options = {}
        if tools_schema:
            call_schemas = []
            for tool in tools_schema:
                function = tool["function"]
                call_schemas.append({
                    "type": "object",
                    "properties": {
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {"const": function["name"]},
                                "arguments": function["parameters"],
                            },
                            "required": ["name", "arguments"],
                            "additionalProperties": False,
                        }
                    },
                    "required": ["function"],
                    "additionalProperties": False,
                })
            options["response_format"] = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "thoughts": {"type": "string"},
                        "text": {"type": "string"},
                        "tool_calls": {"type": "array", "items": {"oneOf": call_schemas}},
                    },
                    "required": ["thoughts", "text", "tool_calls"],
                    "additionalProperties": False,
                },
            }
            messages.append({
                "role": "user",
                "content": (
                    "Return only a JSON object with thoughts (string), text (string), and tool_calls (array). "
                    "Each tool call must be {\"function\": {\"name\": \"tool_name\", \"arguments\": {...}}}. "
                    "Use an empty array when no action is needed. Available tools:\n" + json.dumps(tools_schema)
                ),
            })

        for attempt in range(self.MAX_PARSE_ATTEMPTS):
            completion = self.gguf_model.create_chat_completion(
                messages=messages, max_tokens=self.MAX_TOKENS, **options
            )
            raw = completion["choices"][0]["message"]["content"]
            if not isinstance(raw, str):
                raise ValueError("GGUF model returned no text content")
            if not tools_schema:
                thoughts, separator, text = raw.partition("</think>")
                if separator:
                    thoughts = thoughts.strip().removeprefix("<think>").strip()
                    text = text.strip()
                else:
                    thoughts, text = "", raw
                return {"raw": raw, "thoughts": thoughts, "text": text, "tool_calls": []}
            try:
                result = json.loads(raw)
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("thoughts"), str)
                    or not isinstance(result.get("text"), str)
                    or not isinstance(result.get("tool_calls"), list)
                ):
                    raise ValueError("GGUF response must contain thoughts, text, and tool_calls")
                names = {tool["function"]["name"] for tool in tools_schema}
                for call in result["tool_calls"]:
                    function = call.get("function") if isinstance(call, dict) else None
                    if (
                        not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or function["name"] not in names
                        or not isinstance(function.get("arguments"), dict)
                    ):
                        raise ValueError("GGUF response contains an invalid tool call")
                return {**result, "raw": raw}
            except ValueError:
                if attempt == self.MAX_PARSE_ATTEMPTS - 1:
                    raise
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "The response was invalid. Return only JSON matching the schema."},
                ])

        raise RuntimeError("GGUF response generation exhausted without returning or raising")

    def _generate_response(self, input_text, tools_schema):
        text = self.processor.apply_chat_template(
            input_text,
            tools=tools_schema,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(**inputs, max_new_tokens=self.MAX_TOKENS)
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    def _parse_response(self, response):
        parsed_response = self.processor.parse_response(response)

        thoughts = parsed_response.get("thinking", "")
        text = parsed_response.get("content", "")
        tool_calls = parsed_response.get("tool_calls", [])

        # The shipped response_schema's top-level regex only recognizes a tool
        # call when it immediately follows the thinking block, so when the
        # model emits free text before the tool call, `content` swallows the
        # raw <|tool_call>...<tool_call|> markup and `tool_calls` ends up
        # empty. Re-run just the tool_calls sub-schema against the raw
        # response (its regex searches anywhere in the text) and strip the
        # matched markup back out of `text`.
        if not tool_calls:
            schema = getattr(self.processor.tokenizer, "response_schema", None)
            tool_calls_schema = (schema or {}).get("properties", {}).get("tool_calls")
            if tool_calls_schema:
                tool_calls = recursive_parse(response, tool_calls_schema) or []
                tool_call_pattern = tool_calls_schema.get("x-regex-iterator")
                if tool_calls and tool_call_pattern:
                    text = re.sub(tool_call_pattern, "", text, flags=re.DOTALL).strip()

        return {
            "raw": response,
            "thoughts": thoughts,
            "text": text,
            "tool_calls": tool_calls
        }
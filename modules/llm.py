import json
import platform
import re
from pathlib import Path

from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils.chat_parsing_utils import recursive_parse

from modules.logger import log
from modules.models import is_mlx_model, resolve_gguf, validate_gguf, validate_model


class LLM:
    MAX_PARSE_ATTEMPTS = 2
    MAX_TOKENS=2048
    MLX_TOOL_MAX_TOKENS = 8192

    def __init__(self):
        self.model = None
        self.processor = None
        self.gguf_model = None
        self.mlx_model = None
        self.mlx_tokenizer = None
        self.mlx_constraints = None

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
            model = Llama(model_path=str(gguf), n_ctx=32000, n_gpu_layers=-1, verbose=False)
            self.gguf_model = model
            self.model = None
            self.processor = None
            self.mlx_model = None
            self.mlx_tokenizer = None
            self.mlx_constraints = None
            return

        if is_mlx_model(Path(local_path)):
            validate_model(Path(local_path))
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise RuntimeError("MLX models require macOS on Apple Silicon (arm64).")
            try:
                from mlx_lm import load
            except ModuleNotFoundError as error:
                if error.name not in ("mlx_lm", "mlx", "mlx.core"):
                    raise
                raise RuntimeError(
                    "MLX models require mlx-lm. Install it with: pip install -r requirements.txt"
                ) from error
            model, tokenizer = load(str(Path(local_path).resolve()))
            self.mlx_model = model
            self.mlx_tokenizer = tokenizer
            self.mlx_constraints = None
            self.model = None
            self.processor = None
            self.gguf_model = None
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
        self.mlx_model = None
        self.mlx_tokenizer = None
        self.mlx_constraints = None

    def gen_text(self, input_text, tools_schema=None):
        if self.gguf_model is not None or self.mlx_model is not None:
            return self._gen_local_text(input_text, tools_schema)
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

    def _mlx_logits_processors(self, schema):
        if self.mlx_constraints is None:
            try:
                from modules.mlx_json import MLXJSONSchema
                self.mlx_constraints = MLXJSONSchema(self.mlx_tokenizer)
            except ModuleNotFoundError as error:
                if error.name != "lmformatenforcer":
                    raise
                raise RuntimeError(
                    "MLX tool generation requires lm-format-enforcer. "
                    "Install it with: pip install -r requirements.txt"
                ) from error
        return [self.mlx_constraints.processor(schema)]

    def _gen_local_text(self, input_text, tools_schema):
        backend = "MLX" if self.mlx_model is not None else "GGUF"
        messages = [
            {**message, "role": "assistant" if message["role"] == "model" else message["role"]}
            for message in input_text
        ]
        options = {}
        if tools_schema:
            names = {tool["function"]["name"] for tool in tools_schema}
            max_calls = 2 if "update_progress" in names else 1
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
                        "tool_calls": {
                            "type": "array",
                            "items": {"oneOf": call_schemas},
                            "maxItems": max_calls,
                        },
                    },
                    "required": ["thoughts", "text", "tool_calls"],
                    "additionalProperties": False,
                },
            }
            action_guidance = (
                "Return at most ONE propose_objective call for supported edits, or no tools for discussion. "
                "Do not execute browser actions. Keep advice brief. "
                if "propose_objective" in names else
                "Return at most ONE browser action and at most ONE update_progress call if that tool is "
                "available. Choose only the next action using the current ARIA snapshot, then stop: "
                "you will receive a fresh snapshot after it executes. Do not repeat actions or list future "
                "actions. Keep each intent to one short sentence. "
            )
            messages.append({
                "role": "user",
                "content": (
                    "Return only a JSON object with thoughts (string), text (string), and tool_calls (array). "
                    "Each tool call must be {\"function\": {\"name\": \"tool_name\", \"arguments\": {...}}}. "
                    + action_guidance
                    + "Keep thoughts to one short sentence. "
                    "Keep text brief or empty. "
                    "Use an empty array when no action is needed. Available tools:\n" + json.dumps(tools_schema)
                ),
            })

        for attempt in range(self.MAX_PARSE_ATTEMPTS):
            max_tokens = self.MAX_TOKENS
            if self.mlx_model is not None:
                prompt = self.mlx_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                generation_options = {}
                if tools_schema:
                    from mlx_lm import stream_generate
                    from mlx_lm.sample_utils import make_sampler

                    generation_options["sampler"] = make_sampler(temp=0.2)
                    generation_options["logits_processors"] = self._mlx_logits_processors(
                        options["response_format"]["schema"]
                    )
                    max_tokens = self.MLX_TOOL_MAX_TOKENS
                    pieces = []
                    last_chunk = None
                    for chunk in stream_generate(
                        self.mlx_model, self.mlx_tokenizer,
                        prompt=prompt, max_tokens=max_tokens, **generation_options,
                    ):
                        pieces.append(chunk.text)
                        last_chunk = chunk
                    if last_chunk is None:
                        raise ValueError("MLX model returned no output chunks")
                    raw = "".join(pieces)
                    finish_reason = last_chunk.finish_reason
                    usage = {
                        "prompt_tokens": last_chunk.prompt_tokens,
                        "completion_tokens": last_chunk.generation_tokens,
                    }
                else:
                    from mlx_lm import generate

                    raw = generate(
                        self.mlx_model, self.mlx_tokenizer,
                        prompt=prompt, max_tokens=max_tokens, verbose=False,
                    )
                    finish_reason, usage = "unknown", None
            else:
                completion = self.gguf_model.create_chat_completion(
                    messages=messages, max_tokens=self.MAX_TOKENS, **options
                )
                raw = completion["choices"][0]["message"]["content"]
                finish_reason = completion["choices"][0].get("finish_reason", "unknown")
                usage = completion.get("usage")
            if not isinstance(raw, str):
                raise ValueError(f"{backend} model returned no text content")
            if not tools_schema:
                thoughts, separator, text = raw.partition("</think>")
                if separator:
                    thoughts = thoughts.strip().removeprefix("<think>").strip()
                    text = text.strip()
                else:
                    thoughts, text = "", raw
                return {"raw": raw, "thoughts": thoughts, "text": text, "tool_calls": []}
            try:
                if self.mlx_model is not None and finish_reason == "length":
                    raise ValueError(
                        f"MLX tool response reached the {max_tokens}-token output limit; "
                        "incomplete tool calls will not be executed"
                    )
                result = json.loads(raw)
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("thoughts"), str)
                    or not isinstance(result.get("text"), str)
                    or not isinstance(result.get("tool_calls"), list)
                ):
                    raise ValueError(f"{backend} response must contain thoughts, text, and tool_calls")
                action_count = 0
                progress_count = 0
                for call in result["tool_calls"]:
                    function = call.get("function") if isinstance(call, dict) else None
                    if (
                        not isinstance(function, dict)
                        or not isinstance(function.get("name"), str)
                        or function["name"] not in names
                        or not isinstance(function.get("arguments"), dict)
                    ):
                        raise ValueError(f"{backend} response contains an invalid tool call")
                    if function["name"] == "update_progress":
                        progress_count += 1
                    else:
                        action_count += 1
                if action_count > 1 or progress_count > 1:
                    raise ValueError(
                        f"{backend} response must contain at most one browser action and one update_progress call"
                    )
                return {**result, "raw": raw}
            except ValueError as error:
                log(
                    f"{backend} response parsing failed (attempt {attempt + 1}/{self.MAX_PARSE_ATTEMPTS}): "
                    f"{error}; finish_reason={finish_reason!r}; "
                    f"usage={usage!r}; max_tokens={max_tokens}; "
                    f"response_chars={len(raw)}; response_tail (last 500 characters)={raw[-500:]!r}"
                )
                if attempt == self.MAX_PARSE_ATTEMPTS - 1:
                    raise
                reason = (
                    "The previous response reached the output token limit before producing valid output. "
                    if finish_reason == "length"
                    else f"The previous response was invalid: {error}. "
                )
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": reason + (
                            "Return a much shorter, complete JSON object matching the schema. "
                            "Include at most ONE browser action and at most ONE update_progress call if "
                            "available. Choose only the next action from the current snapshot; do not repeat "
                            "actions or list future actions. Keep thoughts and intents to one short sentence "
                            "each and text brief or empty. Do not reproduce the previous response."
                        ),
                    },
                ]

        raise RuntimeError(f"{backend} response generation exhausted without returning or raising")

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
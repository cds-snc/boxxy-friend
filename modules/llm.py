import re

from transformers import AutoProcessor, AutoModelForMultimodalLM
from transformers.utils.chat_parsing_utils import recursive_parse


class LLM:
    MAX_PARSE_ATTEMPTS = 2

    def __init__(self):
        self.model = None
        self.processor = None

    def load_model(self, local_path):
        # Load model
        processor = AutoProcessor.from_pretrained(local_path)
        model = AutoModelForMultimodalLM.from_pretrained(
            local_path,
            dtype="auto"
        )

        model = model.to("mps")
        self.model = model
        self.processor = processor

    def gen_text(self, input_text, tools_schema=None):
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

        outputs = self.model.generate(**inputs, max_new_tokens=1024)
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
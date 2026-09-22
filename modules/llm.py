from transformers import AutoProcessor, AutoModelForMultimodalLM

class LLM:
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
        text = self.processor.apply_chat_template(
            input_text,
            tools=tools_schema,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        parsed_response = self.processor.parse_response(response)

        thoughts = parsed_response.get("thinking", "")
        text = parsed_response.get("content", "")
        tool_calls = parsed_response.get("tool_calls", [])

        return {
            "raw": response,
            "thoughts": thoughts,
            "text": text,
            "tool_calls": tool_calls
        }
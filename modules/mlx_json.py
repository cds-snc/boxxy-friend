"""JSON-schema token filtering for MLX-LM generation."""

from collections.abc import Callable


def bounded_schema(schema: dict, field: str = "") -> dict:
    """Bound generated content, without changing the caller's tool definitions."""
    result = dict(schema)
    if schema.get("type") == "string" and not any(key in schema for key in ("const", "enum")):
        limit = 120 if field in ("thoughts", "intent", "plan", "done") else 1024
        result.setdefault("maxLength", max(limit, schema.get("minLength", 0)))
    if schema.get("type") == "object" and "properties" in schema:
        result["properties"] = {
            name: bounded_schema(value, name) for name, value in schema["properties"].items()
        }
        result["additionalProperties"] = False
    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
        result["items"] = bounded_schema(schema["items"], field)
        result.setdefault("maxItems", max(15, schema.get("minItems", 0)))
    for keyword in ("oneOf", "anyOf", "allOf"):
        if keyword in schema:
            result[keyword] = [bounded_schema(branch, field) for branch in schema[keyword]]
    return result


class MLXJSONSchema:
    def __init__(self, tokenizer):
        from lmformatenforcer import TokenEnforcerTokenizerData

        special_ids = set(tokenizer.all_special_ids)
        prefix = tokenizer.encode("0", add_special_tokens=False)
        regular_tokens = []
        vocabulary = tokenizer.get_vocab()
        for token_id in vocabulary.values():
            if token_id in special_ids:
                continue
            text = tokenizer.decode([token_id])
            # Contextual decoding preserves spaces on word-start tokens.
            contextual = tokenizer.decode([*prefix, token_id])[1:]
            regular_tokens.append((token_id, contextual, len(contextual) > len(text)))
        self.tokenizer_data = TokenEnforcerTokenizerData(
            regular_tokens,
            lambda tokens: tokenizer.decode(
                tokens, clean_up_tokenization_spaces=False
            ).rstrip("\ufffd"),
            sorted(tokenizer.eos_token_ids),
            use_bitmask=False,
            vocab_size=max(vocabulary.values()) + 1,
        )

    def processor(self, schema: dict) -> Callable:
        import mlx.core as mx
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser, TokenEnforcer

        config = CharacterLevelParserConfig(
            alphabet=self.tokenizer_data.tokenizer_alphabet,
            max_consecutive_whitespaces=1, force_json_field_order=True,
        )
        parser = JsonSchemaParser(bounded_schema(schema), config)
        enforcer = TokenEnforcer(self.tokenizer_data, parser)
        # lm-format-enforcer 0.11.3 replaces the parser config on construction.
        parser.config = config

        def filter_logits(tokens: mx.array, logits: mx.array) -> mx.array:
            allowed = enforcer.get_allowed_tokens(tokens.tolist()).allowed_tokens
            if not allowed:
                raise ValueError("MLX JSON schema allows no next token")
            mask = mx.full_like(logits, -mx.inf)
            mask[..., mx.array(allowed)] = 0
            return logits + mask

        return filter_logits

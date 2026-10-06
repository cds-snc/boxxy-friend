"""Check local Transformers model snapshots without loading their weights."""

import json
from pathlib import Path

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def validate_model(path: Path) -> None:
    """Raise ValueError or OSError for an incomplete local model snapshot."""
    for name in ("config.json", "processor_config.json", "tokenizer_config.json"):
        with (path / name).open(encoding="utf-8") as source:
            config = json.load(source)
        if not isinstance(config, dict) or not config:
            raise ValueError(f"{name} must contain a non-empty JSON object")
        if name == "config.json" and not config.get("model_type"):
            raise ValueError("config.json is missing model_type")

    if not any(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")
    ):
        raise ValueError("Missing tokenizer data")

    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = path / name
        if weights.is_file() and weights.stat().st_size > 0:
            return

    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = path / name
        if not index.is_file():
            continue
        with index.open(encoding="utf-8") as source:
            data = json.load(source)
        weight_map = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{name} is missing a non-empty weight_map")
        for shard in weight_map.values():
            if not isinstance(shard, str) or not shard:
                raise ValueError(f"{name} contains an invalid shard name")
            weights = (path / shard).resolve()
            if not weights.is_relative_to(path.resolve()):
                raise ValueError(f"Weight shard is outside the model folder: {shard}")
            if not weights.is_file() or weights.stat().st_size == 0:
                raise ValueError(f"Missing or empty weight shard: {shard}")
        return
    raise ValueError("Missing model weights (Safetensors or PyTorch)")


def discover_models(models_dir: Path) -> tuple[list[Path], list[str]]:
    """Return complete snapshots and reasons for rejecting other folders."""
    models = []
    rejected = []
    for path in sorted(models_dir.iterdir(), key=lambda entry: entry.name.casefold()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        try:
            validate_model(path)
        except (OSError, ValueError) as error:
            rejected.append(f"{path.name}: {error}")
        else:
            models.append(path)
    return models, rejected

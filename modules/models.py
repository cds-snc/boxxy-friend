"""Check local Transformers snapshots and standalone GGUF models."""

import json
import re
import struct
from pathlib import Path

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def gguf_files(path: Path) -> list[Path]:
    """Return visible GGUF files directly inside a model folder."""
    return sorted(
        (entry for entry in path.iterdir()
         if entry.is_file() and not entry.name.startswith(".") and entry.suffix.casefold() == ".gguf"),
        key=lambda entry: entry.name.casefold(),
    )


def resolve_gguf(path: Path) -> Path | None:
    """Resolve a GGUF file or an unambiguous GGUF folder."""
    if path.suffix.casefold() == ".gguf" and not path.is_dir():
        return path
    if path.is_dir() and not (path / "config.json").is_file():
        files = gguf_files(path)
        if len(files) > 1:
            raise ValueError("Multiple GGUF models in this folder; select a specific GGUF file")
        if files:
            return files[0]
    return None


def validate_gguf(path: Path) -> None:
    """Check a standalone GGUF header; full compatibility is checked on load."""
    if path.name.casefold().startswith("mmproj"):
        raise ValueError("GGUF projection files are not standalone language models")
    if re.search(r"-\d{5}-of-\d{5}\.gguf$", path.name, re.IGNORECASE):
        raise ValueError("Split GGUF models are not supported; use a single-file GGUF")
    with path.open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:4] != b"GGUF":
        raise ValueError("Missing or invalid GGUF header")
    _, version, tensors, metadata = struct.unpack("<4sIQQ", header)
    if version not in (2, 3):
        raise ValueError(f"Unsupported GGUF version: {version}")
    if not tensors or not metadata or path.stat().st_size <= 24:
        raise ValueError("GGUF model is empty or missing tensors/metadata")


def validate_model(path: Path) -> None:
    """Raise ValueError or OSError for an incomplete local model snapshot."""
    gguf = resolve_gguf(path)
    if gguf is not None:
        validate_gguf(gguf)
        return

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
    """Return snapshots/GGUF files and reasons for rejecting other entries."""
    models = []
    rejected = []
    for path in sorted(models_dir.iterdir(), key=lambda entry: entry.name.casefold()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            try:
                candidates = gguf_files(path)
                if not candidates or (path / "config.json").is_file():
                    candidates.insert(0, path)
            except OSError as error:
                rejected.append(f"{path.name}: {error}")
                continue
        elif path.suffix.casefold() == ".gguf":
            candidates = [path]
        else:
            continue
        for candidate in candidates:
            try:
                validate_model(candidate)
            except (OSError, ValueError) as error:
                rejected.append(f"{candidate.relative_to(models_dir)}: {error}")
            else:
                models.append(candidate)
    return models, rejected

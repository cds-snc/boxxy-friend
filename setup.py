#!/usr/bin/env python3
"""Download the default Ornith GGUF model from Hugging Face.

Usage:
  python setup.py
  python setup.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_REPO_ID = "ornith-ai/Ornith-1.5-9B-GGUF"
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_FILENAME = "Ornith-1.5-9B-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Ornith 1.5 9B Q4_K_M GGUF model from Hugging Face."
    )
    parser.add_argument(
        "--allow-patterns",
        nargs="+",
        default=[DEFAULT_FILENAME],
        help="Files to download (default: %(default)s). Use '*' for a full snapshot.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face model repo id (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where model files are saved (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Model revision/branch/tag to download (default: main)",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "Missing dependency: huggingface_hub. Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model '{args.repo_id}' to '{output_dir}'...")

    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        local_dir=str(output_dir),
        allow_patterns=args.allow_patterns,
        token=args.token,
    )

    print(f"Download complete: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
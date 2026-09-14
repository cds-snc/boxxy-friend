#!/usr/bin/env python3
"""Download the Gemma 4 E2B ONNX model snapshot from Hugging Face.

Usage:
  python setup.py
  python setup.py --repo-id onnx-community/gemma-4-E2B-it-ONNX --output-dir models/gemma-4-E2B-it-ONNX
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_REPO_ID = "google/gemma-4-E2B-it"
DEFAULT_OUTPUT_DIR = "models/gemma-4-E2B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Gemma 4 E2B model snapshot from Hugging Face."
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
        local_dir_use_symlinks=False,
        token=args.token,
        resume_download=True,
    )

    print(f"Download complete: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
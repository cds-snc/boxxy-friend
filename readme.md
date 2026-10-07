# Notes on License

While Boxxy is MIT licensed, the License of the LLM you use might be vastly different, please evaluate and only use models appropriately. Gemma (default atm) is Apache 2.0

# To Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup.py
playwright install
playwright install chromium
```

# Run this every time

```bash
source .venv/bin/activate
```

# Testing Env

```bash
python3 test.py
```

# Running Boxxy

```bash
python3 boxxy.py
```

This first opens a model-selection screen listing complete local Transformers model folders and standalone GGUF files in the repository's `models/` directory (regardless of your current working directory). GGUF files can be placed directly in `models/` or inside a model folder; each quantization is listed separately as `folder/filename.gguf`. Select a model and click **Open Boxxy** to enter the main area, or **Cancel** to exit without loading a model. **Refresh** rescans the folder after a download.

The initial check requires non-empty JSON objects in `config.json` (including `model_type`), `processor_config.json`, and `tokenizer_config.json`, tokenizer data, and non-empty Safetensors or PyTorch weights. Sharded models must have an index referencing all their weight files. Incomplete folders are excluded with a reason displayed on the selection screen. This checks snapshot completeness, not weight integrity or runtime compatibility; incompatible models still report a load error in the main area.

GGUF models do not require JSON configs or separate tokenizer files. Their initial check verifies a non-empty GGUF v2/v3 header with tensors and metadata, not the full file's integrity. Split GGUFs and `mmproj` projection files are not supported. GGUF inference uses `llama-cpp-python`, the embedded chat template, an 8192-token context, and GPU offloading where supported. Tool responses use schema-constrained JSON and are converted to Boxxy's usual response structure; Transformers inference is unchanged. Model architecture and chat-template compatibility are verified at load/generation time.

The selected model loads in the background; once it's ready, a haiku about black box testing and the generation time appear in the bottom bar. Enter a URL and press **Start** to run Mode 1. The left pane shows the live ARIA tree, the right pane logs Boxxy's actions, and errors appear in a red bar instead of closing the app.

# Architecture and model support

## Application flow

`boxxy.py` opens the Tkinter interface in `modules/gui.py`. The model-selection
screen lists complete local Transformers snapshots and standalone GGUF files in
the repository's `models/` directory, regardless of the current working
directory. Each GGUF quantization is listed separately, including files one
folder deep as `folder/filename.gguf`. **Refresh** rescans the directory;
**Cancel** exits without loading a model.

The selected model loads in a background worker on a dedicated preparation
screen, then generates a haiku about black box testing. The screen shows
elapsed time and clearly labeled approximate progress/ETA. Neither backend
reports granular progress: estimates use initial budgets of 60 seconds to load
and 30 seconds to generate. Progress stays below the phase boundary until the
worker confirms completion, and an overrun is labeled rather than promising
zero time remaining. These estimates do not control timeouts.

Once both steps succeed, the screen displays the haiku and measured load,
generation, and total times for that readiness run. **Okay** is enabled only
then; pressing it builds the main UI with the already-loaded model, without
loading or generating again. Failures (including empty haikus) display their
details and offer **Retry** (repeats both steps) or **Choose another model**.
**Cancel** closes the app without entering the main UI.

**Start** runs Mode 1 against the entered URL. `modules/mode1.py` coordinates
the model and the Playwright browser in `modules/browser.py`, using ARIA
snapshots to choose actions. The GUI shows the live ARIA tree and action log;
errors appear in a red bar.

Tkinter stays on the main thread. Model loading, generation, and browsing run
in workers and communicate with the UI through a queue polled by the Tk main
loop.

## Model validation

`modules/models.py` checks local files before listing them. Rejected entries
appear with a reason on the selection screen. These checks establish snapshot
completeness, not weight integrity or runtime compatibility; load and generation
errors are still possible.

### Transformers snapshots

A snapshot needs non-empty JSON objects in `config.json` (including
`model_type`), `processor_config.json`, and `tokenizer_config.json`, tokenizer
data, and non-empty Safetensors or PyTorch weights. Sharded models must have an
index referencing all their weight files.

`modules/llm.py` uses `AutoProcessor` and `AutoModelForMultimodalLM` to load
these snapshots and moves the model to `mps`, requiring a supported Apple GPU
and PyTorch MPS environment.

### GGUF models

GGUF files need no separate JSON configs or tokenizer files. Validation checks
a non-empty GGUF v2/v3 header with tensors and metadata, not the full file's
integrity. Split GGUFs and `mmproj` projection files are unsupported.

Inference uses `llama-cpp-python`, the embedded chat template, a 32000-token
context, and GPU offloading where supported. Architecture and chat-template
compatibility are checked at load/generation time. Tool responses use
schema-constrained JSON and are converted to Boxxy's usual response structure.

## Downloading alternative models

`setup.py` defaults to downloading only Ornith 1.5 9B Q4_K_M into `models/`.
Override `--repo-id`, `--output-dir`, `--revision`, and `--allow-patterns` as
needed. For example, to download a complete Transformers snapshot:

```bash
python setup.py --repo-id google/gemma-4-E2B-it \
  --output-dir models/gemma-4-E2B-it --allow-patterns '*'
```

For gated or private repositories, provide `HF_TOKEN` or `--token` with
appropriate access. Downloading a model does not guarantee runtime support.
The smoke script always uses the default Ornith file; use the model selector
to try alternatives.

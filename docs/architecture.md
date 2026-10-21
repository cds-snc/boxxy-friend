# Architecture and model support

## Application flow

`boxxy.py` opens the Tkinter interface in `modules/gui.py`. The model-selection
screen lists complete local Transformers/MLX snapshots and standalone GGUF files in
the repository's `models/` directory, regardless of the current working
directory. Each GGUF quantization is listed separately, including files one
folder deep as `folder/filename.gguf`. **Refresh** rescans the directory;
**Cancel** exits without loading a model.

The selected model loads in a background worker on a dedicated preparation
screen, then generates a haiku about black box testing. The screen shows
elapsed time and clearly labeled approximate progress/ETA. None of the backends
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

The main UI first opens **Define smoke test**. `modules/objective.py` holds an
immutable, validated GC Forms preset and renders the visible Scope / Probe /
Halt contract. Editable fields are URL, title, input count and action/time/
no-progress limits. Types cycle through short answer, long answer and radio
buttons with fixed synthetic labels; only the first input is required.
Unsupported workflow edits are explicitly outside this first preset.

**Discuss** uses the already-loaded model without browser tools. Refinement
receives the current structured draft and at most six recent conversation
messages, and can return one `propose_objective` call using the selected
backend's tool-response format.
Code validates the proposal; the UI displays a diff, requires **Apply
proposal**, then **Approve objective** before **Start**. Editing invalidates
approval. Refinement and execution cannot use the model concurrently.
The execution prompt contains the approved contract, not the discussion.

**Start** runs Mode 1 against the approved URL. `modules/mode1.py` coordinates
the model and the Playwright browser in `modules/browser.py`, using ARIA
snapshots to choose actions. The GUI shows the live ARIA tree and action log;
errors appear in a red bar.

The bounded GUI runner keeps verified checkpoints separate from model-written
progress notes. Dedicated tools check required-empty rejection in Test mode,
fill all approved inputs with synthetic values and check valid-input
validation, then capture a download event and save/validate GC Forms schema-v1
JSON. Export validation checks title, layout, count, types, labels, required
rules and radio options. A click or model assertion cannot complete the run.
Static confirmation content in signed-out preview is not success evidence;
missing validation is a blocking result, not a passed check.

The browser starts with a fresh context and guards document navigations
before requests leave the approved origin and form-builder paths. Auth,
publishing, settings and response paths are outside scope. Browser operations
check cancellation and consume the action budget before attempting actions.
Time limits are cooperative: they are checked between operations, including
after generation, rather than forcibly interrupting inference or Playwright.
Repeated no-progress turns stop the run. Optional zero-based target indices
allow the model to address repeated builder labels such as "Question".

Every bounded run saves `runs/<run-id>/result.json` with the approved
objective, outcome, evidence and recent progress. Validated exports are saved
as `form.json` in that directory; failed exports remain there for diagnosis.
Outcomes are `completed`, `blocked`, `budget_exhausted` and `stopped_by_user`.
An interrupted process can leave a `running` result; this is not success.
Programmatic callers passing a custom `goal` without an `objective` retain the
legacy exploratory loop; the GUI always uses the bounded contract.

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

### MLX models

On macOS with Apple Silicon (`arm64`), `mlx-lm` loads MLX snapshots directly,
including Ornith 1.5 9B MLX 6-bit. It is installed by `requirements.txt` on
supported platforms; a missing dependency or unsupported platform produces an
explicit load error.

Discovery and loading recognize `quantization` or `quantization_config` objects
with `bits` and `group_size` and without another backend's `quant_method`.
An explicit hyphen- or underscore-separated `mlx` folder-name component also
selects this backend, including for unquantized snapshots.

A snapshot needs non-empty `config.json` (including `model_type`) and
`tokenizer_config.json` objects, tokenizer data, and non-empty Safetensors
weights. Sharded weights need an index referencing every shard.
`processor_config.json` is not required. The tokenizer must provide a chat
template, and the architecture must be supported by the installed MLX-LM
version. This backend handles text only, not image/video inputs.

Generation uses the snapshot's tokenizer/chat template with thinking disabled.
Plain-text generation retains a 2048-token output limit; MLX tool generation
has an 8192-token limit to accommodate structured responses and progress notes.
MLX tool generation uses `lm-format-enforcer` to
mask tokens that violate the same JSON response schema used by GGUF, including
tool names and argument shapes. Tool generation uses temperature 0.2 rather
than greedy decoding to reduce repetition. Its schema bounds all otherwise
unbounded strings and arrays, not just `thoughts`: thoughts, intents, and each
plan/done entry default to 120 characters; other strings default to 1024;
arrays default to 15 items. Existing explicit schema limits (including objective
URL/title limits and the tool-call count) are preserved. Generation disallows
undeclared object fields, follows schema field order, and allows at most one
consecutive whitespace character outside strings. These are generation
constraints, not post-generation truncation or JSON repair.
Tokenizer preprocessing is cached for the loaded model; every generation attempt
gets a fresh schema parser. This preprocessing can make the first tool request
slower. Both backends validate
response shape, tool names, argument objects, and the one-action/one-progress
limit. They make at most two attempts, logging malformed output before
retrying or raising an error. MLX tool output is streamed internally so errors
report the actual stop reason and prompt/completion token counts. A token-limit
stop triggers the existing retry; if both attempts fail, an explicit output-limit
error is raised and incomplete tool calls are never executed.
Schema filtering does not prevent token-limit truncation or guarantee that a
syntactically valid tool call selects the correct page element; these still
produce explicit failures rather than guessed or repaired browser actions.
Plain text and tool calls retain Boxxy's `raw`, `thoughts`, `text`, and
`tool_calls` response fields.

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

# Boxxy

Boxxy uses a local language model and Playwright to interact with web pages.

## Setup

Use Python with Tkinter available for the desktop UI. Run these commands from
the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup.py
playwright install chromium
```

Setup downloads only `Ornith-1.5-9B-Q4_K_M.gguf` from
[ornith-ai/Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF)
into `models/`. Internet access is required for installation and download.
Use `python setup.py --help` for download options.

## Run

```bash
source .venv/bin/activate
python boxxy.py
```

Select **Ornith-1.5-9B-Q4_K_M.gguf** and click **Open Boxxy**. A dedicated
preparation screen shows approximate progress, estimated time remaining, and
elapsed time while the model loads and generates a readiness haiku. Review the
haiku and benchmark (model load, generation, and total time), then press
**Okay** to enter the main UI. Estimates are rough and depend on your hardware
and model; measured benchmark times are shown when complete. If preparation
fails, use **Retry** or **Choose another model**; **Cancel** exits.
The **Define smoke test** tab opens before any browser is launched. Review the
visible **Scope / Probe / Halt** contract, edit its fields, or enter a request
and select **Discuss** to refine it with the loaded model. Proposed changes
show a diff and do not take effect until **Apply proposal** is selected.
Select **Approve objective**, then **Start**. Any edit invalidates approval;
the contract is frozen while the test runs.

The first preset supports English GC Forms form-builder URLs (only test
deployments you are authorized to use). It starts signed out with a fresh
browser context. Defaults are three inputs: required short answer, optional
long answer, and optional radio buttons. Larger counts repeat these types;
the first input remains required. Boxxy uses synthetic answers, checks empty
required-field rejection and valid-input validation in Test mode, then saves
and validates the downloaded form JSON. Sign-in, publishing, settings,
responses, and navigation outside the scoped form builder are blocked.

Limits default to 100 browser-action attempts, 900 seconds, and five
consecutive no-progress turns. Compound dropdown selection counts as one
logical action; automated input checks count each fill/check and Submit.
Time and cancellation are checked between operations, not by interrupting
an in-flight model generation or Playwright call.

Outcomes distinguish **completed**, **blocked**, **budget_exhausted**, and
**stopped_by_user**. A failed required check stops the run immediately.
Some signed-out GC Forms previews do not expose submission validation:
Boxxy reports that as blocked, rather than treating the static confirmation
preview as success. The signed-out test does not submit or verify real responses.
Each run saves its approved contract, evidence, and outcome to
`runs/<run-id>/result.json`; a downloaded export is saved alongside it as
`form.json`. These local files are excluded from git.

The left pane shows the page's ARIA tree; the right pane logs actions.
Errors appear in a red bar. Use **Refresh** on the model-selection screen if
you add a model after opening the app.

## Smoke test

After setup, from the repository root:

```bash
source .venv/bin/activate
python test.py
```

This loads the default Ornith model, generates a poem, and launches headless
Chromium to visit `https://example.com` (internet access required).
Success prints the generated response and `Smoke test passed`, saves
`example.png` in the current directory, and exits without an exception.
For a desktop smoke check, also follow the **Run** steps above.

## License and details

Boxxy is [MIT licensed](LICENSE). Model licenses apply separately; review the
[model's license](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF)
before use.

See [architecture and model support](docs/architecture.md) for implementation
details and alternative models.

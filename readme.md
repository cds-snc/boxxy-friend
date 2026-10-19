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
Enter the URL of a site you are authorized to test and press **Start**.
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

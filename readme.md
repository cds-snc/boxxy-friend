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

This opens the Boxxy GUI. The model loads in the background; once it's ready, a haiku about black box testing and the generation time appear in the bottom bar. Enter a URL and press **Start** to run Mode 1. The left pane shows the live ARIA tree, the right pane logs Boxxy's actions, and errors appear in a red bar instead of closing the app.

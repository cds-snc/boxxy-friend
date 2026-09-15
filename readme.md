# To Install

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup.py
playwright install
playwright install chromium

# Run this every time

source .venv/bin/activate
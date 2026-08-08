set shell := ["bash", "-uc"]

setup:
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install -r requirements.txt
    .venv/bin/python scripts/download_model.py

repl:
    .venv/bin/python repl.py

test:
    .venv/bin/pytest -v

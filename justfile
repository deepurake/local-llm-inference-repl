set shell := ["bash", "-uc"]

setup:
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install -r requirements.txt
    mkdir -p models
    test -f models/Qwen3.5-0.8B-Q4_K_M.gguf || curl -L -o models/Qwen3.5-0.8B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"

repl:
    .venv/bin/python repl.py

test:
    .venv/bin/pytest -v

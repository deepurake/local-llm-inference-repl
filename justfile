set shell := ["bash", "-uc"]

setup-model:
    mkdir -p models
    test -f models/Qwen3.5-0.8B-Q4_K_M.gguf || curl -L -o models/Qwen3.5-0.8B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"

setup: setup-model
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install -r requirements.txt
    mkdir -p certs
    test -f certs/server.crt || openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/server.key -out certs/server.crt -days 365 -subj "/CN=localhost"

gen-proto:
    .venv/bin/python -m grpc_tools.protoc -Iproto --python_out=. --grpc_python_out=. proto/inference.proto

repl:
    .venv/bin/python repl.py

test:
    .venv/bin/pytest -v

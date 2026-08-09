# Docker Compose & Minikube Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Host `repl.py`'s conversation logic behind a FastAPI WebSocket endpoint — alongside its existing terminal CLI, not replacing it — and deploy that service via Docker Compose and minikube, calling the gRPC inference server that stays native on the host.

**Architecture:** `repl.py` already has a synchronous, tested `run_turn(stub, tokenizer, context_tokens, user_text)` that calls the inference server's gRPC `Generate` RPC and returns the complete response once the turn finishes (not a token stream — see Global Constraints). A new `/ws/chat` WebSocket endpoint runs that same `run_turn` in a worker thread per message and relays one consolidated JSON response per turn. The service is containerized (single Dockerfile) and deployed two ways: Docker Compose for local dev, minikube manifests for cluster-style local dev — both reaching the inference server, which is never containerized.

**Tech Stack:** FastAPI, Uvicorn, `anyio.to_thread` (to run the existing blocking `run_turn` off the event loop), Docker, Docker Compose, minikube/kubectl.

## Global Constraints

- **`repl.py` already exists and is implemented/tested on `main`** — `run_turn`, `load_tokenizer`, `print_candidates`, `print_prompt_tokens`, and a terminal `main()` are real, reviewed code (see `tests/test_repl.py`). This plan does not rewrite them.
- **The terminal CLI stays.** `main()` and its `input()` loop are kept as-is for native/local dev via `just repl`. The WebSocket service is a second, additive entry point sharing `run_turn` — not a replacement.
- **`run_turn` is synchronous and returns once, not per-token.** It returns `(context_tokens, response_text, interrupted, num_generated)` after the whole turn completes (or after its own internal `RpcError`/`KeyboardInterrupt` handling). The WebSocket protocol reflects this: one `{"type": "response", ...}` JSON message per user turn, not a token-by-token stream.
- **Known gap: no cancel-on-disconnect.** Because `run_turn` runs synchronously in a worker thread, a WebSocket disconnect mid-turn cannot cancel it cooperatively (unlike an async-native design) — the turn runs to completion server-side regardless; a disconnected client just never receives the result. Not solved by this plan.
- **Known gap, not solved by this plan:** the inference server's TLS cert is `CN=localhost` only. Dialing it from a container via `host.docker.internal` or `host.minikube.internal` will fail TLS hostname verification. This will surface as a connection error inside `run_turn` (already handled — it prints to server logs and returns an empty response with the original context restored), not a crash. Fixing the TLS mismatch itself is deferred — see the design spec's Future section.
- No auth, no session persistence, no multi-replica — one WebSocket connection is one conversation, state lives in memory for the connection's lifetime only.
- `INFERENCE_SERVER_HOST` / `INFERENCE_SERVER_PORT` env vars override `config.GRPC_HOST` / `config.GRPC_PORT` — needed so the containerized service can reach the host machine (`host.docker.internal` / `host.minikube.internal`) while native `just repl` keeps defaulting to `localhost`.
- minikube: docker driver assumed (matches this project's macOS + Docker Desktop tooling). Access via `minikube service` / `kubectl port-forward` against a ClusterIP Service — no Ingress.
- Model file handling differs deliberately by target: Docker Compose bind-mounts `./models` read-only (fast iteration); minikube bakes the model into the image at build time (hostPath/`minikube mount` into the VM is driver-fragile — baking in is the reliable choice for a single-dev-machine cluster).

---

### Task 1: FastAPI WebSocket app in `repl.py`

**Files:**
- Modify: `repl.py`
- Modify: `requirements.txt` (add `fastapi`, `uvicorn[standard]`, `httpx`, `anyio`)
- Modify: `tests/test_repl.py` (append WebSocket tests, reusing the existing `FakeStub`/`FakeCall`/`FakeEvent`/`FakeTokenizer`)

**Interfaces:**
- Consumes: existing `run_turn(stub, tokenizer, context_tokens, user_text) -> (context_tokens, response_text, interrupted, num_generated)`, `load_tokenizer()`, `config.CERT_PATH`, `config.GRPC_HOST`, `config.GRPC_PORT` — all already defined in `repl.py`/`config.py`, unchanged.
- Produces: module-level `app` (FastAPI instance) in `repl.py` with a `/ws/chat` WebSocket endpoint; `app.state.stub` / `app.state.tokenizer` as the production/test seam (lifespan only builds them if not already set — tests pre-seed fakes, production leaves them unset so lifespan builds the real stub/tokenizer). Consumed by Task 2's Dockerfile (`uvicorn repl:app`).

- [ ] **Step 1: Add web dependencies**

Edit `requirements.txt`, add four lines:
```
fastapi
uvicorn[standard]
httpx
anyio
```

Run: `.venv/bin/pip install fastapi "uvicorn[standard]" httpx anyio`

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_repl.py` (reusing the `FakeStub`, `FakeCall`, `FakeEvent`, `FakeTokenizer` classes already defined earlier in that file):

```python
from fastapi.testclient import TestClient

from repl import app


def test_chat_websocket_returns_full_response_per_turn():
    call = FakeCall(events=[FakeEvent(100), FakeEvent(101), FakeEvent(999, is_eos=True)])
    app.state.stub = FakeStub(call)
    app.state.tokenizer = FakeTokenizer()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("hello")
            response = websocket.receive_json()

    assert response["type"] == "response"
    assert response["text"] == "100 101"
    assert response["interrupted"] is False
    assert response["tokens_generated"] == 2


def test_chat_websocket_ignores_blank_messages():
    call = FakeCall(events=[FakeEvent(100, is_eos=True)])
    app.state.stub = FakeStub(call)
    app.state.tokenizer = FakeTokenizer()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("   ")
            websocket.send_text("hello")
            response = websocket.receive_json()

    assert response["type"] == "response"
    assert response["text"] == ""


def test_chat_websocket_reuses_context_across_turns():
    call = FakeCall(events=[FakeEvent(100, is_eos=True)])
    stub = FakeStub(call)
    app.state.stub = stub
    app.state.tokenizer = FakeTokenizer()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_text("first")
            websocket.receive_json()
            websocket.send_text("second")
            websocket.receive_json()

    assert len(stub.requests) == 2
    assert len(stub.requests[1].context_tokens) > len(stub.requests[0].context_tokens)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_repl.py -v -k websocket`
Expected: FAIL — `ImportError: cannot import name 'app' from 'repl'`

- [ ] **Step 4: Add the FastAPI app to `repl.py`**

Add `os` to the existing `import time` line's group, add a new `import anyio`, `from contextlib import asynccontextmanager`, and `from fastapi import FastAPI, WebSocket, WebSocketDisconnect` to the imports at the top of `repl.py`. Then add `INFERENCE_HOST`/`INFERENCE_PORT` and a `create_stub()` helper (extracted from `main()`'s inline channel setup), and the FastAPI app itself. The full new file:

```python
import os
import time
from contextlib import asynccontextmanager

import anyio
import grpc
import inference_pb2
import inference_pb2_grpc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from llama_cpp import Llama

import config
from context import truncate_context

INFERENCE_HOST = os.environ.get("INFERENCE_SERVER_HOST", config.GRPC_HOST)
INFERENCE_PORT = int(os.environ.get("INFERENCE_SERVER_PORT", config.GRPC_PORT))


def load_tokenizer():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Run `just setup` to download it."
        )
    return Llama(model_path=str(config.MODEL_PATH), vocab_only=True, verbose=False)


def create_stub():
    with open(config.CERT_PATH, "rb") as f:
        trusted_certs = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
    channel = grpc.secure_channel(f"{INFERENCE_HOST}:{INFERENCE_PORT}", credentials)
    return inference_pb2_grpc.InferenceStub(channel)


def print_candidates(tokenizer, candidates, chosen_id):
    print("  candidates:")
    for candidate in candidates:
        text = tokenizer.detokenize([candidate.token_id], special=True).decode(
            "utf-8", errors="replace"
        )
        marker = " <-- chosen" if candidate.token_id == chosen_id else ""
        print(f"    {candidate.probability:6.3f}  {text!r}{marker}")


def print_prompt_tokens(tokenizer, prompt_tokens):
    print("  prompt tokens:")
    for tid in prompt_tokens:
        piece = tokenizer.detokenize([tid], special=True).decode("utf-8", errors="replace")
        print(f"    {tid}: {piece!r}")


def run_turn(stub, tokenizer, context_tokens, user_text):
    original_context_tokens = list(context_tokens)
    prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens = tokenizer.tokenize(
        prompt.encode("utf-8"), add_bos=(len(context_tokens) == 0), special=True
    )
    print_prompt_tokens(tokenizer, prompt_tokens)
    context_tokens = truncate_context(
        context_tokens + prompt_tokens,
        config.MAX_CONTEXT_TOKENS - config.MAX_NEW_TOKENS - config.CLOSER_TOKEN_MARGIN,
    )

    request = inference_pb2.GenerateRequest(
        context_tokens=context_tokens,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
    )

    generated_tokens = []
    saw_eos = False
    interrupted = False
    call = stub.Generate(request)

    try:
        for event in call:
            print_candidates(tokenizer, event.candidates, event.token_id)
            context_tokens.append(event.token_id)
            if event.is_eos:
                saw_eos = True
            else:
                generated_tokens.append(event.token_id)
    except KeyboardInterrupt:
        call.cancel()
        interrupted = True
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
            print(f"\n[error] {e.details()}\n")
        else:
            print(
                f"\n[error] Could not reach inference server at "
                f"{INFERENCE_HOST}:{INFERENCE_PORT} — is `just serve` running? "
                f"({e.code()}: {e.details()})\n"
            )
        return original_context_tokens, "", False, 0

    if not saw_eos:
        closer = tokenizer.tokenize(b"<|im_end|>\n", add_bos=False, special=True)
        context_tokens.extend(closer)

    response_text = tokenizer.detokenize(generated_tokens, special=False).decode(
        "utf-8", errors="ignore"
    )
    return context_tokens, response_text, interrupted, len(generated_tokens)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "stub"):
        app.state.stub = create_stub()
    if not hasattr(app.state, "tokenizer"):
        app.state.tokenizer = load_tokenizer()
    yield


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/chat")
async def chat(websocket: WebSocket) -> None:
    await websocket.accept()
    context_tokens: list[int] = []
    stub = websocket.app.state.stub
    tokenizer = websocket.app.state.tokenizer

    while True:
        try:
            user_text = await websocket.receive_text()
        except WebSocketDisconnect:
            return

        if not user_text.strip():
            continue

        start = time.perf_counter()
        context_tokens, response_text, interrupted, num_generated = await anyio.to_thread.run_sync(
            run_turn, stub, tokenizer, context_tokens, user_text
        )
        elapsed = time.perf_counter() - start
        tokens_per_sec = num_generated / elapsed if elapsed > 0 else 0.0

        try:
            await websocket.send_json(
                {
                    "type": "response",
                    "text": response_text,
                    "interrupted": interrupted,
                    "tokens_generated": num_generated,
                    "elapsed_s": elapsed,
                    "tokens_per_sec": tokens_per_sec,
                }
            )
        except WebSocketDisconnect:
            return


def main():
    tokenizer = load_tokenizer()
    stub = create_stub()
    context_tokens: list[int] = []

    print(f"Connected to inference server at {INFERENCE_HOST}:{INFERENCE_PORT}")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue

        start = time.perf_counter()
        context_tokens, response_text, interrupted, num_generated = run_turn(
            stub, tokenizer, context_tokens, user_text
        )
        elapsed = time.perf_counter() - start

        if interrupted:
            print(f"\n[interrupted] Partial response: {response_text}\n")
        else:
            tokens_per_sec = num_generated / elapsed if elapsed > 0 else 0.0
            print(f"\nAssistant: {response_text}")
            print(f"[{elapsed:.2f}s, ~{tokens_per_sec:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
```

Note what changed from the current file: `main()`'s inline TLS-channel setup is now `create_stub()` (called from both `main()` and the new `lifespan`), and the two hardcoded `config.GRPC_HOST}:{config.GRPC_PORT}` print sites now read the env-overridable `INFERENCE_HOST`/`INFERENCE_PORT` instead — everything else in `run_turn`, `load_tokenizer`, `print_candidates`, `print_prompt_tokens`, and `main()`'s loop is untouched.

- [ ] **Step 5: Run the full test suite to verify it passes**

Run: `.venv/bin/pytest tests/test_repl.py -v`
Expected: PASS (7 tests — the 4 existing `run_turn` tests plus the 3 new WebSocket tests)

- [ ] **Step 6: Add a `repl-service` justfile recipe for local (non-Docker) runs**

Edit `justfile`, add a new recipe (leave `repl:` untouched):

```
repl-service:
    .venv/bin/uvicorn repl:app --reload --host 0.0.0.0 --port 8000
```

- [ ] **Step 7: Commit**

```bash
git add repl.py requirements.txt justfile tests/test_repl.py
git commit -m "feat: add FastAPI WebSocket service hosting repl.py's conversation logic"
```

---

### Task 2: Dockerfile

**Files:**
- Create: `Dockerfile`

**Interfaces:**
- Consumes: Task 1's `repl:app`, `requirements.txt`.
- Produces: image tagged at build time, `CMD ["uvicorn", "repl:app", "--host", "0.0.0.0", "--port", "8000"]`, listening on container port 8000. Consumed by Task 3 (`docker-compose.yml`) and Task 4 (minikube `deployment.yaml`).

- [ ] **Step 1: Write the Dockerfile**

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py inference_pb2.py inference_pb2_grpc.py context.py repl.py ./
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "repl:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Verify the image builds**

Prerequisite: the model file must be present locally (`just setup-model` if not already run).

Run: `docker build -t repl:local .`
Expected: build completes successfully.

- [ ] **Step 3: Verify the module imports cleanly in the container**

Run: `docker run --rm repl:local python -c "import repl; print('ok')"`
Expected: prints `ok` — confirms `repl.py` imports without requiring the model, certs, or a running inference server (`app.state.stub`/`.tokenizer` are only built lazily in `lifespan` when the app actually starts, not at import time).

- [ ] **Step 4: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile for the repl web service"
```

---

### Task 3: Docker Compose

**Files:**
- Create: `docker-compose.yml`
- Modify: `justfile` (add `compose-up` recipe)

**Interfaces:**
- Consumes: Task 2's `Dockerfile`, `config.GRPC_PORT` (50051, used as the `INFERENCE_SERVER_PORT` value).
- Produces: a running `repl` service reachable at `http://localhost:8000`, `ws://localhost:8000/ws/chat`.

- [ ] **Step 1: Write docker-compose.yml**

Create `docker-compose.yml`:

```yaml
services:
  repl:
    build: .
    ports:
      - "8000:8000"
    environment:
      - INFERENCE_SERVER_HOST=host.docker.internal
      - INFERENCE_SERVER_PORT=50051
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./models:/app/models:ro
      - ./certs:/app/certs:ro
```

- [ ] **Step 2: Add the `compose-up` justfile recipe**

Edit `justfile`, add:

```
compose-up:
    docker compose up --build
```

- [ ] **Step 3: Manually verify the compose service starts and accepts connections**

Prerequisites: `just setup` has been run (models + certs present), `just serve` is running in a separate terminal on the host.

Run: `just compose-up`

In another terminal, install a throwaway test client and drive one turn:

```bash
.venv/bin/pip install websockets
.venv/bin/python3 - <<'EOF'
import asyncio
import websockets

async def main():
    async with websockets.connect("ws://localhost:8000/ws/chat") as ws:
        await ws.send("hello")
        print(await ws.recv())

asyncio.run(main())
EOF
```

Expected: given the known TLS gap (see Global Constraints), this will most likely print a `{"type": "response", "text": "", ...}` line — `run_turn`'s existing `grpc.RpcError` handling already caught the TLS handshake failure, logged it server-side, and returned an empty response rather than crashing. That confirms the WebSocket layer, container networking, and existing error handling all work end-to-end; full token generation requires the deferred TLS fix.

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml justfile
git commit -m "feat: add Docker Compose deployment for the repl service"
```

---

### Task 4: Minikube

**Files:**
- Create: `k8s/deployment.yaml`
- Create: `k8s/service.yaml`
- Modify: `justfile` (add `minikube-deploy` recipe)

**Interfaces:**
- Consumes: Task 2's `Dockerfile` (built as `repl:local` into minikube's docker daemon), `config.GRPC_PORT` (50051).
- Produces: a running `repl` Deployment + ClusterIP Service in minikube, reachable via `minikube service repl --url`.

- [ ] **Step 1: Write the Deployment manifest**

Create `k8s/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: repl
spec:
  replicas: 1
  selector:
    matchLabels:
      app: repl
  template:
    metadata:
      labels:
        app: repl
    spec:
      containers:
        - name: repl
          image: repl:local
          imagePullPolicy: Never
          ports:
            - containerPort: 8000
          env:
            - name: INFERENCE_SERVER_HOST
              value: host.minikube.internal
            - name: INFERENCE_SERVER_PORT
              value: "50051"
          volumeMounts:
            - name: certs
              mountPath: /app/certs
              readOnly: true
      volumes:
        - name: certs
          configMap:
            name: repl-certs
```

- [ ] **Step 2: Write the Service manifest**

Create `k8s/service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: repl
spec:
  selector:
    app: repl
  ports:
    - port: 8000
      targetPort: 8000
```

- [ ] **Step 3: Add the `minikube-deploy` justfile recipe**

Edit `justfile`, add:

```
minikube-deploy:
    eval $(minikube docker-env) && docker build -t repl:local .
    kubectl create configmap repl-certs --from-file=certs/server.crt -o yaml --dry-run=client | kubectl apply -f -
    kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
```

- [ ] **Step 4: Manually verify the deployment**

Prerequisites: `just setup` has been run, `just serve` is running on the host, `minikube start` has been run (docker driver).

Run: `just minikube-deploy`
Run: `kubectl rollout status deployment/repl`
Expected: deployment becomes ready.

Run: `minikube service repl --url` to get the local URL, then adapt Task 3 Step 3's `websockets` test script to point at that URL's host:port instead of `localhost:8000`.

Expected: same as Task 3 Step 3 — an empty `response` from the TLS handshake failure against `host.minikube.internal`, confirming the deployment, networking, and existing error handling all work; full generation is blocked on the deferred TLS fix.

- [ ] **Step 5: Commit**

```bash
git add k8s/deployment.yaml k8s/service.yaml justfile
git commit -m "feat: add minikube deployment manifests for the repl service"
```

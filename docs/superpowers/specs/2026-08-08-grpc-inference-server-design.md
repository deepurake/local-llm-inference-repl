# gRPC Inference Server — Design

## Context

This splits the current single-process `repl.py` into two processes communicating over gRPC: a native macOS **inference server** that owns the model and does the GPU/Metal-accelerated generation work, and `repl.py` itself, converted into a **gRPC client** that keeps owning all conversation logic (ChatML formatting, context tracking, truncation, interrupt handling) — just calling out to the server instead of `llama_cpp` directly.

This is the first step toward a larger, explicitly deferred architecture: conversation logic eventually moving into a containerized service (Docker Compose and/or minikube), with inference staying native on macOS for Metal acceleration. That containerized service, and its deployment tooling, are **out of scope for this spec** — this spec only proves out the inference server and its gRPC contract, with `repl.py` as its first real client.

## Goal

A working `inference_server.py` (native macOS process, GPU/Metal-accelerated) exposing a single gRPC streaming RPC for text generation, plus a `repl.py` fully converted to call it — reproducing today's exact behavior and UX, just across a network boundary instead of in-process.

## Out of scope

- Docker Compose, minikube, or any containerization
- The future conversation-logic service (this spec's `repl.py` *is* the conversation-logic client, running natively, not containerized)
- Any change to the model, sampling algorithm, or ChatML format
- Fixing the three deferred Phase 1 limitations (missing-model traceback, Ctrl+C not covering initial re-eval, no truncation notice) — they carry over unchanged, just relocated

## Architecture

Two local processes on the same machine for this spec, run separately: `just serve` (the inference server, one terminal) and `just repl` (the client, another terminal). No process supervision or auto-start between them.

**Security:** TLS on the gRPC channel via a self-signed certificate (`CN=localhost`), generated once by `just setup` into a gitignored `certs/` directory. This is plaintext-equivalent trust today (both processes are the same user, same machine) but keeps the wiring in place for when a future networked deployment calls this server across a real boundary.

**Config:** a shared `config.py` module holds every constant both processes need — `MAX_CONTEXT_TOKENS`, `MAX_NEW_TOKENS`, `CLOSER_TOKEN_MARGIN`, `TEMPERATURE`, `TOP_P`, `GRPC_PORT = 50051` (gRPC's conventional default port), and the cert/key paths (`certs/server.crt`, `certs/server.key`). Both `repl.py` and `inference_server.py` import from it — one source of truth, no duplicated literals, no RPC needed to keep the two processes' config in sync.

## The gRPC Contract

`proto/inference.proto` defines a single RPC:

```protobuf
syntax = "proto3";

package inference;

service Inference {
  rpc Generate(GenerateRequest) returns (stream GenerateEvent);
}

message GenerateRequest {
  repeated int32 context_tokens = 1;
  int32 max_new_tokens = 2;
  float temperature = 3;
  float top_p = 4;
}

message Candidate {
  int32 token_id = 1;
  string text = 2;
  float probability = 3;
}

message GenerateEvent {
  int32 token_id = 1;
  string text = 2;
  repeated Candidate candidates = 3;
  bool is_eos = 4;
}
```

**Why only `Generate`:** tokenization and detokenization are pure CPU vocabulary/BPE-merge lookups — they happen strictly before prefill and strictly after decode, never touching model weights or doing matrix multiplication, so they don't need Metal/GPU and don't need to be networked. `repl.py` instead loads its own lightweight `llama_cpp.Llama(model_path=..., vocab_only=True)` instance locally (skips loading model weights entirely) and does tokenize/detokenize itself. `GetModelInfo` was considered and dropped too — `n_ctx` and the other tunables live in the shared `config.py` instead, so there's nothing for the client to query at runtime.

**Statelessness:** every `Generate` call carries the full `context_tokens` list; the server holds no session/KV-cache state between calls. All context-window and truncation policy stays entirely client-side (`context.py`'s `truncate_context`, unchanged from Phase 1).

**Context-overflow validation:** the server rejects (`RESOURCE_EXHAUSTED`) any request where `len(context_tokens) + max_new_tokens` exceeds `config.MAX_CONTEXT_TOKENS` — a safety check at the service boundary. This should not normally trigger (the client truncates before calling), but guards against client/server config drift.

**Cancellation:** when `repl.py` catches `KeyboardInterrupt`, it cancels the `Generate` call; the server checks for cancellation between token steps and stops generating early — mirroring today's Ctrl+C behavior. The client's partial response is whatever `GenerateEvent`s arrived before cancellation.

**Wire format:** standard gRPC over HTTP/2 (via `grpcio`), not grpc-web or JSON transcoding. One HTTP/2 stream per `Generate` call; each `GenerateEvent` is sent as a length-prefixed, protobuf-encoded message within that stream's `DATA` frames; the stream closes via HTTP/2 trailers carrying the `grpc-status` (`OK`, `RESOURCE_EXHAUSTED`, or `CANCELLED`).

## File Structure & Components

```
config.py                # shared constants: MAX_CONTEXT_TOKENS, MAX_NEW_TOKENS,
                          # CLOSER_TOKEN_MARGIN, TEMPERATURE, TOP_P, GRPC_PORT, cert paths
proto/
  inference.proto         # the Generate RPC + message definitions
inference_pb2.py          # generated from inference.proto (grpcio-tools), committed
inference_pb2_grpc.py     # generated from inference.proto (grpcio-tools), committed
inference_server.py       # gRPC server: loads the full model (model.py + sampling.py),
                           # implements Generate, binds TLS, listens on config.GRPC_PORT
repl.py                   # gRPC client: owns ChatML formatting, context_tokens,
                           # truncate_context, interrupt→cancel; local vocab_only
                           # Llama instance for tokenize/detokenize; calls Generate
model.py, sampling.py     # unchanged, now only imported by inference_server.py
context.py                # unchanged, still imported by repl.py
certs/                     # gitignored — self-signed cert+key, generated by `just setup`
```

## Data Flow

**Client side (`repl.py`), per turn:**
1. Format the ChatML user turn, tokenize it locally via the `vocab_only=True` instance, print the token IDs/pieces (unchanged from Phase 1).
2. Append to `context_tokens`, truncate via `context.py`'s `truncate_context` against `config.py`'s budget (unchanged logic).
3. Open a `Generate` streaming call with `context_tokens`, `config.MAX_NEW_TOKENS`, `config.TEMPERATURE`, `config.TOP_P`.
4. For each `GenerateEvent` received: print candidates + chosen marker (same shape as Phase 1's `print_candidates`), append the token to `context_tokens`.
5. On `KeyboardInterrupt`: cancel the RPC call; whatever events already arrived form the partial response — same "print partial, don't crash" behavior as Phase 1.
6. After the stream closes (naturally or via cancellation): append the ChatML closer tokens locally (tokenized client-side, same as Phase 1), print the response text (assembled client-side from the streamed token texts) and stats.

**Server side (`inference_server.py`), per `Generate` call:**
1. Validate `len(context_tokens) + max_new_tokens ≤ config.MAX_CONTEXT_TOKENS`; if not, abort with `RESOURCE_EXHAUSTED`.
2. `model.reset()` + `model.eval(context_tokens)` (unchanged from Phase 1's `run_turn`).
3. Loop up to `max_new_tokens`: read logits, `sample_token` (unchanged `sampling.py`), stream a `GenerateEvent`, `model.eval([token_id])`, break on EOS or on gRPC context cancellation.

## Error Handling

- **Server unreachable** (not started, wrong port, TLS handshake failure): `repl.py`'s `Generate` call fails immediately with a gRPC connection error — caught and printed as a clear message ("Could not reach inference server at `localhost:{config.GRPC_PORT}` — is `just serve` running?"), not a raw traceback.
- **Context overflow**: server returns `RESOURCE_EXHAUSTED`; client prints a clear message rather than crashing.
- **Cert/TLS errors**: surfaced as connection errors, same handling as "server unreachable."
- The three deferred Phase 1 limitations (missing-model traceback, Ctrl+C not covering initial re-eval, no truncation notice) carry over unchanged — this spec doesn't fix them, just relocates where the code lives.

## Testing

- **`inference_server.py`**: automated tests spin up a real server instance on a test port (with a test TLS cert) in a `pytest` fixture, then drive it with a real gRPC client stub — verifying `Generate` streams the expected events for a known prompt, validates context-overflow correctly, and stops cleanly on cancellation. No mocking of `sampling.py`/`model.py` — real model, same pattern as Phase 1's `test_model.py`.
- **`repl.py`**: same manual-verification approach as Phase 1 (no interactive terminal in CI), but now requires `just serve` running in the background before driving the client — the non-interactive `subprocess.Popen` driver pattern from Phase 1 still applies, just against the two-process setup.
- **`config.py`**: trivial, no dedicated tests needed (it's just constants).

## Setup & Tooling

- **Dependencies added to `requirements.txt`:** `grpcio`, `grpcio-tools`.
- **`justfile` changes:**
  - New standalone `setup-model` recipe (model download only), with `setup` depending on it:
    ```
    setup-model:
        mkdir -p models
        test -f models/Qwen3.5-0.8B-Q4_K_M.gguf || curl -L -o models/Qwen3.5-0.8B-Q4_K_M.gguf "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf"

    setup: setup-model
        python3 -m venv .venv
        .venv/bin/pip install --upgrade pip
        CMAKE_ARGS="-DGGML_METAL=on" .venv/bin/pip install -r requirements.txt
        test -f certs/server.crt || openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/server.key -out certs/server.crt -days 365 -subj "/CN=localhost"
    ```
  - New `gen-proto` recipe: regenerates `inference_pb2.py`/`inference_pb2_grpc.py` from `proto/inference.proto` via `grpcio-tools`. Run manually whenever the `.proto` changes, not part of `setup` — the generated files are committed, so a fresh clone doesn't need `protoc` just to run `just repl`/`just serve`.
  - New `serve` recipe: runs `inference_server.py`.
  - `repl` recipe unchanged in name; now runs the gRPC client version.
  - `test` recipe unchanged in name; now also covers `inference_server.py`'s new tests.
- **`.gitignore`:** add `certs/`.

## Future (not designed yet)

The conversation-logic service (currently `repl.py`'s responsibilities) eventually moves into a container, deployed via Docker Compose and/or minikube, calling this same inference server across a real network boundary. That work — including revisiting TLS/auth for a genuine network boundary, and whether `repl.py` remains as a native reference client alongside the containerized service — is a separate future spec.

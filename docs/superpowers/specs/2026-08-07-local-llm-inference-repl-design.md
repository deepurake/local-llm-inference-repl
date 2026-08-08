# Local LLM Inference REPL — Design (Phase 1)

## Context

This is Phase 1 of a two-phase learning project to build a local, private ChatGPT-like tool. The end goal (Phase 2, not designed here) is a full chat app: an API server plus a frontend UI, backed by a local model. This spec covers **only** Phase 1: understanding how local LLM inference actually works, hands-on, by instrumenting the generation pipeline ourselves.

Hardware: Apple M5, 24GB RAM (macOS).

## Goal

A single interactive script that lets you type a prompt and watch every stage of local LLM inference happen — tokenization, the token-by-token generation loop, raw logits, candidate probabilities, and your own sampling logic choosing the next token — using `llama-cpp-python`'s lower-level API on top of a small Qwen3.5-0.8B GGUF model.

## Out of scope

- Any HTTP server or persistent background process
- Chat history persistence across sessions
- Containerization (Docker Compose, minikube/Kubernetes)
- Any frontend UI

These are Phase 2 concerns and will be designed in a separate spec once Phase 1 is complete and has informed what that architecture should look like.

## Architecture

Native macOS process, run directly from a Python virtual environment — no containers, so `llama-cpp-python` gets full Metal GPU acceleration on the M5.

**Model:** Qwen3.5-0.8B-Instruct, GGUF format, Q4_K_M quantization (~500-600MB). Chosen for fast iteration during learning — small enough that generation is near-instant, so the focus stays on the mechanics rather than waiting on output.

**Inference library:** `llama-cpp-python`, used via its lower-level API (manual `tokenize` → `eval` → `get_logits` → custom sampling → `detokenize` loop) rather than the high-level `create_chat_completion()` convenience method, so every stage of the pipeline is visible and inspectable.

## Components & Data Flow

Single file: `repl.py` at the project root.

Flow per user turn:

1. Read a line of user input from the terminal.
2. Tokenize it with the model's tokenizer (`llama.tokenize(...)`). Print the token IDs and their string pieces.
3. Loop, one token at a time, up to a max of 256 generated tokens or until an end-of-sequence token, whichever comes first:
   - Run one forward pass (`llama.eval(...)`) to get raw logits for the next token.
   - Print the top-5 candidate tokens with their softmax probabilities.
   - Apply a hand-written sampling function (temperature + top-p) to pick the next token. Show which candidate was picked and why.
   - Detokenize and print that token's text; append it to the running context.
4. Print the full assembled response, plus simple stats (tokens generated, tokens/sec).
5. Loop back to step 1 for the next user turn, keeping the growing context until it nears the context-window limit.

## Error Handling

- **Missing model file:** if the GGUF file isn't found at the expected path, print a clear message pointing to `just setup` rather than a raw stack trace.
- **Context window overflow:** context is capped at a fixed size (4096 tokens) for this small model. When a conversation would exceed it, truncate the oldest turns to keep the REPL usable across a session, with a printed notice.
- **Ctrl+C mid-generation:** caught; print the partial output generated so far plus a note that it was interrupted, then return to the prompt. Does not crash.
- **Model load failure** (corrupt file, incompatible quant, etc.): let the underlying `llama-cpp-python` exception surface with its original message — no wrapping needed for a learning tool.

## Testing

- **Unit tests (`pytest`)** for pure-Python logic that doesn't require the model to be loaded:
  - Sampling function: given a hand-constructed logits array, verify temperature scaling and top-p filtering select the expected token(s).
  - Context-truncation logic: given a token list exceeding the cap, verify it trims the oldest turns correctly.
- **Manual verification:** run `just repl`, confirm tokenization output looks sane for a known input, confirm generation produces coherent text, confirm Ctrl+C doesn't crash the session.
- No integration/API tests — there is no server in this phase.

## Setup & Tooling

- **Dependency management:** Python venv (`.venv`), dependencies declared in `requirements.txt` (or `pyproject.toml`): `llama-cpp-python` (built with Metal support via `CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`), `huggingface_hub` (model download), `pytest` (tests).
- **Model storage:** downloaded into a local `models/` directory, gitignored — not committed to the repo.
- **`justfile` recipes:**
  - `just setup` — create the venv, install dependencies, download Qwen3.5-0.8B-GGUF (Q4_K_M) from Hugging Face into `models/`.
  - `just repl` — activate the venv and launch `repl.py`.
  - `just test` — run the pytest suite.

## Future: Phase 2 (not designed yet)

Once Phase 1 is built and has informed what the generation pipeline actually looks like, Phase 2 will design the full chat app: an API server, conversation persistence, and a frontend UI. That design will also revisit the deployment question raised during this brainstorm — running inference natively for Metal acceleration while containerizing the rest of the stack via Docker Compose (minikube was considered but rejected for local dev, since it forces CPU-only inference inside the VM with no Metal passthrough).

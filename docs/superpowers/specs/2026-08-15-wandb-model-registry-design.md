# W&B Model Registry — Switchable Open-Source Models

Status: approved
Date: 2026-08-15

## Problem

The inference server and repl each hardcode a single model: `config.MODEL_PATH`
points at one local `.gguf` file. There's no way to try a different
open-source model without editing `config.py`. This is a learning exercise in
wiring up a model registry (Weights & Biases Model Registry) so models can be
swapped by name.

## Goals

- Introduce a small model registry keyed by name, backed by the real `wandb`
  SDK call shape, so the integration is genuine reference material.
- Default behavior is unchanged: with no flag, the system runs the currently
  deployed model (`qwen3.5-0.8b`), already cached locally — no network calls.
- Add one additional open-source model (`tinyllama-1.1b`) that can be
  selected explicitly to prove switching works end-to-end.
- Runs without a W&B account: if `WANDB_API_KEY` isn't set (or the wandb
  pull fails), fall back to a direct HTTP download of the model file.

## Non-goals

- Uploading/publishing artifacts *to* W&B (this is pull-only).
- A general plugin system for arbitrary model architectures — both entries
  are llama.cpp-compatible GGUF files, same as today.
- Production-grade retry/resume logic for downloads.

## Design

New module `model_registry.py`:

- `ModelEntry` (frozen dataclass): `name`, `filename`, `wandb_artifact`
  (e.g. `"local-llm-inference-repl/model-registry/tinyllama-1.1b-chat:latest"`),
  `download_url` (HTTP fallback, `None` for the already-local default).
- `MODEL_REGISTRY: dict[str, ModelEntry]` with two entries: `"qwen3.5-0.8b"`
  (default, `download_url=None` since it's already checked in under
  `models/`) and `"tinyllama-1.1b"` (new).
- `DEFAULT_MODEL_NAME = "qwen3.5-0.8b"`.
- `resolve_model_path(name: str) -> Path`:
  1. Unknown name → `ModelRegistryError` listing known names.
  2. If the file is already cached under `models/`, return it immediately
     (no network, no wandb import — this is the path the default model
     always takes).
  3. Else, if `WANDB_API_KEY` is set, try `wandb.Api().artifact(...).download(...)`.
     Any failure (not installed, auth, missing artifact) is caught and logged;
     falls through to step 4 rather than crashing.
  4. Else/fallback: stream-download `entry.download_url` via `requests` into
     `models/`. If there's no `download_url` either, raise
     `ModelRegistryError`.

Call sites:

- `inference_server.py::serve()` gains `--model` (argparse, default
  `DEFAULT_MODEL_NAME`), resolves it, passes the path to the existing
  `create_server(model_path=...)` — that function already accepts an
  override, so `create_server` itself is unchanged.
- `repl.py::load_tokenizer()` takes a `model_path` parameter (was hardcoded
  to `config.MODEL_PATH`); `main()` gains the same `--model` flag and passes
  the resolved path through. This matters for correctness, not just
  symmetry: the repl sends raw token IDs to the server, so its tokenizer
  must come from the same model the server loaded, or token IDs will
  decode to garbage.

## Testing

- `tests/test_model_registry.py`: monkeypatch `wandb.Api` and the HTTP
  download function (no real network/account needed). Cover: (a) an
  already-cached file short-circuits before touching wandb or HTTP, (b)
  wandb path is attempted when `WANDB_API_KEY` is set and its result is
  used, (c) wandb failure falls back to HTTP, (d) no `WANDB_API_KEY` goes
  straight to HTTP, (e) unknown model name raises `ModelRegistryError` with
  the known-names list in the message.
- Existing `tests/test_model.py`, `tests/test_repl.py`,
  `tests/test_inference_server.py` continue to pass unmodified except where
  `load_tokenizer`'s new required parameter needs a call-site update.

## Error handling

- Unknown model name: clear `ModelRegistryError` listing valid names.
- wandb unavailable/unauthenticated: silent-to-user fallback to HTTP (this
  is the expected path until a real W&B account is wired up), logged once
  at the point of fallback.
- No download source at all for a requested model: clear
  `ModelRegistryError`, mirroring the existing "run `just setup`"-style
  message in `model.py`.

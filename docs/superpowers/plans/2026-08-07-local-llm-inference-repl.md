# Local LLM Inference REPL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single interactive REPL script that runs a local Qwen3.5-0.8B GGUF model via `llama-cpp-python`'s low-level API, printing tokenization, per-token logits/candidates, and a hand-written sampling decision at every generation step.

**Architecture:** A native macOS Python process (venv, no containers). Three small pure-logic modules (`sampling.py`, `context.py`, `model.py`) are unit tested in isolation; `repl.py` wires them together into the interactive loop and is verified manually by running it.

**Tech Stack:** Python 3, `llama-cpp-python` (built with Metal support), `numpy`, `huggingface_hub`, `pytest`, `just` (command runner).

## Global Constraints

- Model: `unsloth/Qwen3.5-0.8B-GGUF`, file `Qwen3.5-0.8B-Q4_K_M.gguf`, downloaded into `models/` (gitignored, not committed).
- Context window cap: 4096 tokens total (prompt + generated), enforced by truncating the oldest tokens.
- Max newly generated tokens per turn: 256.
- `llama-cpp-python` must be installed with Metal acceleration: `CMAKE_ARGS="-DGGML_METAL=on"`.
- Chat format is ChatML, matching Qwen3.5's tokenizer config: `<|im_start|>{role}\n{content}<|im_end|>\n`, generation primed with `<|im_start|>assistant\n`. The model's `eos_token` is `<|im_end|>`.
- No HTTP server, no persistent background process, no containers, no frontend — out of scope per the spec.
- `justfile` recipes: `just setup`, `just repl`, `just test`.

---

## File Structure

```
requirements.txt          # deps: llama-cpp-python, huggingface_hub, numpy, pytest
justfile                  # setup / repl / test recipes
.gitignore                # .venv/, models/, __pycache__/, .pytest_cache/
scripts/download_model.py # fetches the GGUF file into models/
sampling.py               # pure: softmax, top_p_filter, top_candidates, sample_token
context.py                # pure: truncate_context
model.py                  # load_model(): wraps Llama() with clear error handling
repl.py                   # orchestrates the interactive loop; not unit tested, verified manually
tests/test_sampling.py
tests/test_context.py
tests/test_model.py
```

---

### Task 1: Project scaffolding and model download

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `justfile`
- Create: `scripts/download_model.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a working `.venv/` with all dependencies installed, and `models/Qwen3.5-0.8B-Q4_K_M.gguf` on disk after `just setup` runs. All later tasks depend on this venv and model file existing.

- [ ] **Step 1: Create `requirements.txt`**

```
llama-cpp-python
huggingface_hub
numpy
pytest
```

- [ ] **Step 2: Create `.gitignore`**

```
.venv/
models/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 3: Create `scripts/download_model.py`**

```python
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "unsloth/Qwen3.5-0.8B-GGUF"
FILENAME = "Qwen3.5-0.8B-Q4_K_M.gguf"


def main() -> None:
    target_dir = Path(__file__).resolve().parent.parent / "models"
    target_dir.mkdir(exist_ok=True)
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=target_dir)
    print(f"Downloaded to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `justfile`**

```
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
```

- [ ] **Step 5: Run setup and verify**

Run: `just setup`

This installs `llama-cpp-python` from source with Metal support (a few minutes to compile) and downloads the ~500MB model file — expect this step to take several minutes on first run.

Expected: no errors; `.venv/bin/python -c "import llama_cpp, numpy, huggingface_hub"` exits with no output; `ls models/` shows `Qwen3.5-0.8B-Q4_K_M.gguf`.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore justfile scripts/download_model.py
git commit -m "chore: add project scaffolding and model download script"
```

---

### Task 2: Sampling module

**Files:**
- Create: `sampling.py`
- Test: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `numpy` (from Task 1's venv).
- Produces:
  - `softmax(logits: np.ndarray) -> np.ndarray`
  - `top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray` (zeroes out excluded entries, renormalizes the rest)
  - `top_candidates(probs: np.ndarray, k: int) -> list[tuple[int, float]]` (sorted descending by probability)
  - `sample_token(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator) -> tuple[int, list[tuple[int, float]]]`
  - Consumed by Task 5 (`repl.py`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_sampling.py`:

```python
import numpy as np
import pytest

from sampling import sample_token, softmax, top_candidates, top_p_filter


def test_softmax_sums_to_one():
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert probs.sum() == pytest.approx(1.0)


def test_softmax_prefers_larger_logit():
    probs = softmax(np.array([1.0, 5.0, 2.0]))
    assert probs.argmax() == 1


def test_top_p_filter_drops_low_probability_tail():
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    filtered = top_p_filter(probs, top_p=0.8)
    assert filtered[3] == 0.0
    assert filtered[0] > 0
    assert filtered[1] > 0
    assert filtered.sum() == pytest.approx(1.0)


def test_top_p_filter_keeps_everything_when_top_p_is_one():
    probs = np.array([0.4, 0.35, 0.25])
    filtered = top_p_filter(probs, top_p=1.0)
    assert np.all(filtered > 0)


def test_top_candidates_returns_sorted_top_k():
    probs = np.array([0.1, 0.5, 0.05, 0.25, 0.08])
    candidates = top_candidates(probs, k=3)
    assert candidates[0] == (1, pytest.approx(0.5))
    assert candidates[1] == (3, pytest.approx(0.25))
    assert candidates[2] == (0, pytest.approx(0.1))


def test_sample_token_only_picks_within_top_p_set():
    logits = np.array([5.0, 4.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(42)
    for _ in range(50):
        token_id, _candidates = sample_token(logits, temperature=1.0, top_p=0.5, rng=rng)
        assert token_id in (0, 1)


def test_sample_token_low_temperature_is_greedy():
    logits = np.array([1.0, 9.0, 2.0])
    rng = np.random.default_rng(7)
    token_id, _candidates = sample_token(logits, temperature=0.01, top_p=1.0, rng=rng)
    assert token_id == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sampling'`

- [ ] **Step 3: Implement `sampling.py`**

```python
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    cutoff = int(np.searchsorted(cumulative, top_p) + 1)
    keep_idx = sorted_idx[:cutoff]

    filtered = np.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    return filtered / filtered.sum()


def top_candidates(probs: np.ndarray, k: int) -> list[tuple[int, float]]:
    top_idx = np.argsort(probs)[::-1][:k]
    return [(int(i), float(probs[i])) for i in top_idx]


def sample_token(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> tuple[int, list[tuple[int, float]]]:
    scaled = logits / max(temperature, 1e-6)
    probs = softmax(scaled)
    filtered = top_p_filter(probs, top_p)
    candidates = top_candidates(filtered, k=5)
    token_id = int(rng.choice(len(filtered), p=filtered))
    return token_id, candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sampling.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add sampling.py tests/test_sampling.py
git commit -m "feat: add sampling module with temperature/top-p logic"
```

---

### Task 3: Context truncation module

**Files:**
- Create: `context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: nothing (pure Python, no dependencies).
- Produces: `truncate_context(tokens: list[int], max_tokens: int) -> list[int]`. Consumed by Task 5 (`repl.py`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_context.py`:

```python
from context import truncate_context


def test_truncate_context_no_op_when_under_limit():
    tokens = [1, 2, 3]
    assert truncate_context(tokens, max_tokens=10) == [1, 2, 3]


def test_truncate_context_keeps_most_recent_tokens():
    tokens = list(range(10))
    assert truncate_context(tokens, max_tokens=4) == [6, 7, 8, 9]


def test_truncate_context_exact_limit_is_no_op():
    tokens = [1, 2, 3, 4]
    assert truncate_context(tokens, max_tokens=4) == [1, 2, 3, 4]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'context'`

- [ ] **Step 3: Implement `context.py`**

```python
def truncate_context(tokens: list[int], max_tokens: int) -> list[int]:
    if len(tokens) <= max_tokens:
        return tokens
    return tokens[-max_tokens:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_context.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add context.py tests/test_context.py
git commit -m "feat: add context truncation module"
```

---

### Task 4: Model loader module

**Files:**
- Create: `model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `llama_cpp.Llama` (from Task 1's venv), the model file at `models/Qwen3.5-0.8B-Q4_K_M.gguf` (from Task 1).
- Produces: `load_model(model_path: Path, n_ctx: int = 4096) -> Llama`, loaded with `logits_all=True` (required for per-token logit inspection) and `verbose=False`. Consumed by Task 5 (`repl.py`).

- [ ] **Step 1: Write failing tests**

Create `tests/test_model.py`:

```python
from pathlib import Path

import pytest

from model import load_model

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"


def test_load_model_missing_file_raises_clear_error(tmp_path):
    missing = tmp_path / "does-not-exist.gguf"
    with pytest.raises(FileNotFoundError, match="just setup"):
        load_model(missing)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not downloaded; run `just setup` first")
def test_load_model_real_file_loads_successfully():
    model = load_model(MODEL_PATH)
    assert model.n_vocab() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model'`

- [ ] **Step 3: Implement `model.py`**

```python
from pathlib import Path

from llama_cpp import Llama


def load_model(model_path: Path, n_ctx: int = 4096) -> Llama:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `just setup` to download it."
        )
    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        logits_all=True,
        verbose=False,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_model.py -v`
Expected: PASS (2 passed — the second test only runs if `just setup` has already downloaded the model)

- [ ] **Step 5: Commit**

```bash
git add model.py tests/test_model.py
git commit -m "feat: add model loader with clear missing-file error"
```

---

### Task 5: REPL orchestrator

**Files:**
- Create: `repl.py`

**Interfaces:**
- Consumes: `sample_token` from `sampling.py` (Task 2), `truncate_context` from `context.py` (Task 3), `load_model` from `model.py` (Task 4).
- Produces: the runnable Phase 1 deliverable — `just repl` launches an interactive session. Nothing downstream consumes this (final task).

- [ ] **Step 1: Implement `repl.py`**

```python
import time
from pathlib import Path

import numpy as np

from context import truncate_context
from model import load_model
from sampling import sample_token

MODEL_PATH = Path(__file__).resolve().parent / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"
MAX_CONTEXT_TOKENS = 4096
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.9


def print_candidates(model, candidates, chosen_id):
    print("  candidates:")
    for token_id, prob in candidates:
        text = model.detokenize([token_id], special=True).decode("utf-8", errors="replace")
        marker = " <-- chosen" if token_id == chosen_id else ""
        print(f"    {prob:6.3f}  {text!r}{marker}")


def run_turn(model, context_tokens, user_text, rng):
    prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens = model.tokenize(
        prompt.encode("utf-8"), add_bos=(len(context_tokens) == 0), special=True
    )
    context_tokens = truncate_context(context_tokens + prompt_tokens, MAX_CONTEXT_TOKENS)

    model.reset()
    model.eval(context_tokens)

    generated_tokens: list[int] = []
    eos_id = model.token_eos()
    interrupted = False

    try:
        for _ in range(MAX_NEW_TOKENS):
            logits = np.array(model.eval_logits[-1], dtype=np.float64)
            token_id, candidates = sample_token(logits, TEMPERATURE, TOP_P, rng)
            print_candidates(model, candidates, token_id)

            context_tokens.append(token_id)
            model.eval([token_id])

            if token_id == eos_id:
                break
            generated_tokens.append(token_id)
    except KeyboardInterrupt:
        interrupted = True

    if not context_tokens or context_tokens[-1] != eos_id:
        closer = model.tokenize(b"<|im_end|>\n", add_bos=False, special=True)
        context_tokens.extend(closer)
        model.eval(closer)

    response_text = model.detokenize(generated_tokens, special=False).decode(
        "utf-8", errors="ignore"
    )
    return context_tokens, response_text, interrupted


def main():
    model = load_model(MODEL_PATH, n_ctx=MAX_CONTEXT_TOKENS)
    rng = np.random.default_rng()
    context_tokens: list[int] = []

    print(f"Loaded {MODEL_PATH.name} (vocab={model.n_vocab()}, n_ctx={model.n_ctx()})")

    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            break

        if not user_text:
            continue

        start = time.perf_counter()
        context_tokens, response_text, interrupted = run_turn(
            model, context_tokens, user_text, rng
        )
        elapsed = time.perf_counter() - start

        if interrupted:
            print(f"\n[interrupted] Partial response: {response_text}\n")
        else:
            tokens_per_sec = len(response_text.split()) / elapsed if elapsed > 0 else 0.0
            print(f"\nAssistant: {response_text}")
            print(f"[{elapsed:.2f}s, ~{tokens_per_sec:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Manual verification — basic turn**

Run: `just repl`

Type: `What is the capital of France?`

Expected: tokenization/candidate output streams per generated token, followed by `Assistant: ...` containing a coherent answer mentioning Paris, then stats line, then a new `You:` prompt.

- [ ] **Step 3: Manual verification — multi-turn context**

At the next `You:` prompt, type: `What language do they speak there?`

Expected: the response is relevant to the previous answer (i.e., references the country/language from turn 1), confirming `context_tokens` correctly persists across turns.

- [ ] **Step 4: Manual verification — interrupt handling**

Start a new turn with a prompt likely to generate a long response (e.g. `Write a long story about a dragon.`), then press Ctrl+C mid-generation.

Expected: prints `[interrupted] Partial response: ...` with whatever text had been generated so far, does not crash, and returns to the `You:` prompt.

- [ ] **Step 5: Manual verification — missing model file**

Temporarily rename `models/Qwen3.5-0.8B-Q4_K_M.gguf`, run `just repl`, confirm the error message mentions `just setup` rather than a raw traceback, then rename the file back.

- [ ] **Step 6: Run full test suite**

Run: `just test`
Expected: all tests from Tasks 2-4 pass.

- [ ] **Step 7: Commit**

```bash
git add repl.py
git commit -m "feat: add REPL orchestrator tying together model, sampling, and context"
```

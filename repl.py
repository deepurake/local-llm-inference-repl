import time
from pathlib import Path

import numpy as np

from context import truncate_context
from model import load_model
from sampling import sample_token

MODEL_PATH = Path(__file__).resolve().parent / "models" / "Qwen3.5-0.8B-Q4_K_M.gguf"
MAX_CONTEXT_TOKENS = 4096
MAX_NEW_TOKENS = 256
CLOSER_TOKEN_MARGIN = 8  # headroom for the ChatML closer ("<|im_end|>\n"), which is ~2 tokens
TEMPERATURE = 0.8
TOP_P = 0.9


def print_candidates(model, candidates, chosen_id):
    print("  candidates:")
    for token_id, prob in candidates:
        text = model.detokenize([token_id], special=True).decode("utf-8", errors="replace")
        marker = " <-- chosen" if token_id == chosen_id else ""
        print(f"    {prob:6.3f}  {text!r}{marker}")


def print_prompt_tokens(model, prompt_tokens):
    print("  prompt tokens:")
    for tid in prompt_tokens:
        piece = model.detokenize([tid], special=True).decode("utf-8", errors="replace")
        print(f"    {tid}: {piece!r}")


def run_turn(model, context_tokens, user_text, rng):
    prompt = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    prompt_tokens = model.tokenize(
        prompt.encode("utf-8"), add_bos=(len(context_tokens) == 0), special=True
    )
    print_prompt_tokens(model, prompt_tokens)
    context_tokens = truncate_context(
        context_tokens + prompt_tokens,
        MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS - CLOSER_TOKEN_MARGIN,
    )

    model.reset()
    model.eval(context_tokens)

    generated_tokens: list[int] = []
    eos_id = model.token_eos()
    interrupted = False

    try:
        for _ in range(MAX_NEW_TOKENS):
            logits = np.array(model.scores[model.n_tokens - 1, :], dtype=np.float64)
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
    return context_tokens, response_text, interrupted, len(generated_tokens)


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
        context_tokens, response_text, interrupted, num_generated = run_turn(
            model, context_tokens, user_text, rng
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

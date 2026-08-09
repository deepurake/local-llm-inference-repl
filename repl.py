import time

import grpc
import inference_pb2
import inference_pb2_grpc
from llama_cpp import Llama

import config
from context import truncate_context


def load_tokenizer():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {config.MODEL_PATH}. Run `just setup` to download it."
        )
    return Llama(model_path=str(config.MODEL_PATH), vocab_only=True, verbose=False)


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
    eos_id = tokenizer.token_eos()
    interrupted = False
    call = stub.Generate(request)

    try:
        for event in call:
            print_candidates(tokenizer, event.candidates, event.token_id)
            context_tokens.append(event.token_id)
            if not event.is_eos:
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
                f"{config.GRPC_HOST}:{config.GRPC_PORT} — is `just serve` running? "
                f"({e.code()}: {e.details()})\n"
            )
        return original_context_tokens, "", False, 0

    if not context_tokens or context_tokens[-1] != eos_id:
        closer = tokenizer.tokenize(b"<|im_end|>\n", add_bos=False, special=True)
        context_tokens.extend(closer)

    response_text = tokenizer.detokenize(generated_tokens, special=False).decode(
        "utf-8", errors="ignore"
    )
    return context_tokens, response_text, interrupted, len(generated_tokens)


def main():
    tokenizer = load_tokenizer()

    with open(config.CERT_PATH, "rb") as f:
        trusted_certs = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
    channel = grpc.secure_channel(f"{config.GRPC_HOST}:{config.GRPC_PORT}", credentials)
    stub = inference_pb2_grpc.InferenceStub(channel)

    context_tokens: list[int] = []

    print(f"Connected to inference server at {config.GRPC_HOST}:{config.GRPC_PORT}")

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

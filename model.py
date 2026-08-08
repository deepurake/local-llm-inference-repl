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

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import config

_MODELS_DIR = config.MODEL_PATH.parent


@dataclass(frozen=True)
class ModelEntry:
    name: str
    filename: str
    wandb_artifact: str
    download_url: str | None


MODEL_REGISTRY: dict[str, ModelEntry] = {
    "qwen3.5-0.8b": ModelEntry(
        name="qwen3.5-0.8b",
        filename=config.MODEL_PATH.name,
        wandb_artifact="local-llm-inference-repl/model-registry/qwen3.5-0.8b:latest",
        download_url=None,
    ),
    "tinyllama-1.1b": ModelEntry(
        name="tinyllama-1.1b",
        filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        wandb_artifact="local-llm-inference-repl/model-registry/tinyllama-1.1b-chat:latest",
        download_url=(
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
            "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        ),
    ),
}

DEFAULT_MODEL_NAME = "qwen3.5-0.8b"


class ModelRegistryError(RuntimeError):
    pass


def resolve_model_path(name: str) -> Path:
    try:
        entry = MODEL_REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ModelRegistryError(f"Unknown model {name!r}. Known models: {known}") from None

    local_path = _MODELS_DIR / entry.filename
    if local_path.exists():
        return local_path

    if os.environ.get("WANDB_API_KEY"):
        try:
            return _download_from_wandb(entry, local_path)
        except Exception as e:
            print(f"[model_registry] wandb download failed ({e}); falling back to HTTP source")

    if entry.download_url:
        return _download_http(entry.download_url, local_path)

    raise ModelRegistryError(
        f"Model {name!r} is not cached at {local_path} and no download source "
        "is configured (no WANDB_API_KEY and no download_url)."
    )


def _download_from_wandb(entry: ModelEntry, dest: Path) -> Path:
    import wandb

    api = wandb.Api()
    artifact = api.artifact(entry.wandb_artifact)
    downloaded_dir = Path(artifact.download(root=str(_MODELS_DIR)))
    downloaded_path = downloaded_dir / entry.filename
    if not downloaded_path.exists():
        raise ModelRegistryError(
            f"wandb artifact {entry.wandb_artifact!r} did not contain {entry.filename!r}"
        )
    return downloaded_path


def _download_http(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp_path.rename(dest)
    return dest

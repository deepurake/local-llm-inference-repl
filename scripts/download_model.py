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

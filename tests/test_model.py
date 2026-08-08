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

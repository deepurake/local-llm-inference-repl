from pathlib import Path

import pytest

import model_registry
from model_registry import ModelEntry, ModelRegistryError, resolve_model_path


@pytest.fixture
def fake_registry(tmp_path, monkeypatch):
    """Point the registry at an isolated tmp_path models dir with one
    cached entry and one not-yet-downloaded entry, so tests never touch
    the real models/ directory or the network."""
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)

    cached_entry = ModelEntry(
        name="cached-model",
        filename="cached.gguf",
        wandb_artifact="entity/model-registry/cached:latest",
        download_url="https://example.invalid/cached.gguf",
    )
    (tmp_path / cached_entry.filename).write_bytes(b"already here")

    remote_entry = ModelEntry(
        name="remote-model",
        filename="remote.gguf",
        wandb_artifact="entity/model-registry/remote:latest",
        download_url="https://example.invalid/remote.gguf",
    )

    registry = {"cached-model": cached_entry, "remote-model": remote_entry}
    monkeypatch.setattr(model_registry, "MODEL_REGISTRY", registry)
    return tmp_path, registry


def test_unknown_model_name_raises_with_known_names_listed(fake_registry):
    with pytest.raises(ModelRegistryError, match="cached-model.*remote-model|remote-model.*cached-model"):
        resolve_model_path("does-not-exist")


def test_already_cached_model_short_circuits(fake_registry, monkeypatch):
    tmp_path, _ = fake_registry

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt any download for an already-cached model")

    monkeypatch.setattr(model_registry, "_download_from_wandb", fail_if_called)
    monkeypatch.setattr(model_registry, "_download_http", fail_if_called)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    result = resolve_model_path("cached-model")

    assert result == tmp_path / "cached.gguf"


def test_uses_wandb_when_api_key_present(fake_registry, monkeypatch):
    tmp_path, registry = fake_registry
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    calls = []

    def fake_download_from_wandb(entry, dest):
        calls.append(entry.name)
        dest.write_bytes(b"from wandb")
        return dest

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not fall back to HTTP when wandb succeeds")

    monkeypatch.setattr(model_registry, "_download_from_wandb", fake_download_from_wandb)
    monkeypatch.setattr(model_registry, "_download_http", fail_if_called)

    result = resolve_model_path("remote-model")

    assert calls == ["remote-model"]
    assert result.read_bytes() == b"from wandb"


def test_falls_back_to_http_when_wandb_fails(fake_registry, monkeypatch):
    tmp_path, registry = fake_registry
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    def broken_wandb_download(entry, dest):
        raise RuntimeError("simulated wandb auth failure")

    def fake_http_download(url, dest):
        dest.write_bytes(b"from http fallback")
        return dest

    monkeypatch.setattr(model_registry, "_download_from_wandb", broken_wandb_download)
    monkeypatch.setattr(model_registry, "_download_http", fake_http_download)

    result = resolve_model_path("remote-model")

    assert result.read_bytes() == b"from http fallback"


def test_falls_back_to_http_when_no_api_key_set(fake_registry, monkeypatch):
    tmp_path, registry = fake_registry
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt wandb without WANDB_API_KEY")

    def fake_http_download(url, dest):
        dest.write_bytes(b"from http")
        return dest

    monkeypatch.setattr(model_registry, "_download_from_wandb", fail_if_called)
    monkeypatch.setattr(model_registry, "_download_http", fake_http_download)

    result = resolve_model_path("remote-model")

    assert result.read_bytes() == b"from http"


def test_raises_clear_error_when_no_download_source_available(fake_registry, monkeypatch):
    tmp_path, registry = fake_registry
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    registry["remote-model"] = ModelEntry(
        name="remote-model",
        filename="remote.gguf",
        wandb_artifact="entity/model-registry/remote:latest",
        download_url=None,
    )

    with pytest.raises(ModelRegistryError, match="no download source"):
        resolve_model_path("remote-model")

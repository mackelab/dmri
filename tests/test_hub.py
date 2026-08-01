from dataclasses import FrozenInstanceError

import pytest

from dmri import (
    DEFAULT_MODEL_NAME,
    DEFAULT_REPO_ID,
    PretrainedModel,
    list_pretrained_models,
    load_pretrained,
)


class FakeModel:
    cfg = {"name": "model-config"}

    def __init__(self):
        self.evaluated = False

    def eval(self):
        self.evaluated = True


def test_load_pretrained_uses_defaults_and_ema(monkeypatch):
    calls = {}
    model = FakeModel()
    simulator = object()

    def fake_load_checkpoint(**kwargs):
        calls["load"] = kwargs
        return (
            {"params": "params", "params_ema": "ema", "step": 12},
            model,
            simulator,
        )

    monkeypatch.setattr("dmri.hub.load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        "dmri.hub.nnx.update",
        lambda updated_model, params: calls.update(update=(updated_model, params)),
    )

    result = load_pretrained()

    assert calls["load"] == {
        "repo_id": DEFAULT_REPO_ID,
        "model_name": DEFAULT_MODEL_NAME,
        "which": "best",
        "revision": None,
        "cache_dir": None,
        "token": None,
        "local_files_only": False,
    }
    assert calls["update"] == (model, "ema")
    assert result == PretrainedModel(model, simulator, model.cfg, 12)
    assert model.evaluated is True


def test_load_pretrained_forwards_options_and_falls_back_to_params(monkeypatch):
    calls = {}
    model = FakeModel()

    def fake_load_checkpoint(**kwargs):
        calls["load"] = kwargs
        return {"params": "params", "params_ema": None}, model, "simulator"

    monkeypatch.setattr("dmri.hub.load_checkpoint", fake_load_checkpoint)
    monkeypatch.setattr(
        "dmri.hub.nnx.update",
        lambda updated_model, params: calls.update(update=(updated_model, params)),
    )

    result = load_pretrained(
        "other-model",
        "owner/repo",
        "latest",
        revision="v1",
        cache_dir="cache",
        token="secret",
        local_files_only=True,
    )

    assert calls["load"] == {
        "repo_id": "owner/repo",
        "model_name": "other-model",
        "which": "latest",
        "revision": "v1",
        "cache_dir": "cache",
        "token": "secret",
        "local_files_only": True,
    }
    assert calls["update"] == (model, "params")
    assert result.step is None


def test_pretrained_model_is_frozen():
    result = PretrainedModel(None, None, None, None)

    with pytest.raises(FrozenInstanceError):
        result.step = 1


def test_list_pretrained_models(monkeypatch):
    calls = {}

    class FakeApi:
        def __init__(self, token):
            calls["token"] = token

        def list_repo_files(self, **kwargs):
            calls["list"] = kwargs
            return [
                "z-model/config.yaml",
                "README.md",
                "a-model/checkpoints/best/1/data",
                "a-model/config.yaml",
                "nested/model/config.yaml",
            ]

    monkeypatch.setattr("dmri.hub.HfApi", FakeApi)

    result = list_pretrained_models("owner/repo", revision="v2", token="secret")

    assert result == ["a-model", "z-model"]
    assert calls == {
        "token": "secret",
        "list": {
            "repo_id": "owner/repo",
            "repo_type": "model",
            "revision": "v2",
        },
    }

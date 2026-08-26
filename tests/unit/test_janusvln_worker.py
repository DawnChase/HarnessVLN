from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vln.janusvln.worker import (
    JanusVLNBackend,
    NativeJanusPolicy,
    sample_history_indices,
)


class Policy:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.loaded = None
        self.reset_count = 0
        self.history_lengths = []
        self.closed = False

    def load(self, upstream_root, checkpoint, options):
        self.loaded = (upstream_root, checkpoint, dict(options))

    def reset(self):
        self.reset_count += 1

    def predict(self, images, instruction):
        self.history_lengths.append((len(images), instruction))
        return next(self.actions)

    def close(self):
        self.closed = True


class Tools:
    def __init__(self):
        self.moves = []

    def observe(self):
        return {"channels": {"rgb": np.zeros((480, 640, 3), dtype=np.uint8)}}

    def move_discrete(self, action):
        self.moves.append(action)


def load(backend, *, max_steps=10, num_history=8):
    backend.load(
        {
            "upstream_root": "/upstream",
            "checkpoint": "/checkpoint",
            "options": {"max_steps": max_steps, "num_history": num_history},
        }
    )


def test_janus_backend_maps_actions_and_resets_each_job() -> None:
    policy = Policy(["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP", "STOP"])
    backend = JanusVLNBackend(policy)
    tools = Tools()
    load(backend)

    reason = backend.navigate("go", {}, tools, threading.Event())
    second = backend.navigate("again", {}, tools, threading.Event())
    backend.close()

    assert reason == "model emitted STOP"
    assert second == "model emitted STOP"
    assert tools.moves == ["forward", "turn_left", "turn_right"]
    assert policy.reset_count == 2
    assert policy.history_lengths == [(1, "go"), (2, "go"), (3, "go"), (4, "go"), (1, "again")]
    assert policy.closed


def test_janus_history_sampling_matches_official_linspace() -> None:
    assert sample_history_indices(9, 8) == tuple(range(9))
    assert sample_history_indices(10, 8) == (0, 1, 2, 3, 4, 5, 6, 7, 9)
    assert sample_history_indices(18, 8) == (0, 2, 4, 6, 8, 10, 12, 14, 17)


def test_janus_cancel_and_step_limit_do_not_emit_extra_moves() -> None:
    cancelled = threading.Event()
    cancelled.set()
    policy = Policy(["MOVE_FORWARD"])
    backend = JanusVLNBackend(policy)
    tools = Tools()
    load(backend, max_steps=1)

    assert backend.navigate("go", {}, tools, cancelled) == "cancelled"
    cancelled.clear()
    assert backend.navigate("go", {}, tools, cancelled) == "maximum step count reached: 1"
    assert tools.moves == ["forward"]


def test_janus_job_can_tighten_but_not_expand_step_limit() -> None:
    backend = JanusVLNBackend(Policy(["MOVE_FORWARD", "MOVE_FORWARD"]))
    load(backend, max_steps=2)

    assert backend.navigate("go", {"max_steps": 1}, Tools(), threading.Event()) == (
        "maximum step count reached: 1"
    )
    with pytest.raises(ValueError, match="exceeds worker limit"):
        backend.navigate("go", {"max_steps": 3}, Tools(), threading.Event())


def test_janus_cancel_after_inference_fences_motion() -> None:
    cancelled = threading.Event()

    class CancellingPolicy(Policy):
        def predict(self, images, instruction):
            action = super().predict(images, instruction)
            cancelled.set()
            return action

    backend = JanusVLNBackend(CancellingPolicy(["MOVE_FORWARD"]))
    tools = Tools()
    load(backend)

    assert backend.navigate("go", {}, tools, cancelled) == "cancelled"
    assert tools.moves == []


def test_native_janus_loader_uses_official_evaluation_options(
    tmp_path: Path, monkeypatch
) -> None:
    upstream = tmp_path / "JanusVLN"
    source_root = upstream / "src"
    model_source = source_root / "qwen_vl" / "model"
    model_source.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls = {}
    config = object()

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["config"] = (path, kwargs)
            return config

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["tokenizer"] = (path, kwargs)
            return "tokenizer"

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["processor"] = (path, kwargs)
            return "processor"

    class Model:
        def eval(self):
            calls["eval"] = True
            return self

    model = Model()

    class ModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model"] = (path, kwargs)
            return model

    modules = {
        "torch": SimpleNamespace(bfloat16="bf16"),
        "transformers": SimpleNamespace(
            AutoConfig=AutoConfig,
            AutoTokenizer=AutoTokenizer,
            AutoProcessor=AutoProcessor,
        ),
        "qwen_vl.model.modeling_qwen2_5_vl": SimpleNamespace(
            __file__=str(model_source / "modeling_qwen2_5_vl.py"),
            Qwen2_5_VLForConditionalGenerationForJanusVLN=ModelClass,
        ),
        "qwen_vl.model.vggt.utils.load_fn": SimpleNamespace(
            load_and_preprocess_images="preprocess"
        ),
        "qwen_vl_utils": SimpleNamespace(extract_vision_info="extract"),
        "PIL.Image": SimpleNamespace(),
    }
    monkeypatch.setattr(
        "vln.janusvln.worker.importlib.import_module", lambda name: modules[name]
    )

    policy = NativeJanusPolicy()
    policy.load(upstream, checkpoint, {"device": "cuda:2", "local_files_only": True})

    assert calls["config"] == (checkpoint, {"local_files_only": True})
    model_path, model_options = calls["model"]
    assert model_path == checkpoint
    assert model_options == {
        "config": config,
        "torch_dtype": "bf16",
        "device_map": {"": "cuda:2"},
        "attn_implementation": "flash_attention_2",
        "mode": "evaluation",
        "local_files_only": True,
    }
    assert calls["tokenizer"] == (
        checkpoint,
        {"padding_side": "left", "local_files_only": True},
    )
    assert calls["processor"][0] == checkpoint
    assert calls["processor"][1]["max_pixels"] == 1_605_632
    assert calls["processor"][1]["use_fast"] is False
    assert calls["eval"] is True
    assert policy.model is model
    assert policy.extract_vision_info == "extract"
    assert policy.load_and_preprocess_images == "preprocess"

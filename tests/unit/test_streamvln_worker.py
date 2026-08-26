from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vln.streamvln.worker import (
    NativeStreamFrame,
    NativeStreamPolicy,
    StreamVLNBackend,
    _intrinsic_matrix,
    _pose_matrix,
)


class Policy:
    def __init__(self, action_batches, *, num_frames=32, num_history=8):
        self.action_batches = iter(action_batches)
        self.num_frames = num_frames
        self.num_history = num_history
        self.reset_count = 0
        self.prepared = []
        self.inferences = []
        self.closed = False

    def load(self, upstream_root, checkpoint, options):
        return {
            "num_frames": self.num_frames,
            "num_history": self.num_history,
            "num_future_steps": 4,
            "width": 640,
            "height": 480,
        }

    def reset(self):
        self.reset_count += 1

    def prepare_frame(self, rgb, depth, pose, camera_intrinsics):
        frame = int(rgb[0, 0, 0])
        self.prepared.append((frame, pose, camera_intrinsics))
        return frame

    def infer(self, frames, instruction, step_id, time_ids):
        self.inferences.append((tuple(frames), instruction, step_id, tuple(time_ids)))
        return next(self.action_batches)

    def close(self):
        self.closed = True


class Tools:
    def __init__(self):
        self.index = 0
        self.moves = []

    def observe(self):
        value = observation(self.index)
        self.index += 1
        return value

    def move_discrete(self, action):
        self.moves.append(action)
        return {"action": action}


def observation(index=0):
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[0, 0, 0] = index
    return {
        "channels": {
            "rgb": rgb,
            "depth": np.zeros((480, 640, 1), dtype=np.float32),
            "pose": {
                "frame": "habitat_episode",
                "x": 1.0,
                "y": 2.0,
                "z": 0.0,
                "yaw": 0.5,
            },
            "camera_intrinsics": np.eye(4),
        }
    }


def load(backend, *, max_steps=20):
    backend.load(
        {
            "upstream_root": "/upstream",
            "checkpoint": "/checkpoint",
            "options": {"max_steps": max_steps},
        }
    )


def test_stream_action_queue_observes_each_step_without_reinferring() -> None:
    policy = Policy([[1, 2, 3], [0]])
    backend = StreamVLNBackend(policy)
    tools = Tools()
    load(backend)

    assert backend.navigate("go upstairs", {}, tools, threading.Event()) == (
        "model emitted STOP"
    )
    backend.close()

    assert tools.moves == ["forward", "turn_left", "turn_right"]
    assert [call[2] for call in policy.inferences] == [0, 3]
    assert [call[0] for call in policy.inferences] == [(0,), (3,)]
    assert len(policy.prepared) == 4
    assert policy.reset_count == 1
    assert policy.closed


def test_stream_window_reset_samples_slow_history_at_boundary() -> None:
    policy = Policy([[1], [1], [1], [1], [0]], num_frames=4, num_history=2)
    backend = StreamVLNBackend(policy)
    tools = Tools()
    load(backend)

    assert backend.navigate("go", {}, tools, threading.Event()) == "model emitted STOP"

    assert tools.moves == ["forward"] * 4
    assert policy.reset_count == 2
    assert policy.inferences[-1] == ((0, 2, 4), "go", 4, (4,))


def test_stream_cancel_after_inference_fences_motion() -> None:
    cancelled = threading.Event()

    class CancellingPolicy(Policy):
        def infer(self, frames, instruction, step_id, time_ids):
            actions = super().infer(frames, instruction, step_id, time_ids)
            cancelled.set()
            return actions

    backend = StreamVLNBackend(CancellingPolicy([[1]]))
    tools = Tools()
    load(backend)

    assert backend.navigate("go", {}, tools, cancelled) == "cancelled"
    assert tools.moves == []


def test_stream_empty_generation_stops_and_step_limit_fails() -> None:
    backend = StreamVLNBackend(Policy([[]]))
    load(backend, max_steps=1)
    assert backend.navigate("go", {}, Tools(), threading.Event()) == "model emitted STOP"

    backend = StreamVLNBackend(Policy([[1]]))
    load(backend, max_steps=1)
    with pytest.raises(RuntimeError, match="exceeded maximum step count: 1"):
        backend.navigate("go", {}, Tools(), threading.Event())


def test_stream_rejects_missing_or_malformed_sensor_channels() -> None:
    backend = StreamVLNBackend(Policy([[0]]))
    load(backend)
    value = observation()
    del value["channels"]["pose"]
    tools = Tools()
    tools.observe = lambda: value
    with pytest.raises(ValueError, match="missing channels.*pose"):
        backend.navigate("go", {}, tools, threading.Event())

    backend = StreamVLNBackend(Policy([[0]]))
    load(backend)
    value = observation()
    value["channels"]["depth"] = np.zeros((480, 640), dtype=np.float32)
    tools.observe = lambda: value
    with pytest.raises(ValueError, match="depth shape"):
        backend.navigate("go", {}, tools, threading.Event())


def test_stream_pose_and_intrinsic_contracts() -> None:
    matrix = _pose_matrix({"x": 1, "y": 2, "z": 0, "yaw": 0}, 1.25)
    assert matrix.shape == (4, 4)
    assert matrix[:3, 3].tolist() == [1.0, -2.0, 1.25]
    assert np.array_equal(_intrinsic_matrix(np.eye(4)), np.eye(4))
    with pytest.raises(ValueError, match="finite 4x4"):
        _intrinsic_matrix(np.eye(3))


def test_native_stream_loader_uses_separate_tokenizer_and_exact_model_options(
    tmp_path, monkeypatch
) -> None:
    upstream = tmp_path / "StreamVLN"
    source_root = upstream / "streamvln"
    (source_root / "model").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    calls = {}

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.setdefault(cls.__name__, []).append((path, kwargs))
            if cls is Tokenizer:
                return FakeTokenizer()
            return "config"

    class Tokenizer(Loader):
        pass

    class Config(Loader):
        pass

    class FakeTokenizer:
        def add_tokens(self, *args, **kwargs):
            return 0

        def convert_tokens_to_ids(self, token):
            return {"<image>": 10, "<memory>": 11}[token]

    class Model:
        def __init__(self):
            self.model = SimpleNamespace(num_history=None)
            self.reset_count = 0
            self.device = None

        def reset(self, count):
            self.reset_count = count

        def requires_grad_(self, value):
            calls["requires_grad"] = value
            return self

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def get_vision_tower(self):
            return SimpleNamespace(image_processor="processor")

    model = Model()

    class ModelClass:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["model"] = (path, kwargs)
            return model

    modules = {
        "torch": SimpleNamespace(bfloat16="bf16"),
        "transformers": SimpleNamespace(AutoTokenizer=Tokenizer, AutoConfig=Config),
        "model.stream_video_vln": SimpleNamespace(
            __file__=str(source_root / "model" / "stream_video_vln.py"),
            StreamVLNForCausalLM=ModelClass,
        ),
        "utils.utils": SimpleNamespace(
            __file__=str(source_root / "utils" / "utils.py"),
            IMAGE_TOKEN_INDEX=-200,
            MEMORY_TOKEN_INDEX=-300,
        ),
        "PIL.Image": SimpleNamespace(),
        "depth_camera_filtering": SimpleNamespace(filter_depth=lambda value, **_: value),
    }
    monkeypatch.setattr(
        "vln.streamvln.worker.importlib.import_module", lambda name: modules[name]
    )

    policy = NativeStreamPolicy()
    settings = policy.load(
        upstream,
        checkpoint,
        {
            "tokenizer_path": "/tokenizers/qwen",
            "device": "cuda:2",
            "num_history": 12,
        },
    )

    assert calls["Tokenizer"][0][0] == "/tokenizers/qwen"
    assert calls["Config"][0][0] == checkpoint
    model_path, model_kwargs = calls["model"]
    assert model_path == checkpoint
    assert model_kwargs["attn_implementation"] == "flash_attention_2"
    assert model_kwargs["torch_dtype"] == "bf16"
    assert model_kwargs["low_cpu_mem_usage"] is False
    assert model.model.num_history == 12
    assert model.reset_count == 1
    assert model.device == "cuda:2"
    assert calls["requires_grad"] is False
    assert settings["num_frames"] == 32


def test_native_stream_generate_uses_task_type_and_preserves_kv_cache() -> None:
    class Tensor:
        device = "cuda:0"

        def unsqueeze(self, dim):
            return self

        def to(self, *args, **kwargs):
            return self

    class Torch:
        bfloat16 = "bf16"

        @staticmethod
        def stack(values):
            assert values
            return Tensor()

    calls = {}

    class Model:
        def generate(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(sequences=Tensor(), past_key_values="next-kv")

    policy = NativeStreamPolicy()
    policy.torch = Torch()
    policy.model = Model()
    policy.tokenizer = SimpleNamespace(
        batch_decode=lambda *args, **kwargs: ["↑ ← → STOP"]
    )
    policy.output_ids = None
    policy.past_key_values = "prior-kv"
    policy._input_ids = lambda instruction, step_id: Tensor()
    frame = NativeStreamFrame(Tensor(), Tensor(), Tensor(), Tensor())

    assert policy.infer([frame], "go", 1, [0, 1]) == (1, 2, 3, 0)
    assert calls["env_id"] == 0
    assert calls["time_ids"] == [[0, 1]]
    assert calls["task_type"] == [0]
    assert "task_ids" not in calls
    assert calls["past_key_values"] == "prior-kv"
    assert policy.past_key_values == "next-kv"

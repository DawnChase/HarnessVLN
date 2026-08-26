from __future__ import annotations

import threading

import numpy as np

from vln.janusvln_worker import JanusVLNBackend, sample_history_indices


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

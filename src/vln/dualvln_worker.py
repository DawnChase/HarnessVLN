from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vln.worker import WorkerTools, run_worker


DEFAULT_MODEL_SETTINGS: dict[str, Any] = {
    "env_num": 1,
    "sim_num": 1,
    "camera_intrinsic": [
        [585.0, 0.0, 320.0],
        [0.0, 585.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    "width": 640,
    "height": 480,
    "hfov": 79,
    "resize_w": 384,
    "resize_h": 384,
    "max_new_tokens": 1024,
    "num_frames": 32,
    "num_history": 8,
    "num_future_steps": 4,
    "device": "cuda:0",
    "predict_step_nums": 32,
    "continuous_traj": True,
    "infer_mode": "partial_async",
    "vis_debug": False,
    "vis_debug_path": "./runs/dualvln_debug",
}

ACTION_MAP = {
    -1: "stand_still",
    1: "forward",
    2: "turn_left",
    3: "turn_right",
}


class DualPolicy(Protocol):
    def load(
        self, upstream_root: Path, checkpoint: Path, options: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def reset(self) -> None: ...

    def step(self, observations: Sequence[Mapping[str, Any]]) -> Any: ...

    def close(self) -> None: ...


class DualVLNBackend:
    model_name = "dualvln"

    def __init__(self, policy: DualPolicy | None = None) -> None:
        self.policy = policy or NativeDualPolicy()
        self.max_steps = 1000
        self.width = 640
        self.height = 480
        self._loaded = False
        self._job_lock = threading.Lock()

    def load(self, hello: Mapping[str, Any]) -> None:
        if self._loaded:
            raise RuntimeError("DualVLN backend is already loaded")
        options = hello.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError("DualVLN worker options must be an object")
        self.max_steps = _positive_int(options.get("max_steps", 1000), "max_steps")
        upstream_root = Path(str(hello["upstream_root"])).resolve()
        checkpoint = Path(str(hello["checkpoint"])).resolve()
        settings = self.policy.load(upstream_root, checkpoint, options)
        self.width = _positive_int(settings.get("width", 640), "model width")
        self.height = _positive_int(settings.get("height", 480), "model height")
        self._loaded = True

    def navigate(
        self,
        instruction: str,
        options: Mapping[str, Any],
        tools: WorkerTools,
        cancelled: threading.Event,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("DualVLN backend is not loaded")
        limit = _job_step_limit(options, self.max_steps)
        with self._job_lock:
            if cancelled.is_set():
                return "cancelled"
            self.policy.reset()
            for _ in range(limit):
                if cancelled.is_set():
                    return "cancelled"
                observation = tools.observe()
                rgb, depth = _require_rgbd(observation, self.height, self.width)
                if cancelled.is_set():
                    return "cancelled"
                output = self.policy.step(
                    [{"rgb": rgb, "depth": depth, "instruction": instruction}]
                )
                if cancelled.is_set():
                    return "cancelled"
                action = _require_action(output)
                if action == 0:
                    return "model emitted STOP"
                mapped = ACTION_MAP.get(action)
                if mapped is None:
                    raise ValueError(f"unsupported DualVLN action: {action}")
                if cancelled.is_set():
                    return "cancelled"
                tools.move_discrete(mapped)
                if cancelled.is_set():
                    return "cancelled"
            raise RuntimeError(f"DualVLN exceeded maximum step count: {limit}")

    def close(self) -> None:
        if self._loaded:
            self.policy.close()
            self._loaded = False


class NativeDualPolicy:
    def __init__(self) -> None:
        self.agent: Any = None

    def load(
        self, upstream_root: Path, checkpoint: Path, options: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if not (upstream_root / "internnav").is_dir():
            raise FileNotFoundError(
                f"InternNav source package not found under: {upstream_root}"
            )
        sys.path.insert(0, str(upstream_root))
        agent_module = importlib.import_module("internnav.agent")
        module_path = Path(str(agent_module.__file__)).resolve()
        if not module_path.is_relative_to(upstream_root):
            raise RuntimeError(
                f"InternNav agent resolved outside configured upstream: {module_path}"
            )
        config_module = importlib.import_module("internnav.configs.agent")
        model_config_module = importlib.import_module("internnav.configs.model")
        base_config = model_config_module.internvla_n1_cfg.model_dump()
        settings = build_model_settings(base_config, checkpoint, options)
        config = config_module.AgentCfg(
            model_name="internvla_n1",
            ckpt_path=str(checkpoint),
            model_settings=settings,
        )
        self.agent = agent_module.Agent.init(config)
        return settings

    def reset(self) -> None:
        self.agent.reset([0])

    def step(self, observations: Sequence[Mapping[str, Any]]) -> Any:
        return self.agent.step(list(observations))

    def close(self) -> None:
        # InternNav owns an infinite daemon S2 thread. Process teardown is its
        # lifecycle boundary; dropping the instance avoids racing a final reset.
        self.agent = None


def build_model_settings(
    base: Mapping[str, Any], checkpoint: Path, options: Mapping[str, Any]
) -> dict[str, Any]:
    overrides = options.get("model_settings", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("DualVLN model_settings must be an object")
    settings = {**dict(base), **DEFAULT_MODEL_SETTINGS, **dict(overrides)}
    if "device" in options:
        settings["device"] = str(options["device"])
    settings["model_path"] = str(checkpoint)
    settings["policy_name"] = "InternVLAN1_Policy"
    return settings


def _job_step_limit(options: Mapping[str, Any], maximum: int) -> int:
    limit = _positive_int(options.get("max_steps", maximum), "job max_steps")
    if limit > maximum:
        raise ValueError(f"job max_steps {limit} exceeds worker limit {maximum}")
    return limit


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"DualVLN {name} must be a positive integer")
    return value


def _require_rgbd(
    observation: Mapping[str, Any], height: int, width: int
) -> tuple[Any, Any]:
    channels = observation.get("channels")
    if not isinstance(channels, Mapping):
        raise ValueError("DualVLN observation channels must be an object")
    if "rgb" not in channels or "depth" not in channels:
        raise ValueError("DualVLN observation requires rgb and depth channels")
    rgb = channels["rgb"]
    depth = channels["depth"]
    if tuple(getattr(rgb, "shape", ())) != (height, width, 3):
        raise ValueError(
            f"DualVLN expects RGB shape {(height, width, 3)}, "
            f"got {getattr(rgb, 'shape', None)}"
        )
    if str(getattr(rgb, "dtype", "")) != "uint8":
        raise ValueError("DualVLN expects uint8 RGB")
    if tuple(getattr(depth, "shape", ())) != (height, width, 1):
        raise ValueError(
            f"DualVLN expects depth shape {(height, width, 1)}, "
            f"got {getattr(depth, 'shape', None)}"
        )
    if str(getattr(depth, "dtype", "")) != "float32":
        raise ValueError("DualVLN expects float32 depth")
    if not np.isfinite(depth).all():
        raise ValueError("DualVLN depth contains non-finite values")
    if depth.size and (float(depth.min()) < 0.0 or float(depth.max()) > 1.0):
        raise ValueError("DualVLN expects depth normalized to [0, 1]")
    return rgb, depth


def _require_action(output: Any) -> int:
    if not isinstance(output, list) or len(output) != 1:
        raise ValueError("DualVLN output must be a one-item list")
    item = output[0]
    if not isinstance(item, Mapping):
        raise ValueError("DualVLN output item must be an object")
    if item.get("ideal_flag") is not True:
        raise ValueError("DualVLN velocity output is not supported")
    action = item.get("action")
    if not isinstance(action, list) or len(action) != 1 or type(action[0]) is not int:
        raise ValueError("DualVLN output action must be a one-item integer list")
    return action[0]


if __name__ == "__main__":
    run_worker(DualVLNBackend())

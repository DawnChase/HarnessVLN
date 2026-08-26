from __future__ import annotations

import importlib
import re
import sys
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from vln.worker import WorkerTools, run_worker


ACTION_MAP = {1: "forward", 2: "turn_left", 3: "turn_right"}
ACTION_PATTERN = re.compile(r"STOP|↑|←|→")
ACTION_TOKEN_MAP = {"STOP": 0, "↑": 1, "←": 2, "→": 3}


class StreamPolicy(Protocol):
    def load(
        self, upstream_root: Path, checkpoint: Path, options: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def reset(self) -> None: ...

    def prepare_frame(
        self, rgb: Any, depth: Any, pose: Any, camera_intrinsics: Any
    ) -> Any: ...

    def infer(
        self,
        frames: Sequence[Any],
        instruction: str,
        step_id: int,
        time_ids: Sequence[int],
    ) -> Sequence[int]: ...

    def close(self) -> None: ...


class StreamVLNBackend:
    model_name = "streamvln"

    def __init__(self, policy: StreamPolicy | None = None) -> None:
        self.policy = policy or NativeStreamPolicy()
        self.num_frames = 32
        self.num_history: int | None = 8
        self.num_future_steps = 4
        self.max_steps = 500
        self.height = 480
        self.width = 640
        self._loaded = False
        self._job_lock = threading.Lock()

    def load(self, hello: Mapping[str, Any]) -> None:
        if self._loaded:
            raise RuntimeError("StreamVLN backend is already loaded")
        options = hello.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError("StreamVLN worker options must be an object")
        self.max_steps = _positive_int(options.get("max_steps", 500), "max_steps")
        upstream_root = Path(str(hello["upstream_root"])).resolve()
        checkpoint = Path(str(hello["checkpoint"])).resolve()
        settings = self.policy.load(upstream_root, checkpoint, options)
        self.num_frames = _positive_int(settings.get("num_frames", 32), "num_frames")
        history = settings.get("num_history", 8)
        self.num_history = (
            None if history is None else _positive_int(history, "num_history")
        )
        self.num_future_steps = _positive_int(
            settings.get("num_future_steps", 4), "num_future_steps"
        )
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
            raise RuntimeError("StreamVLN backend is not loaded")
        limit = _job_step_limit(options, self.max_steps)
        with self._job_lock:
            if cancelled.is_set():
                return "cancelled"
            self.policy.reset()
            frames: list[Any] = []
            time_ids: list[int] = []
            action_queue: deque[int] = deque()
            step_id = 0
            while step_id < limit:
                if cancelled.is_set():
                    return "cancelled"
                observation = tools.observe()
                rgb, depth, pose, intrinsics = _require_observation(
                    observation, self.height, self.width
                )
                frame = self.policy.prepare_frame(rgb, depth, pose, intrinsics)
                frames.append(frame)
                time_ids.append(step_id)
                if cancelled.is_set():
                    return "cancelled"
                if not action_queue:
                    selected = self._select_frames(frames, step_id, time_ids)
                    actions = self.policy.infer(
                        selected, instruction, step_id, tuple(time_ids)
                    )
                    action_queue.extend(_require_actions(actions))
                    if not action_queue:
                        action_queue.append(0)
                if cancelled.is_set():
                    return "cancelled"
                action = action_queue.popleft()
                if action == 0:
                    return "model emitted STOP"
                mapped = ACTION_MAP[action]
                tools.move_discrete(mapped)
                if cancelled.is_set():
                    return "cancelled"
                step_id += 1
                if step_id % self.num_frames == 0:
                    self.policy.reset()
                    time_ids.clear()
            raise RuntimeError(f"StreamVLN exceeded maximum step count: {limit}")

    def _select_frames(
        self, frames: Sequence[Any], step_id: int, time_ids: Sequence[int]
    ) -> tuple[Any, ...]:
        current = frames[-1]
        if step_id == 0 or step_id % self.num_frames != 0:
            return (current,)
        start = time_ids[0]
        stride = (
            self.num_future_steps
            if self.num_history is None
            else max(start // self.num_history, 1)
        )
        return (*frames[0:start:stride], current)

    def close(self) -> None:
        if self._loaded:
            self.policy.close()
            self._loaded = False


@dataclass(frozen=True, slots=True)
class NativeStreamFrame:
    image: Any
    depth: Any
    pose: Any
    intrinsics: Any


class NativeStreamPolicy:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.image_processor: Any = None
        self.torch: Any = None
        self.image_type: Any = None
        self.filter_depth: Any = None
        self.device = "cuda:0"
        self.max_new_tokens = 10000
        self.camera_height = 1.25
        self.output_ids: Any = None
        self.past_key_values: Any = None
        self.image_token_index: int | None = None
        self.memory_token_index: int | None = None
        self.image_placeholder = -200
        self.memory_placeholder = -300

    def load(
        self, upstream_root: Path, checkpoint: Path, options: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        source_root = (upstream_root / "streamvln").resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"StreamVLN source directory not found: {source_root}")
        tokenizer_path = options.get("tokenizer_path")
        if not isinstance(tokenizer_path, str) or not tokenizer_path:
            raise ValueError("StreamVLN worker requires a non-empty tokenizer_path")
        vision_tower_path = options.get("vision_tower_path")
        if not isinstance(vision_tower_path, str) or not vision_tower_path:
            raise ValueError("StreamVLN worker requires a non-empty vision_tower_path")
        local_only = bool(options.get("local_files_only", True))
        if local_only and not Path(tokenizer_path).is_dir():
            raise FileNotFoundError(
                f"StreamVLN tokenizer directory not found: {tokenizer_path}"
            )
        if local_only and not Path(vision_tower_path).is_dir():
            raise FileNotFoundError(
                f"StreamVLN vision tower directory not found: {vision_tower_path}"
            )
        sys.path.insert(0, str(upstream_root))
        sys.path.insert(0, str(source_root))
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        modeling = importlib.import_module("model.stream_video_vln")
        module_path = Path(str(modeling.__file__)).resolve()
        if not module_path.is_relative_to(source_root):
            raise RuntimeError(
                f"StreamVLN model resolved outside configured upstream: {module_path}"
            )
        constants = importlib.import_module("utils.utils")
        constants_path = Path(str(constants.__file__)).resolve()
        if not constants_path.is_relative_to(source_root):
            raise RuntimeError(
                f"StreamVLN utilities resolved outside configured upstream: {constants_path}"
            )
        pil_image = importlib.import_module("PIL.Image")
        filtering = importlib.import_module("depth_camera_filtering")
        model_max_length = _positive_int(
            options.get("model_max_length", 4096), "model_max_length"
        )
        num_history = options.get("num_history", 8)
        if num_history is not None:
            num_history = _positive_int(num_history, "num_history")
        self.device = str(options.get("device", "cuda:0"))
        self.max_new_tokens = _positive_int(
            options.get("max_new_tokens", 10000), "max_new_tokens"
        )
        self.camera_height = float(options.get("camera_height", 1.25))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            model_max_length=model_max_length,
            padding_side="right",
            local_files_only=local_only,
        )
        config = transformers.AutoConfig.from_pretrained(
            checkpoint, local_files_only=local_only
        )
        config.mm_vision_tower = vision_tower_path
        config.vision_tower = vision_tower_path
        self.model = modeling.StreamVLNForCausalLM.from_pretrained(
            checkpoint,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=False,
            local_files_only=local_only,
        )
        self.model.model.num_history = num_history
        self.model.reset(1)
        self.model.requires_grad_(False)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.tokenizer.add_tokens(["<memory>"], special_tokens=True)
        self.tokenizer.chat_template = (
            "{% for message in messages %}{{'<|im_start|>' + message['role'] + "
            "'\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        self.image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        self.memory_token_index = self.tokenizer.convert_tokens_to_ids("<memory>")
        self.image_placeholder = int(constants.IMAGE_TOKEN_INDEX)
        self.memory_placeholder = int(constants.MEMORY_TOKEN_INDEX)
        self.image_processor = self.model.get_vision_tower().image_processor
        self.torch = torch
        self.image_type = pil_image
        self.filter_depth = filtering.filter_depth
        return {
            "num_frames": _positive_int(options.get("num_frames", 32), "num_frames"),
            "num_history": num_history,
            "num_future_steps": _positive_int(
                options.get("num_future_steps", 4), "num_future_steps"
            ),
            "width": 640,
            "height": 480,
        }

    def reset(self) -> None:
        self.model.reset_for_env(0)
        self.output_ids = None
        self.past_key_values = None

    def prepare_frame(
        self, rgb: Any, depth: Any, pose: Any, camera_intrinsics: Any
    ) -> NativeStreamFrame:
        image = self.image_type.fromarray(rgb).convert("RGB")
        image_tensor = self.image_processor.preprocess(
            images=image, return_tensors="pt"
        )["pixel_values"][0]
        depth_2d = self.filter_depth(
            np.array(depth[:, :, 0], dtype=np.float32, copy=True), blur_type=None
        )
        depth_mm = np.clip(depth_2d * 10_000.0, 0, 65_535).astype(np.uint16)
        depth_image = self.image_type.fromarray(depth_mm, mode="I;16")
        crop = self.image_processor.crop_size
        target = (int(crop["width"]), int(crop["height"]))
        depth_image = depth_image.resize(target, self.image_type.Resampling.NEAREST)
        depth_tensor = self.torch.from_numpy(
            np.array(depth_image, dtype=np.float32, copy=True) / 1000.0
        ).float()
        pose_tensor = self.torch.from_numpy(
            _pose_matrix(pose, self.camera_height)
        )
        intrinsic = _intrinsic_matrix(camera_intrinsics)
        intrinsic[0] /= rgb.shape[1] / target[0]
        intrinsic[1] /= rgb.shape[0] / target[1]
        intrinsic[0, 2] -= (target[0] - target[1]) / 2
        intrinsic_tensor = self.torch.from_numpy(intrinsic).float()
        return NativeStreamFrame(
            image_tensor, depth_tensor, pose_tensor, intrinsic_tensor
        )

    def infer(
        self,
        frames: Sequence[NativeStreamFrame],
        instruction: str,
        step_id: int,
        time_ids: Sequence[int],
    ) -> Sequence[int]:
        input_ids = self._input_ids(instruction, step_id)
        if self.output_ids is not None:
            input_ids = self.torch.cat(
                [self.output_ids, input_ids.to(self.output_ids.device)], dim=1
            )
        input_dict = {
            "images": self.torch.stack([frame.image for frame in frames])
            .unsqueeze(0)
            .to(self.device, dtype=self.torch.bfloat16),
            "depths": self.torch.stack([frame.depth for frame in frames])
            .unsqueeze(0)
            .to(self.device, dtype=self.torch.bfloat16),
            "poses": self.torch.stack([frame.pose for frame in frames])
            .unsqueeze(0)
            .to(self.device, dtype=self.torch.bfloat16),
            "intrinsics": self.torch.stack([frame.intrinsics for frame in frames])
            .unsqueeze(0)
            .to(self.device, dtype=self.torch.bfloat16),
            "inputs": input_ids.to(self.device),
            "env_id": 0,
            "time_ids": [list(time_ids)],
            "task_type": [0],
        }
        outputs = self.model.generate(
            **input_dict,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            past_key_values=self.past_key_values,
        )
        self.output_ids = outputs.sequences
        self.past_key_values = outputs.past_key_values
        text = self.tokenizer.batch_decode(
            self.output_ids, skip_special_tokens=False
        )[0].strip()
        return tuple(ACTION_TOKEN_MAP[token] for token in ACTION_PATTERN.findall(text))

    def _input_ids(self, instruction: str, step_id: int) -> Any:
        if self.output_ids is None:
            prompt = (
                "<video>\nYou are an autonomous navigation assistant. Your task is to "
                "<instruction>. Devise an action sequence to follow the instruction using "
                "the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE "
                "FORWARD (↑) by 25 centimeters, or STOP."
            )
            if step_id != 0:
                prompt += " These are your historical observations <memory>."
            prompt = prompt.replace("<video>\n", "").replace(
                "<instruction>.", instruction
            )
            sources = [
                {"role": "user", "content": f"{prompt} you can see <image>."},
                {"role": "assistant", "content": ""},
            ]
            messages = [{"role": "system", "content": "You are a helpful assistant."}]
            messages.extend(sources)
        else:
            messages = [
                {"role": "user", "content": "you can see <image>."},
                {"role": "assistant", "content": ""},
            ]
        token_ids: list[int] = []
        for message in messages:
            token_ids.extend(self.tokenizer.apply_chat_template([message]))
        token_ids = [
            self.image_placeholder
            if token == self.image_token_index
            else self.memory_placeholder
            if token == self.memory_token_index
            else token
            for token in token_ids
        ]
        return self.torch.tensor([token_ids], dtype=self.torch.long)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self.output_ids = None
        self.past_key_values = None


def _require_observation(
    observation: Mapping[str, Any], height: int, width: int
) -> tuple[Any, Any, Any, Any]:
    channels = observation.get("channels")
    if not isinstance(channels, Mapping):
        raise ValueError("StreamVLN observation channels must be an object")
    required = ("rgb", "depth", "pose", "camera_intrinsics")
    missing = [name for name in required if name not in channels]
    if missing:
        raise ValueError(f"StreamVLN observation is missing channels: {missing}")
    rgb = channels["rgb"]
    depth = channels["depth"]
    if tuple(getattr(rgb, "shape", ())) != (height, width, 3):
        raise ValueError(f"StreamVLN expects RGB shape {(height, width, 3)}")
    if str(getattr(rgb, "dtype", "")) != "uint8":
        raise ValueError("StreamVLN expects uint8 RGB")
    if tuple(getattr(depth, "shape", ())) != (height, width, 1):
        raise ValueError(f"StreamVLN expects depth shape {(height, width, 1)}")
    if str(getattr(depth, "dtype", "")) != "float32":
        raise ValueError("StreamVLN expects float32 depth")
    if not np.isfinite(depth).all():
        raise ValueError("StreamVLN depth contains non-finite values")
    if depth.size and (float(depth.min()) < 0.0 or float(depth.max()) > 1.0):
        raise ValueError("StreamVLN expects depth normalized to [0, 1]")
    return rgb, depth, channels["pose"], channels["camera_intrinsics"]


def _require_actions(actions: Sequence[int]) -> tuple[int, ...]:
    if isinstance(actions, (str, bytes)):
        raise ValueError("StreamVLN actions must be an integer sequence")
    try:
        value = tuple(actions)
    except TypeError as error:
        raise ValueError("StreamVLN actions must be an integer sequence") from error
    if any(type(action) is not int or action not in (0, 1, 2, 3) for action in value):
        raise ValueError(f"StreamVLN emitted invalid action sequence: {value!r}")
    return value


def _pose_matrix(pose: Any, camera_height: float) -> np.ndarray:
    array = np.asarray(pose)
    if array.shape == (4, 4):
        return array.astype(np.float64, copy=True)
    if not isinstance(pose, Mapping):
        raise ValueError("StreamVLN pose must be a pose object or 4x4 matrix")
    try:
        x = float(pose["x"])
        y = float(pose["y"])
        z = float(pose.get("z", 0.0))
        yaw = float(pose["yaw"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("StreamVLN pose object requires x, y, z, and yaw") from error
    transform = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0, x],
            [np.sin(yaw), np.cos(yaw), 0.0, -y],
            [0.0, 0.0, 1.0, camera_height + z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    axis_align = np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return transform @ axis_align


def _intrinsic_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("StreamVLN camera_intrinsics must be a finite 4x4 matrix")
    return matrix.copy()


def _job_step_limit(options: Mapping[str, Any], maximum: int) -> int:
    limit = _positive_int(options.get("max_steps", maximum), "job max_steps")
    if limit > maximum:
        raise ValueError(f"job max_steps {limit} exceeds worker limit {maximum}")
    return limit


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"StreamVLN {name} must be a positive integer")
    return value


if __name__ == "__main__":
    run_worker(StreamVLNBackend())

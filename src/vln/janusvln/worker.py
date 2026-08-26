from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from vln.worker import WorkerTools, run_worker


ACTION_MAP = {
    "MOVE_FORWARD": "forward",
    "TURN_LEFT": "turn_left",
    "TURN_RIGHT": "turn_right",
}


class JanusPolicy(Protocol):
    def load(
        self, upstream_root: Path, checkpoint: Path, options: Mapping[str, Any]
    ) -> None: ...

    def reset(self) -> None: ...

    def predict(self, images: Sequence[Any], instruction: str) -> str: ...

    def close(self) -> None: ...


class JanusVLNBackend:
    model_name = "janusvln"

    def __init__(self, policy: JanusPolicy | None = None) -> None:
        self.policy = policy or NativeJanusPolicy()
        self.num_history = 8
        self.max_steps = 400
        self._loaded = False

    def load(self, hello: Mapping[str, Any]) -> None:
        if self._loaded:
            raise RuntimeError("JanusVLN backend is already loaded")
        options = hello.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError("JanusVLN worker options must be an object")
        self.num_history = _positive_int(options.get("num_history", 8), "num_history")
        self.max_steps = _positive_int(options.get("max_steps", 400), "max_steps")
        upstream_root = Path(str(hello["upstream_root"])).resolve()
        checkpoint = Path(str(hello["checkpoint"])).resolve()
        self.policy.load(upstream_root, checkpoint, options)
        self._loaded = True

    def navigate(
        self,
        instruction: str,
        options: Mapping[str, Any],
        tools: WorkerTools,
        cancelled: threading.Event,
    ) -> str:
        if not self._loaded:
            raise RuntimeError("JanusVLN backend is not loaded")
        limit = _job_step_limit(options, self.max_steps)
        self.policy.reset()
        history: list[Any] = []
        for _ in range(limit):
            if cancelled.is_set():
                return "cancelled"
            observation = tools.observe()
            rgb = _require_rgb(observation)
            history.append(rgb)
            selected = [history[index] for index in sample_history_indices(
                len(history), self.num_history
            )]
            action = self.policy.predict(selected, instruction).strip()
            if cancelled.is_set():
                return "cancelled"
            if action == "STOP":
                return "model emitted STOP"
            mapped = ACTION_MAP.get(action)
            if mapped is None:
                return f"unknown model action treated as STOP: {action!r}"
            tools.move_discrete(mapped)
        return f"maximum step count reached: {limit}"

    def close(self) -> None:
        if self._loaded:
            self.policy.close()
            self._loaded = False


class NativeJanusPolicy:
    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.torch: Any = None
        self.image_type: Any = None
        self.extract_vision_info: Any = None
        self.load_and_preprocess_images: Any = None

    def load(
        self, upstream_root: Path, checkpoint: Path, options: Mapping[str, Any]
    ) -> None:
        source_root = (upstream_root / "src").resolve()
        if not source_root.is_dir():
            raise FileNotFoundError(f"JanusVLN source directory not found: {source_root}")
        sys.path.insert(0, str(source_root))
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        modeling = importlib.import_module(
            "qwen_vl.model.modeling_qwen2_5_vl"
        )
        module_path = Path(str(modeling.__file__)).resolve()
        if not module_path.is_relative_to(source_root):
            raise RuntimeError(
                f"JanusVLN model resolved outside configured upstream: {module_path}"
            )
        load_module = importlib.import_module("qwen_vl.model.vggt.utils.load_fn")
        qwen_utils = importlib.import_module("qwen_vl_utils")
        pil_image = importlib.import_module("PIL.Image")
        device = str(options.get("device", "cuda:0"))
        local_only = bool(options.get("local_files_only", True))
        config = transformers.AutoConfig.from_pretrained(
            checkpoint, local_files_only=local_only
        )
        model_class = modeling.Qwen2_5_VLForConditionalGenerationForJanusVLN
        self.model = model_class.from_pretrained(
            checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="flash_attention_2",
            mode="evaluation",
            local_files_only=local_only,
        ).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint, padding_side="left", local_files_only=local_only
        )
        self.processor = transformers.AutoProcessor.from_pretrained(
            checkpoint,
            min_pixels=28 * 28,
            max_pixels=1_605_632,
            padding_side="left",
            use_fast=False,
            local_files_only=local_only,
        )
        self.torch = torch
        self.image_type = pil_image
        self.extract_vision_info = qwen_utils.extract_vision_info
        self.load_and_preprocess_images = load_module.load_and_preprocess_images

    def reset(self) -> None:
        self.model.past_key_values_vggt = None

    def predict(self, images: Sequence[Any], instruction: str) -> str:
        pil_images = [self.image_type.fromarray(image).convert("RGB") for image in images]
        context = (
            "These images are your historical observations and your current observation.\n"
            f" Your task is to {instruction} \n You should take one of the following actions:\n"
            " MOVE_FORWARD\n TURN_LEFT\n TURN_RIGHT\n STOP."
        )
        message = [
            {
                "role": "system",
                "content": (
                    "You are a visual language navigation model, and your should go to the "
                    "locations to complete the given task. Compare the observation and "
                    "instruction to infer your current progress, and then select the correct "
                    "direction from the candidates to go to the target location and finish the task."
                ),
            },
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in pil_images),
                    {"type": "text", "text": context},
                ],
            },
        ]
        messages = [message]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        image_inputs = []
        images_vggt = []
        for current_message in messages:
            vision_info = self.extract_vision_info(current_message)
            current_vggt = []
            for index, element in enumerate(vision_info):
                image = self.load_and_preprocess_images([element["image"]])[0]
                if index == len(vision_info) - 1:
                    current_vggt.append(image)
                _, height, width = image.shape
                width -= (width // patch_size) % merge_size * patch_size
                height -= (height // patch_size) % merge_size * patch_size
                image_inputs.append(image[:, :height, :width])
            images_vggt.append(self.torch.stack(current_vggt))
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )
        device = self.model.device
        inputs["images_vggt"] = [feature.to(device) for feature in images_vggt]
        inputs = inputs.to(device)
        generated = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            temperature=0,
            top_p=None,
            num_beams=1,
            max_new_tokens=24,
        )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def close(self) -> None:
        self.model = None


def sample_history_indices(length: int, num_history: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    history_len = length - 1
    if history_len <= num_history:
        return tuple(range(length))
    return tuple(int(index * history_len / num_history) for index in range(num_history + 1))


def _require_rgb(observation: Mapping[str, Any]) -> Any:
    channels = observation.get("channels")
    if not isinstance(channels, Mapping) or "rgb" not in channels:
        raise ValueError("JanusVLN observation has no rgb channel")
    rgb = channels["rgb"]
    shape = getattr(rgb, "shape", ())
    if tuple(shape) != (480, 640, 3):
        raise ValueError(f"JanusVLN expects RGB shape (480, 640, 3), got {shape}")
    if str(getattr(rgb, "dtype", "")) != "uint8":
        raise ValueError("JanusVLN expects uint8 RGB")
    return rgb


def _job_step_limit(options: Mapping[str, Any], maximum: int) -> int:
    limit = _positive_int(options.get("max_steps", maximum), "job max_steps")
    if limit > maximum:
        raise ValueError(f"job max_steps {limit} exceeds worker limit {maximum}")
    return limit


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"JanusVLN {name} must be a positive integer")
    return value


if __name__ == "__main__":
    run_worker(JanusVLNBackend())

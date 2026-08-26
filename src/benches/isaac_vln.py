from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from benches._io import load_json, require_fields
from benches.base import BenchmarkCase, MetricSet
from harness.errors import HarnessError
from harness.runtime import NavigationResult
from schemas import NavGoal, NavTask


class _IsaacVLNBenchmark:
    name: str
    validation_status = "data_contract"
    dataset_type: str

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "val_unseen",
        instruction_type: str = "formal",
        filter_same_trajectory: bool = False,
        filter_stairs: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.instruction_type = instruction_type
        self.filter_same_trajectory = filter_same_trajectory
        self.filter_stairs = filter_stairs

    @property
    def dataset_path(self) -> Path:
        path = self.root / self.split / f"{self.split}.json.gz"
        if not path.exists():
            raise HarnessError(f"{self.name} split file not found: {path}")
        return path

    def cases(self) -> Iterable[BenchmarkCase]:
        document = load_json(self.dataset_path)
        if not isinstance(document, dict) or not isinstance(document.get("episodes"), list):
            raise HarnessError(f"invalid {self.name} dataset: {self.dataset_path}")
        indexed = list(enumerate(document["episodes"]))
        indexed.sort(key=lambda item: (self._scene_id(item[1]), item[0]))
        seen_trajectories: set[str] = set()
        seen_cases: set[str] = set()
        for _, raw in indexed:
            require_fields(
                raw,
                {"trajectory_id", "episode_id", "instruction", "start_position", "start_rotation"},
                f"{self.name} episode",
            )
            trajectory_id = str(raw["trajectory_id"])
            if self.filter_same_trajectory and trajectory_id in seen_trajectories:
                continue
            seen_trajectories.add(trajectory_id)
            if self.filter_stairs and self._has_vertical_transition(raw):
                continue
            path_key = f"{trajectory_id}_{raw['episode_id']}"
            case_id = f"{self.name}:{self.split}:{path_key}"
            if case_id in seen_cases:
                raise HarnessError(f"duplicate {self.name} path key: {path_key}")
            seen_cases.add(case_id)
            instruction = self._instruction(raw)
            scene_id = self._scene_id(raw)
            goal = NavGoal(f"{case_id}:goal:0", instruction)
            truth = {
                key: raw[key]
                for key in ("reference_path", "goals", "info")
                if key in raw
            }
            yield BenchmarkCase(
                case_id,
                NavTask(
                    case_id,
                    goal,
                    scene_id=scene_id,
                    public={"split": self.split},
                ),
                {
                    "dataset_root": str(self.root),
                    "dataset_type": self.dataset_type,
                    "path_key": path_key,
                    "native_episode": raw,
                },
                truth,
            )

    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet:
        del case
        metrics = _metric_record(result.environment.get("native_metrics"))
        aliases = {
            "sr": "success",
            "spl": "spl",
            "ne": "NE",
            "os": "osr",
            "ndtw": "ndtw",
            "tl": "TL",
        }
        output: dict[str, float] = {}
        for public_name, native_name in aliases.items():
            if native_name in metrics:
                output[public_name] = float(metrics[native_name])
        if not output:
            raise HarnessError(f"{self.name} native metrics contain no score fields")
        return output

    def _instruction(self, raw: Mapping[str, Any]) -> str:
        instruction = raw["instruction"]
        if not isinstance(instruction, Mapping):
            raise HarnessError(f"{self.name} instruction must be an object")
        text = instruction.get("instruction_text")
        if isinstance(text, Mapping):
            text = text.get(self.instruction_type)
        if not isinstance(text, str) or not text.strip():
            raise HarnessError(
                f"{self.name} episode has no {self.instruction_type!r} instruction"
            )
        return text.strip()

    def _scene_id(self, raw: Mapping[str, Any]) -> str:
        scene = raw.get("scan") if self.dataset_type == "kujiale" else raw.get("scene_id")
        if not isinstance(scene, str) or not scene:
            raise HarnessError(f"{self.name} episode has no scene identifier")
        return scene

    @staticmethod
    def _has_vertical_transition(raw: Mapping[str, Any]) -> bool:
        path = raw.get("reference_path")
        if not isinstance(path, list):
            return False
        heights = [point[1] for point in path if isinstance(point, list) and len(point) >= 3]
        return any(abs(current - previous) > 0.3 for previous, current in zip(heights, heights[1:]))


class VLNPEBenchmark(_IsaacVLNBenchmark):
    name = "vln_pe"
    dataset_type = "mp3d"

    def __init__(self, root: str | Path, **kwargs: Any) -> None:
        kwargs.setdefault("filter_stairs", True)
        super().__init__(root, **kwargs)


class VLNVerseBenchmark(_IsaacVLNBenchmark):
    name = "vlnverse"
    dataset_type = "kujiale"


def _metric_record(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarnessError("native_metrics must be an object")
    if any(key in value for key in ("success", "spl", "NE", "osr")):
        return value
    for nested in value.values():
        if isinstance(nested, list) and nested and isinstance(nested[0], Mapping):
            return nested[0]
        if isinstance(nested, Mapping):
            return nested
    raise HarnessError("native_metrics contain no metric record")

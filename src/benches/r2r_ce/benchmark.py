from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benches._io import load_json, require_fields
from benches.base import BenchmarkCase, MetricSet, spl
from harness.errors import HarnessError
from harness.runtime import NavigationResult
from schemas import NavGoal, NavTask


class R2RCEBenchmark:
    name = "r2r_ce"
    validation_status = "data_contract"

    def __init__(self, root: str | Path, *, split: str = "val_unseen") -> None:
        self.root = Path(root)
        self.split = split

    @property
    def dataset_path(self) -> Path:
        directory = self.root / self.split
        compressed = directory / f"{self.split}.json.gz"
        plain = directory / f"{self.split}.json"
        if compressed.exists():
            return compressed
        if plain.exists():
            return plain
        raise HarnessError(f"R2R-CE split file not found under {directory}")

    def cases(self) -> Iterable[BenchmarkCase]:
        document = load_json(self.dataset_path)
        if not isinstance(document, dict) or not isinstance(document.get("episodes"), list):
            raise HarnessError(f"invalid R2R-CE document: {self.dataset_path}")
        for raw in document["episodes"]:
            require_fields(
                raw,
                {
                    "episode_id",
                    "scene_id",
                    "start_position",
                    "start_rotation",
                    "instruction",
                },
                "R2R-CE episode",
            )
            instruction = raw["instruction"].get("instruction_text")
            if not isinstance(instruction, str) or not instruction.strip():
                raise HarnessError(f"R2R-CE episode {raw['episode_id']} has no instruction")
            case_id = f"r2r_ce:{self.split}:{raw['episode_id']}"
            goal = NavGoal(f"{case_id}:goal:0", instruction.strip(), "language")
            setup: dict[str, Any] = {
                "scene_id": raw["scene_id"],
                "start_position": raw["start_position"],
                "start_rotation": raw["start_rotation"],
            }
            truth: dict[str, Any] = {}
            if raw.get("goals"):
                setup["goals"] = raw["goals"]
                truth["goals"] = raw["goals"]
            if "reference_path" in raw:
                truth["reference_path"] = raw["reference_path"]
            if "info" in raw and "geodesic_distance" in raw["info"]:
                truth["shortest_path_length"] = raw["info"]["geodesic_distance"]
            yield BenchmarkCase(
                case_id,
                NavTask(case_id, goal, scene_id=raw["scene_id"], public={"split": self.split}),
                setup,
                truth,
            )

    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet:
        environment = result.environment
        success = bool(environment.get("success", False))
        if "spl" in environment:
            spl_value = float(environment["spl"])
        elif "path_length" in environment:
            spl_value = spl(
                success,
                float(case.truth["shortest_path_length"]),
                float(environment["path_length"]),
            )
        else:
            raise HarnessError("R2R-CE result has neither native SPL nor path length")
        if "distance_to_goal" not in environment:
            raise HarnessError("R2R-CE result has no distance to goal")
        return {
            "sr": float(success),
            "spl": spl_value,
            "ne": float(environment["distance_to_goal"]),
            "os": float(bool(environment.get("oracle_success", success))),
        }

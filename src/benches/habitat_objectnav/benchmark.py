from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benches._io import load_json, require_fields
from benches.base import BenchmarkCase, MetricSet, spl
from harness.errors import HarnessError
from harness.runtime import NavigationResult
from schemas import NavGoal, NavTask


class HabitatObjectNavBenchmark:
    validation_status = "data_contract"

    def __init__(
        self,
        root: str | Path,
        *,
        dataset: str,
        split: str = "val",
    ) -> None:
        if dataset not in {"mp3d", "hm3d"}:
            raise ValueError("dataset must be mp3d or hm3d")
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.name = f"habitat_objectnav_{dataset}"

    def cases(self) -> Iterable[BenchmarkCase]:
        directory = self.root / self.split / "content"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise HarnessError(f"ObjectNav content shards not found under {directory}")
        seen: set[str] = set()
        for path in paths:
            document = load_json(path)
            if not isinstance(document, dict):
                raise HarnessError(f"invalid ObjectNav shard: {path}")
            episodes = document.get("episodes")
            goals = document.get("goals_by_category")
            if not isinstance(episodes, list) or not isinstance(goals, dict):
                raise HarnessError(f"invalid ObjectNav shard structure: {path}")
            scene_key = path.name.removesuffix(".json.gz")
            for native_index, raw in enumerate(episodes):
                require_fields(
                    raw,
                    {
                        "episode_id",
                        "scene_id",
                        "start_position",
                        "start_rotation",
                        "object_category",
                        "info",
                    },
                    "Habitat ObjectNav episode",
                )
                category = raw["object_category"]
                if not isinstance(category, str) or not category:
                    raise HarnessError(f"ObjectNav episode in {path} has no category")
                goal_key = f"{Path(raw['scene_id']).name}_{category}"
                if not isinstance(goals.get(goal_key), list) or not goals[goal_key]:
                    raise HarnessError(f"ObjectNav goal table has no key {goal_key}")
                case_id = (
                    f"habitat_objectnav:{self.dataset}:{self.split}:"
                    f"{scene_key}:{native_index}"
                )
                if case_id in seen:
                    raise HarnessError(f"duplicate ObjectNav case id: {case_id}")
                seen.add(case_id)
                goal = NavGoal(
                    f"{case_id}:goal:0",
                    f"Find the {category}.",
                    "object",
                    {"category": category},
                )
                info = raw["info"]
                if not isinstance(info, dict) or "geodesic_distance" not in info:
                    raise HarnessError(f"ObjectNav episode {case_id} has no distance")
                setup: dict[str, Any] = {
                    "native_episode_index": native_index,
                    "source_episode_id": raw["episode_id"],
                    "scene_id": raw["scene_id"],
                    "start_position": raw["start_position"],
                    "start_rotation": raw["start_rotation"],
                    "object_category": category,
                }
                for name in ("scene_dataset_config", "additional_obj_config_paths"):
                    if name in raw:
                        setup[name] = raw[name]
                yield BenchmarkCase(
                    case_id,
                    NavTask(
                        case_id,
                        goal,
                        scene_id=raw["scene_id"],
                        public={"split": self.split, "dataset": self.dataset},
                    ),
                    setup,
                    {"shortest_path_length": float(info["geodesic_distance"])},
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
            raise HarnessError("ObjectNav result has neither native SPL nor path length")
        if "distance_to_goal" not in environment:
            raise HarnessError("ObjectNav result has no distance to goal")
        return {
            "sr": float(success),
            "spl": spl_value,
            "ne": float(environment["distance_to_goal"]),
        }

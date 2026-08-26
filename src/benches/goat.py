from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from benches._io import load_json, require_fields
from benches.base import BenchmarkCase, MetricSet
from harness.errors import HarnessError
from harness.runtime import NavigationResult
from schemas import NavGoal, NavTask


class GOATBenchmark:
    name = "goat_bench"
    validation_status = "data_contract"

    def __init__(self, root: str | Path, *, split: str = "val_unseen") -> None:
        self.root = Path(root)
        self.split = split

    def cases(self) -> Iterable[BenchmarkCase]:
        directory = self.root / self.split / "content"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise HarnessError(f"GOAT content shards not found under {directory}")
        seen: set[str] = set()
        for path in paths:
            document = load_json(path)
            if not isinstance(document, dict):
                raise HarnessError(f"invalid GOAT shard: {path}")
            episodes = document.get("episodes")
            goal_table = document.get("goals")
            if not isinstance(episodes, list) or not isinstance(goal_table, dict):
                raise HarnessError(f"invalid GOAT shard structure: {path}")
            scene_key = path.name.removesuffix(".json.gz")
            for raw in episodes:
                require_fields(
                    raw,
                    {"episode_id", "scene_id", "start_position", "start_rotation", "tasks"},
                    "GOAT episode",
                )
                case_id = f"goat:{self.split}:{scene_key}:{raw['episode_id']}"
                if case_id in seen:
                    raise HarnessError(f"duplicate GOAT episode id: {case_id}")
                seen.add(case_id)
                goals, goal_truth, goal_specs = self._resolve_goals(
                    case_id, raw["scene_id"], raw["tasks"], goal_table
                )
                setup = {
                    "scene_id": raw["scene_id"],
                    "scene_dataset_config": raw.get("scene_dataset_config"),
                    "start_position": raw["start_position"],
                    "start_rotation": raw["start_rotation"],
                    "goal_stream": goals,
                    "goal_specs": goal_specs,
                    "native_tasks": raw["tasks"],
                }
                yield BenchmarkCase(
                    case_id,
                    NavTask(
                        case_id,
                        goals[0],
                        scene_id=raw["scene_id"],
                        public={"split": self.split, "goal_count": len(goals)},
                    ),
                    setup,
                    {"goals": goal_truth},
                )

    def _resolve_goals(
        self,
        case_id: str,
        scene_id: str,
        tasks: list[list[Any]],
        table: dict[str, list[dict[str, Any]]],
    ) -> tuple[
        tuple[NavGoal, ...],
        tuple[dict[str, Any], ...],
        tuple[dict[str, Any], ...],
    ]:
        if not tasks:
            raise HarnessError(f"GOAT episode {case_id} has no goals")
        scene_filename = Path(scene_id).name
        public_goals: list[NavGoal] = []
        truth: list[dict[str, Any]] = []
        specs: list[dict[str, Any]] = []
        for index, native in enumerate(tasks):
            if len(native) < 3:
                raise HarnessError(f"invalid GOAT goal in {case_id}: {native!r}")
            category, modality, object_id = native[:3]
            key = f"{scene_filename}_{category}"
            instances = table.get(key)
            if not instances:
                raise HarnessError(f"GOAT goal table has no key {key}")
            selected = None
            image_index = None
            if modality in {"description", "image"}:
                selected = next(
                    (item for item in instances if item.get("object_id") == object_id), None
                )
                if selected is None:
                    raise HarnessError(f"GOAT object {object_id} is missing from {key}")
            if modality == "object":
                instruction = f"Find the {category}."
            elif modality == "description":
                assert selected is not None
                instruction = str(selected.get("lang_desc", "")).strip()
                if not instruction:
                    raise HarnessError(f"GOAT object {object_id} has no description")
            elif modality == "image":
                assert selected is not None
                if len(native) != 4 or not isinstance(native[3], int):
                    raise HarnessError(f"invalid GOAT image goal in {case_id}: {native!r}")
                image_index = native[3]
                images = selected.get("image_goals", [])
                if image_index < 0 or image_index >= len(images):
                    raise HarnessError(f"GOAT image index is out of range in {case_id}")
                instruction = "Navigate to the object shown in the goal image."
            else:
                raise HarnessError(f"unknown GOAT modality {modality!r}")

            goal_id = f"{case_id}:goal:{index}"
            public = {"goal_type": modality}
            if modality == "image":
                public["observation_channel"] = "cache_instance_imagegoal"
            public_goals.append(NavGoal(goal_id, instruction, modality, public))
            target_keys = [key]
            if modality == "object":
                children = {
                    child
                    for item in instances
                    for child in item.get("children_object_categories", [])
                }
                target_keys.extend(
                    child_key
                    for child in sorted(children)
                    if (child_key := f"{scene_filename}_{child}") in table
                )
            truth.append(
                {
                    "goal_id": goal_id,
                    "modality": modality,
                    "category": category,
                    "target_keys": target_keys,
                    "object_id": object_id,
                    "image_index": image_index,
                }
            )
            spec: dict[str, Any] = {
                "modality": modality,
                "category": category,
                "object_id": object_id,
            }
            if modality == "image":
                assert selected is not None and image_index is not None
                spec["image_goal"] = dict(selected["image_goals"][image_index])
            specs.append(spec)
        return tuple(public_goals), tuple(truth), tuple(specs)

    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet:
        del case
        goals = result.environment.get("goal_results")
        if not isinstance(goals, list) or not goals:
            raise HarnessError("GOAT environment result has no per-goal metrics")
        metrics: dict[str, float] = {
            "success": sum(float(bool(goal["success"])) for goal in goals) / len(goals),
            "spl": sum(float(goal["spl"]) for goal in goals) / len(goals),
        }
        for modality in ("object", "description", "image"):
            selected = [goal for goal in goals if goal["modality"] == modality]
            if selected:
                metrics[f"success_{modality}"] = sum(
                    float(bool(goal["success"])) for goal in selected
                ) / len(selected)
                metrics[f"spl_{modality}"] = sum(
                    float(goal["spl"]) for goal in selected
                ) / len(selected)
        return metrics

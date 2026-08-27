from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from benches._io import load_json, require_fields
from benches.base import BenchmarkCase, MetricSet, spl
from harness.errors import HarnessError
from harness.runtime import NavigationResult
from schemas import NavGoal, NavTask


class RoboTHORObjectNavBenchmark:
    name = "robothor_objectnav_2021"
    validation_status = "data_contract"

    def __init__(self, root: str | Path, *, split: str = "val") -> None:
        self.root = Path(root)
        self.split = split

    def cases(self) -> Iterable[BenchmarkCase]:
        directory = self.root / self.split / "episodes"
        paths = sorted(directory.glob("*.json.gz"))
        if not paths:
            raise HarnessError(f"RoboTHOR episode files not found under {directory}")
        seen: set[str] = set()
        for path in paths:
            episodes = load_json(path)
            if not isinstance(episodes, list):
                raise HarnessError(f"RoboTHOR file must contain a list: {path}")
            for raw in episodes:
                require_fields(
                    raw,
                    {
                        "id",
                        "scene",
                        "object_type",
                        "initial_position",
                        "initial_orientation",
                        "initial_horizon",
                    },
                    "RoboTHOR episode",
                )
                if raw["scene"] != path.name.removesuffix(".json.gz"):
                    raise HarnessError(f"RoboTHOR scene/file mismatch in {path}")
                case_id = f"robothor:{self.split}:{raw['id']}"
                if case_id in seen:
                    raise HarnessError(f"duplicate RoboTHOR episode id: {case_id}")
                seen.add(case_id)
                category = raw["object_type"]
                goal = NavGoal(
                    f"{case_id}:goal:0",
                    f"Find the {category}.",
                    "object",
                    {"category": category},
                )
                setup = {
                    "episode_id": raw["id"],
                    "scene": raw["scene"],
                    "initial_position": raw["initial_position"],
                    "initial_orientation": raw["initial_orientation"],
                    "initial_horizon": raw["initial_horizon"],
                    "object_type": category,
                }
                truth = {
                    key: raw[key]
                    for key in ("shortest_path", "shortest_path_length")
                    if key in raw
                }
                yield BenchmarkCase(
                    case_id,
                    NavTask(case_id, goal, scene_id=raw["scene"], public={"split": self.split}),
                    setup,
                    truth,
                )

    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet:
        if "shortest_path_length" not in case.truth:
            raise HarnessError(f"split {self.split} has no scoring ground truth")
        success = bool(result.environment.get("success", False))
        shortest = float(case.truth["shortest_path_length"])
        actual = float(result.environment.get("path_length", 0.0))
        return {"sr": float(success), "spl": spl(success, shortest, actual)}

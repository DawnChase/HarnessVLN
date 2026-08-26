from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from benches import (
    GOATBenchmark,
    HabitatObjectNavBenchmark,
    R2RCEBenchmark,
    RoboTHORObjectNavBenchmark,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "datasets"


def require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"local dataset is not available: {path}")
    return path


def test_r2r_ce_val_unseen_real_data_contract() -> None:
    benchmark = R2RCEBenchmark(require(DATA / "r2r"), split="val_unseen")
    cases = list(benchmark.cases())
    assert len(cases) == 1839
    assert len({case.case_id for case in cases}) == 1839
    assert len({case.task.scene_id for case in cases}) == 11
    assert all(case.task.goal.modality == "language" for case in cases)
    assert all("start_position" not in case.task.public for case in cases)
    assert all("goals" not in case.task.public for case in cases)
    assert all(case.truth["shortest_path_length"] > 0 for case in cases)


def test_goat_val_unseen_real_shards_and_private_future_goals() -> None:
    root = require(DATA / "goat_bench" / "hm3d" / "v1")
    benchmark = GOATBenchmark(root, split="val_unseen")
    cases = list(benchmark.cases())
    goals = [goal for case in cases for goal in case.setup["goal_stream"]]
    modalities = Counter(goal.modality for goal in goals)

    assert len(cases) == 360
    assert len({case.case_id for case in cases}) == 360
    assert len(goals) == 2669
    assert modalities == {"object": 991, "image": 822, "description": 856}
    assert all(5 <= len(case.setup["goal_stream"]) <= 10 for case in cases)
    assert all(case.task.goal == case.setup["goal_stream"][0] for case in cases)
    assert all("goal_stream" not in case.task.public for case in cases)
    assert all("goal_specs" not in case.task.public for case in cases)
    assert all("native_tasks" not in case.task.public for case in cases)
    assert all("object_id" not in case.task.goal.public for case in cases)
    image_goals = [goal for goal in goals if goal.modality == "image"]
    assert all(
        goal.public["observation_channel"] == "cache_instance_imagegoal"
        for goal in image_goals
    )
    image_specs = [
        spec
        for case in cases
        for spec in case.setup["goal_specs"]
        if spec["modality"] == "image"
    ]
    assert len(image_specs) == 822
    assert all("position" in spec["image_goal"] for spec in image_specs)
    assert all("rotation" in spec["image_goal"] for spec in image_specs)


def test_robothor_val_real_data_contract() -> None:
    benchmark = RoboTHORObjectNavBenchmark(
        require(DATA / "robothor-objectnav-2021"), split="val"
    )
    cases = list(benchmark.cases())
    categories = {case.task.goal.public["category"] for case in cases}

    assert len(cases) == 1800
    assert len({case.case_id for case in cases}) == 1800
    assert len(categories) == 12
    assert all(case.task.goal.instruction.startswith("Find the ") for case in cases)
    assert all(
        case.case_id == f"robothor:val:{case.setup['episode_id']}" for case in cases
    )
    assert all("initial_position" not in case.task.public for case in cases)
    assert all(case.truth["shortest_path_length"] >= 0 for case in cases)


@pytest.mark.parametrize(
    ("dataset", "version", "count", "scene_count"),
    (("mp3d", "v1", 2195, 11), ("hm3d", "v2", 1000, 36)),
)
def test_habitat_objectnav_real_shards_use_native_episode_indices(
    dataset: str, version: str, count: int, scene_count: int
) -> None:
    benchmark = HabitatObjectNavBenchmark(
        require(DATA / "objectnav" / dataset / version),
        dataset=dataset,
        split="val",
    )
    cases = list(benchmark.cases())

    assert len(cases) == count
    assert len({case.case_id for case in cases}) == count
    assert len({case.task.scene_id for case in cases}) == scene_count
    assert all(case.task.goal.modality == "object" for case in cases)
    assert all(case.task.instruction.startswith("Find the ") for case in cases)
    assert all("start_position" not in case.task.public for case in cases)
    assert all(case.truth["shortest_path_length"] >= 0 for case in cases)
    assert all(
        case.case_id.endswith(f":{case.setup['native_episode_index']}")
        for case in cases
    )

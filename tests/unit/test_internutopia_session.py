from __future__ import annotations

import math

import numpy as np
import pytest

from benches.base import BenchmarkCase
from envs.internutopia import prepare_native_episode, resource_status
from harness.errors import HarnessError
from schemas import NavGoal, NavTask


def isaac_case(dataset_type: str, native_episode: dict) -> BenchmarkCase:
    path_key = f"{native_episode['trajectory_id']}_{native_episode['episode_id']}"
    task = NavTask(
        f"{dataset_type}:{path_key}",
        NavGoal("goal", "go"),
        public={"split": "val_unseen"},
    )
    return BenchmarkCase(
        task.task_id,
        task,
        {
            "dataset_root": "/fixture/data",
            "dataset_type": dataset_type,
            "path_key": path_key,
            "native_episode": native_episode,
        },
    )


def test_mp3d_episode_uses_vln_pe_coordinates_and_robot_offset() -> None:
    fixture = isaac_case(
        "mp3d",
        {
            "scene_id": "mp3d/scan-a/scan-a.glb",
            "trajectory_id": 10,
            "episode_id": 4,
            "start_position": [1.0, 2.0, 3.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "reference_path": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "instruction": {"instruction_text": "go"},
        },
    )

    episode, scan = prepare_native_episode(
        fixture.environment_episode,
        dataset_type="mp3d",
        robot_offset=np.array([0.0, 0.0, 1.05]),
    )

    assert scan == "scan-a"
    assert episode["original_start_position"] == [1.0, 2.0, 3.0]
    assert episode["start_position"] == pytest.approx([1.0, -3.0, 3.05])
    assert episode["reference_path"][0] == pytest.approx([1.0, -3.0, 3.05])
    assert episode["reference_path"][1] == pytest.approx([4.0, -6.0, 6.05])
    root = math.sqrt(0.5)
    assert episode["start_rotation"] == pytest.approx([-root, 0.0, 0.0, -root])
    assert episode["instruction"]["instruction_tokens"] == []


def test_vlnverse_episode_keeps_kujiale_coordinates() -> None:
    fixture = isaac_case(
        "kujiale",
        {
            "scan": "room-a",
            "trajectory_id": "route",
            "episode_id": 9,
            "start_position": [1.0, 2.0, 3.0],
            "start_rotation": [1.0, 0.0, 0.0, 0.0],
            "reference_path": [[1.0, 2.0, 3.0]],
            "instruction": {"instruction_text": {"formal": "go"}},
        },
    )

    episode, scan = prepare_native_episode(
        fixture.environment_episode,
        dataset_type="kujiale",
        robot_offset=[0.0, 0.0, 1.05],
        reviser=lambda value: value,
    )

    assert scan == "room-a"
    assert episode["start_position"] == pytest.approx([1.0, 2.0, 4.05])
    assert episode["start_rotation"] == [1.0, 0.0, 0.0, 0.0]
    assert episode["reference_path"][0] == pytest.approx([1.0, 2.0, 4.05])


def test_native_episode_rejects_path_key_drift() -> None:
    fixture = isaac_case(
        "kujiale",
        {
            "scan": "room-a",
            "trajectory_id": "route",
            "episode_id": 9,
            "start_position": [0, 0, 0],
            "start_rotation": [1, 0, 0, 0],
        },
    )
    fixture = BenchmarkCase(
        fixture.case_id,
        fixture.task,
        {**fixture.env_setup, "path_key": "wrong"},
    )

    with pytest.raises(HarnessError, match="path_key mismatch"):
        prepare_native_episode(
            fixture.environment_episode,
            dataset_type="kujiale",
            robot_offset=[0, 0, 0],
        )


def test_resource_status_reports_missing_episode_assets_without_importing_isaac(
    tmp_path,
) -> None:
    source = tmp_path / "project"
    source.mkdir()
    config = source / "eval.py"
    config.write_text("eval_cfg = None\n")

    status = resource_status(
        scene_data_dir=tmp_path / "missing-scenes",
        robot_usd_path=tmp_path / "missing-robot.usd",
        eval_config="eval.py",
        internutopia_root=tmp_path / "missing-internutopia",
        project_root=source,
    )

    assert status == {
        "internutopia_root": False,
        "project_root": True,
        "eval_config": True,
        "scene_data_dir": False,
        "robot_usd_path": False,
        "locomotion_policy": False,
    }

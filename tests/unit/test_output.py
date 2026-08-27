from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from harness.errors import HarnessError
from harness.output import RunOutput
from schemas import NavGoal, NavTask


def read_json(path):
    return json.loads(path.read_text())


def test_run_output_writes_scoped_records_and_a_real_video(tmp_path) -> None:
    output = RunOutput(
        {
            "root": str(tmp_path),
            "run_id": "fixture-run",
            "video": {"enabled": True, "fps": 5},
        },
        resolved_config={"runner": {"seed": 7}, "agent": {"factory": "fixture"}},
        config_sources=("runner.yaml", "agent.yaml"),
        config_digest="a" * 64,
        provenance={"suite": "fixture"},
    )
    bench = output.benchmark(0, "r2r_ce", "val_unseen")
    episode = bench.episode(
        3,
        "r2r_ce:val_unseen:3",
        {
            "case_id": "r2r_ce:val_unseen:3",
            "task": NavTask("task-3", NavGoal("goal-3", "Go forward.")),
            "environment_setup": {"scene_id": "scene.glb"},
        },
    )
    environment = episode.module("environment")
    environment.record({"profile": {"observation_channels": ["rgb"]}})
    environment.frame(
        "main_camera",
        np.zeros((4, 6, 3), dtype=np.uint8),
        {"observation_id": "0", "source_time": 1.0},
    )
    environment.frame(
        "main_camera",
        np.full((4, 6, 3), 255, dtype=np.uint8),
        {"observation_id": "1", "source_time": 2.0},
    )
    episode.event(
        {"sequence": 1, "actor": "agent", "name": "nav.observe", "outcome": "ok"}
    )
    assert episode.finish({"case_id": "r2r_ce:val_unseen:3", "metrics": {"sr": 1.0}}) == ()
    bench.finish(
        {
            "name": "r2r_ce",
            "split": "val_unseen",
            "aggregate_metrics": {"sr": 1.0},
            "episodes": [
                {
                    "index": 3,
                    "case_id": "r2r_ce:val_unseen:3",
                    "path": "episodes/000003-r2r_ce_val_unseen_3/result.json",
                }
            ],
            "error": None,
        }
    )
    manifest_path = output.finish(aggregate_metrics={"sr": 1.0}, status="completed")

    run_dir = tmp_path / "fixture-run"
    assert manifest_path == run_dir / "manifest.json"
    assert (run_dir / "config/resolved.yaml").is_file()
    assert (run_dir / "config/sources.json").is_file()
    episode_dir = (
        run_dir
        / "benches/000-r2r_ce-val_unseen/episodes/000003-r2r_ce_val_unseen_3"
    )
    assert read_json(episode_dir / "episode.json")["task"]["task_id"] == "task-3"
    assert read_json(episode_dir / "result.json")["metrics"] == {"sr": 1.0}
    assert len((episode_dir / "events.jsonl").read_text().splitlines()) == 1

    environment_record = read_json(episode_dir / "environment.json")
    stream = environment_record["streams"]["main_camera"]
    assert stream["status"] == "saved"
    assert stream["frame_count"] == 2
    assert stream["width"] == 6
    assert stream["height"] == 4
    video = run_dir / stream["path"]
    assert video.is_file() and video.stat().st_size == stream["size_bytes"]
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(probe.stdout)["streams"] == [{"width": 6, "height": 4}]
    assert read_json(manifest_path)["status"] == "completed"


def test_missing_optional_camera_is_explicit_and_does_not_create_video(tmp_path) -> None:
    output = RunOutput(
        {"root": str(tmp_path), "run_id": "no-camera"},
        resolved_config={},
        config_sources=(),
        config_digest="b" * 64,
        provenance={},
    )
    bench = output.benchmark(0, "dummy", "smoke")
    episode = bench.episode(0, "case", {"case_id": "case"})
    environment = episode.module("environment")
    environment.unavailable("main_camera", "environment has no visual sensor")
    assert episode.finish({"case_id": "case"}) == ()

    stream = read_json(episode.path / "environment.json")["streams"]["main_camera"]
    assert stream == {
        "status": "unavailable",
        "reason": "environment has no visual sensor",
    }
    assert not (episode.path / "artifacts/environment/main_camera.mp4").exists()


def test_required_camera_rejects_an_unavailable_stream(tmp_path) -> None:
    output = RunOutput(
        {
            "root": str(tmp_path),
            "run_id": "required-camera",
            "video": {"required": True},
        },
        resolved_config={},
        config_sources=(),
        config_digest="c" * 64,
        provenance={},
    )
    episode = output.benchmark(0, "fixture", "test").episode(
        0, "case", {"case_id": "case"}
    )

    with pytest.raises(HarnessError, match="required video stream.*unavailable"):
        episode.module("environment").unavailable(
            "main_camera", "fixture has no camera"
        )

from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

from benches import VLNPEBenchmark, VLNVerseBenchmark


def write_split(root, episodes):
    directory = root / "val_unseen"
    directory.mkdir(parents=True)
    with gzip.open(directory / "val_unseen.json.gz", "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes}, stream)


def test_vln_pe_loader_keeps_private_native_episode_and_filters_stairs(tmp_path) -> None:
    base = {
        "scene_id": "mp3d/scan/scan.glb",
        "start_position": [1.0, 2.0, 3.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "instruction": {"instruction_text": "Walk down the hall."},
        "info": {"geodesic_distance": 2.0},
    }
    write_split(
        tmp_path,
        [
            {
                **base,
                "trajectory_id": 10,
                "episode_id": 1,
                "reference_path": [[0, 0, 0], [1, 0.1, 0]],
            },
            {
                **base,
                "trajectory_id": 11,
                "episode_id": 2,
                "reference_path": [[0, 0, 0], [1, 0.5, 0]],
            },
        ],
    )

    cases = list(VLNPEBenchmark(tmp_path).cases())

    assert [case.setup["path_key"] for case in cases] == ["10_1"]
    assert cases[0].task.instruction == "Walk down the hall."
    assert "native_episode" not in cases[0].task.public
    assert cases[0].truth["info"]["geodesic_distance"] == 2.0


def test_vlnverse_instruction_variant_and_native_metric_mapping(tmp_path) -> None:
    write_split(
        tmp_path,
        [
            {
                "scan": "kujiale_scene",
                "trajectory_id": "trace",
                "episode_id": 7,
                "start_position": [1.0, 2.0, 3.0],
                "start_rotation": [1.0, 0.0, 0.0, 0.0],
                "instruction": {
                    "instruction_text": {
                        "formal": "Proceed to the kitchen.",
                        "casual": "Head to the kitchen.",
                    }
                },
                "reference_path": [[0, 0, 0], [1, 0, 0]],
            }
        ],
    )
    benchmark = VLNVerseBenchmark(tmp_path, instruction_type="casual")
    fixture = next(iter(benchmark.cases()))

    assert fixture.task.instruction == "Head to the kitchen."
    assert fixture.task.scene_id == "kujiale_scene"
    result = SimpleNamespace(
        environment={
            "native_metrics": {
                "revision_metric": [
                    {
                        "success": 1,
                        "spl": 0.5,
                        "NE": 0.25,
                        "osr": 1,
                        "ndtw": 0.75,
                        "TL": 4.0,
                    }
                ]
            }
        }
    )
    assert benchmark.score(fixture, result) == {
        "sr": 1.0,
        "spl": 0.5,
        "ne": 0.25,
        "os": 1.0,
        "ndtw": 0.75,
        "tl": 4.0,
    }

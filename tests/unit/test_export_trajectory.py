from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "export_trajectory.py"
SPEC = importlib.util.spec_from_file_location("export_trajectory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
export_trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_trajectory)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_select_record_reads_legacy_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 2,
        "benchmarks": [
            {
                "records": [
                    {"case_id": "case-1", "audit": [{"sequence": 1}]},
                    {"case_id": "case-2", "audit": [{"sequence": 2}]},
                ]
            }
        ],
    }

    record = export_trajectory.select_record(manifest_path, manifest, "case-2")

    assert record == {"case_id": "case-2", "audit": [{"sequence": 2}]}


def test_select_record_resolves_scoped_manifest_artifacts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "benches/bench/summary.json"
    result_path = tmp_path / "benches/bench/episodes/case/result.json"
    write_json(
        summary_path,
        {
            "episodes": [
                {"case_id": "case-3", "path": "episodes/case/result.json"}
            ]
        },
    )
    write_json(
        result_path,
        {
            "case_id": "case-3",
            "events_path": "events.jsonl",
            "environment_path": "environment.json",
            "metrics": {"sr": 1.0},
        },
    )
    (result_path.parent / "events.jsonl").write_text(
        json.dumps({"sequence": 1, "name": "vln.navigate.start"}) + "\n"
        + json.dumps({"sequence": 2, "name": "nav.move.discrete"})
        + "\n",
        encoding="utf-8",
    )
    write_json(result_path.parent / "environment.json", {"success": 1.0})
    manifest = {
        "schema_version": 3,
        "benchmarks": [{"path": "benches/bench/summary.json"}],
    }

    record = export_trajectory.select_record(manifest_path, manifest, "case-3")

    assert record["metrics"] == {"sr": 1.0}
    assert [event["sequence"] for event in record["audit"]] == [1, 2]
    assert record["environment"] == {"success": 1.0}


def test_scoped_manifest_rejects_paths_outside_run(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run/manifest.json"
    manifest = {
        "schema_version": 3,
        "benchmarks": [{"path": "../outside.json"}],
    }

    try:
        export_trajectory.select_record(manifest_path, manifest, None)
    except ValueError as error:
        assert "escapes its output scope" in str(error)
    else:
        raise AssertionError("expected a path traversal error")

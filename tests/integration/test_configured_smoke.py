from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness.app import run_config


ROOT = Path(__file__).resolve().parents[2]


def test_yaml_to_batch_manifest_smoke(tmp_path) -> None:
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        "extends: "
        f"{ROOT / 'config/runners/dummy_passthrough.yaml'}\n"
        f"output:\n  root: {tmp_path / 'run'}\n"
    )

    suite, manifest = asyncio.run(
        run_config(
            runner,
            ROOT / "config/agents/passthrough.yaml",
        )
    )
    summary = suite.runs[0]
    document = json.loads(manifest.read_text())

    assert len(summary.records) == 2
    assert document["aggregate_metrics"] == {"success": 1.0}
    assert document["benchmarks"][0]["validation_status"] == "contract"
    assert document["config_digest"]
    assert document["provenance"]["simulator"] == "dummy"
    records = document["benchmarks"][0]["records"]
    assert all(record["terminal"]["actor"] == "agent" for record in records)
    assert all(record["audit"][-1]["name"] == "nav.stop" for record in records)


def test_runner_executes_multiple_referenced_benches(tmp_path) -> None:
    bench = ROOT / "config/benches/dummy.yaml"
    runner = tmp_path / "multi.yaml"
    runner.write_text(
        "runner:\n"
        "  benches:\n"
        f"    - {bench}\n"
        f"    - {bench}\n"
        "  bench_parallelism: 2\n"
        "  task_parallelism: 1\n"
        "  timeout_s: 10\n"
        f"output:\n  root: {tmp_path / 'multi-run'}\n"
    )

    suite, manifest = asyncio.run(
        run_config(runner, ROOT / "config/agents/passthrough.yaml")
    )
    document = json.loads(manifest.read_text())

    assert [run.benchmark for run in suite.runs] == [
        "dummy_navigation",
        "dummy_navigation",
    ]
    assert [len(run.records) for run in suite.runs] == [2, 2]
    assert len(document["benchmarks"]) == 2

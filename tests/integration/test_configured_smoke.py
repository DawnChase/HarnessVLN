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


def test_runner_isolates_one_bench_construction_failure(tmp_path) -> None:
    broken_bench = tmp_path / "broken-bench.yaml"
    broken_bench.write_text(
        "benchmark:\n"
        "  factory: benches.dummy:DummyBenchmark\n"
        "  params:\n"
        "    split: broken\n"
        f"  environment: {ROOT / 'config/envs/dummy.yaml'}\n"
    )
    runner = tmp_path / "partially-broken-runner.yaml"
    runner.write_text(
        "runner:\n"
        "  benches:\n"
        f"    - {broken_bench}\n"
        f"    - {ROOT / 'config/benches/dummy.yaml'}\n"
        "  bench_parallelism: 2\n"
        "  task_parallelism: 1\n"
        "  timeout_s: 10\n"
        f"output:\n  root: {tmp_path / 'partial-run'}\n"
    )

    suite, manifest = asyncio.run(
        run_config(runner, ROOT / "config/agents/passthrough.yaml")
    )
    document = json.loads(manifest.read_text())

    failed, completed = suite.runs
    assert failed.benchmark == "benches.dummy:DummyBenchmark"
    assert failed.split == "broken"
    assert failed.validation_status == "unavailable"
    assert failed.records == ()
    assert failed.error is not None and "missing" in failed.error
    assert completed.error is None
    assert len(completed.records) == 2
    assert document["benchmarks"][0]["error"] == failed.error
    assert document["benchmarks"][0]["records"] == []
    assert document["benchmarks"][1]["error"] is None
    assert len(document["benchmarks"][1]["records"]) == 2

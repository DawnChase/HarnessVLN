from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from harness.app import execute_runner
from harness.errors import HarnessError


ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path):
    return json.loads(path.read_text())


def test_runner_config_pins_episode_workers_to_distinct_devices(
    tmp_path, monkeypatch, capfd
) -> None:
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        "environment:\n"
        "  factory: tests.fixtures.device_environment:from_episode\n"
    )
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "benchmark:\n"
        "  factory: benches.dummy:DummyBenchmark\n"
        "  params:\n"
        "    split: process-pool\n"
        "    cases:\n"
        "      - {task_id: first, instruction: Forward., target: 2}\n"
        "      - {task_id: second, instruction: Backward., target: -2}\n"
        "      - {task_id: third, instruction: Forward., target: 1}\n"
        "      - {task_id: fourth, instruction: Backward., target: -1}\n"
        f"  environment: {environment}\n"
    )
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        "runner:\n"
        f"  benches: [{benchmark}]\n"
        "  devices: [17, 23]\n"
        "  workers_per_device: 1\n"
        "  task_parallelism: 2\n"
        "  timeout_s: 10\n"
        "  shutdown_timeout_s: 2\n"
        f"output:\n  root: {tmp_path / 'runs'}\n"
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-device")

    summary, manifest = asyncio.run(
        execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent-device"
    assert [record.index for record in summary.benchmarks[0].records] == [0, 1, 2, 3]
    assert all(record.error is None for record in summary.benchmarks[0].records)
    result_paths = sorted(manifest.parent.glob("benches/*/episodes/*/result.json"))
    results = [_read_json(path) for path in result_paths]
    environments = [_read_json(path.parent / "environment.json") for path in result_paths]

    assert {item["resources"]["gpu"]["physical_device"] for item in results} == {
        17,
        23,
    }
    assert {item["resources"]["gpu"]["local_device"] for item in results} == {0}
    assert len({item["resources"]["pid"] for item in results}) == 2
    assert {
        item["result"]["cuda_visible_devices"] for item in environments
    } == {"17", "23"}
    assert {
        item["result"]["worker_pid"] for item in environments
    } == {item["resources"]["pid"] for item in results}
    worker_logs = {
        item["resources"]["worker_log"] for item in results
    }
    assert len(worker_logs) == 2
    log_text = "".join(
        (manifest.parent / path).read_text() for path in sorted(worker_logs)
    )
    assert "fixture Habitat scene switch stdout" in log_text
    assert "fixture Habitat scene switch stderr" in log_text
    terminal = capfd.readouterr()
    assert "fixture Habitat scene switch" not in terminal.out
    assert "fixture Habitat scene switch" not in terminal.err
    assert _read_json(manifest)["status"] == "completed"


def test_runner_rejects_more_processes_than_configured_slots(tmp_path) -> None:
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        "runner:\n"
        f"  benches: [{ROOT / 'config/benches/dummy.yaml'}]\n"
        "  devices: [0]\n"
        "  task_parallelism: 2\n"
    )

    try:
        asyncio.run(
            execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
        )
    except HarnessError as error:
        assert "exceeds configured GPU worker slots" in str(error)
    else:
        raise AssertionError("runner accepted more processes than GPU slots")


def test_parallel_benches_share_one_global_device_pool(tmp_path) -> None:
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        "environment:\n"
        "  factory: tests.fixtures.device_environment:from_episode\n"
    )
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "benchmark:\n"
        "  factory: benches.dummy:DummyBenchmark\n"
        "  params:\n"
        "    split: process-pool\n"
        "    cases:\n"
        "      - {task_id: first, instruction: Forward., target: 1}\n"
        "      - {task_id: second, instruction: Backward., target: -1}\n"
        f"  environment: {environment}\n"
    )
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        "runner:\n"
        f"  benches: [{benchmark}, {benchmark}]\n"
        "  devices: [31, 37]\n"
        "  workers_per_device: 1\n"
        "  bench_parallelism: 2\n"
        "  task_parallelism: 1\n"
        "  timeout_s: 10\n"
        "  shutdown_timeout_s: 2\n"
        f"output:\n  root: {tmp_path / 'runs'}\n"
    )

    summary, manifest = asyncio.run(
        execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
    )

    assert len(summary.benchmarks) == 2
    benchmark_dirs = sorted((manifest.parent / "benches").iterdir())
    assigned_devices = []
    for benchmark_dir in benchmark_dirs:
        results = [
            _read_json(path)
            for path in sorted(benchmark_dir.glob("episodes/*/result.json"))
        ]
        devices = {
            result["resources"]["gpu"]["physical_device"] for result in results
        }
        assert len(devices) == 1
        assigned_devices.extend(devices)
    assert set(assigned_devices) == {31, 37}

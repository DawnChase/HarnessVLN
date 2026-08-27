from __future__ import annotations

import asyncio
import json
from pathlib import Path

from benches.dummy import DummyBenchmark
from envs import DummyNavigationEnvironment
from harness.app import execute_runner


ROOT = Path(__file__).resolve().parents[2]


class ScoreFailureBenchmark(DummyBenchmark):
    def score(self, case, result):
        del case, result
        raise RuntimeError("fixture scoring failed")


class PartiallyInvalidMetricBenchmark(DummyBenchmark):
    def score(self, case, result):
        if case.case_id == "invalid":
            return {"success": float("nan")}
        return super().score(case, result)


class InvalidConfiguredAgent:
    required_tools = frozenset()


class StopConfiguredAgent:
    required_tools = frozenset()

    async def run(self, context):
        await context.nav.stop("completed")


class TrackingDummyEnvironment(DummyNavigationEnvironment):
    total_start_calls = 0

    async def start(self, task):
        type(self).total_start_calls += 1
        return await super().start(task)


_configured_agent_calls = 0


def partially_invalid_agent():
    global _configured_agent_calls
    _configured_agent_calls += 1
    return InvalidConfiguredAgent() if _configured_agent_calls == 1 else StopConfiguredAgent()


def tracking_dummy_environment(episode):
    return TrackingDummyEnvironment(
        episode.setup.get("goal_stream", (episode.task.goal,)),
        targets=episode.setup.get("targets", (0,)),
    )


def test_yaml_to_batch_manifest_smoke(tmp_path) -> None:
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        "extends: "
        f"{ROOT / 'config/runners/dummy_passthrough.yaml'}\n"
        f"output:\n  root: {tmp_path / 'run'}\n"
    )

    run_summary, manifest = asyncio.run(
        execute_runner(
            runner,
            ROOT / "config/agents/passthrough.yaml",
        )
    )
    summary = run_summary.benchmarks[0]
    document = json.loads(manifest.read_text())

    assert len(summary.records) == 2
    assert document["aggregate_metrics"] == {"success": 1.0}
    assert document["benchmarks"][0]["validation_status"] == "contract"
    assert document["config_digest"]
    assert document["provenance"]["simulator"] == "dummy"
    records = document["benchmarks"][0]["records"]
    assert all(record["error_stage"] is None for record in records)
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

    run_summary, manifest = asyncio.run(
        execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
    )
    document = json.loads(manifest.read_text())

    assert [bench.benchmark for bench in run_summary.benchmarks] == [
        "dummy_navigation",
        "dummy_navigation",
    ]
    assert [len(bench.records) for bench in run_summary.benchmarks] == [2, 2]
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

    run_summary, manifest = asyncio.run(
        execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
    )
    document = json.loads(manifest.read_text())

    failed, completed = run_summary.benchmarks
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


def test_score_failure_manifest_retains_execution_evidence(tmp_path) -> None:
    benchmark = tmp_path / "score-failure-bench.yaml"
    benchmark.write_text(
        "benchmark:\n"
        f"  factory: {__name__}:ScoreFailureBenchmark\n"
        "  params:\n"
        "    split: fixture\n"
        "    cases:\n"
        "      - {task_id: scored, instruction: Stay., target: 0}\n"
        f"  environment: {ROOT / 'config/envs/dummy.yaml'}\n"
    )
    runner = tmp_path / "score-failure-runner.yaml"
    runner.write_text(
        "runner:\n"
        f"  benches: [{benchmark}]\n"
        "  task_parallelism: 1\n"
        f"output:\n  root: {tmp_path / 'score-failure-run'}\n"
    )

    run_summary, manifest = asyncio.run(
        execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
    )
    record = run_summary.benchmarks[0].records[0]
    document = json.loads(manifest.read_text())
    serialized = document["benchmarks"][0]["records"][0]

    assert record.result is not None
    assert record.error_stage == "score"
    assert serialized["error_stage"] == "score"
    assert serialized["terminal"]["status"] == "completed"
    assert serialized["environment"]["position"] == 0
    assert serialized["audit"][-1]["name"] == "nav.stop"
    assert serialized["metrics"] == {}


def test_invalid_metric_is_isolated_before_manifest_aggregation(tmp_path) -> None:
    benchmark = tmp_path / "partially-invalid-metric-bench.yaml"
    benchmark.write_text(
        "benchmark:\n"
        f"  factory: {__name__}:PartiallyInvalidMetricBenchmark\n"
        "  params:\n"
        "    split: fixture\n"
        "    cases:\n"
        "      - {task_id: invalid, instruction: Stay., target: 0}\n"
        "      - {task_id: valid, instruction: Stay., target: 0}\n"
        f"  environment: {ROOT / 'config/envs/dummy.yaml'}\n"
    )
    runner = tmp_path / "partially-invalid-metric-runner.yaml"
    runner.write_text(
        "runner:\n"
        f"  benches: [{benchmark}]\n"
        "  task_parallelism: 2\n"
        f"output:\n  root: {tmp_path / 'partially-invalid-metric-run'}\n"
    )

    run_summary, manifest = asyncio.run(
        execute_runner(runner, ROOT / "config/agents/passthrough.yaml")
    )

    def reject_nonstandard_constant(value: str) -> None:
        raise AssertionError(f"manifest contains non-standard JSON constant: {value}")

    document = json.loads(
        manifest.read_text(), parse_constant=reject_nonstandard_constant
    )
    failed, completed = run_summary.benchmarks[0].records
    serialized_failed, serialized_completed = document["benchmarks"][0]["records"]

    assert failed.error_stage == "score"
    assert failed.result is not None
    assert failed.result.terminal.status == "completed"
    assert failed.metrics == {}
    assert completed.error is None
    assert completed.metrics == {"success": 1.0}
    assert serialized_failed["error_stage"] == "score"
    assert serialized_failed["terminal"]["status"] == "completed"
    assert serialized_failed["metrics"] == {}
    assert serialized_completed["error_stage"] is None
    assert serialized_completed["metrics"] == {"success": 1.0}
    assert document["benchmarks"][0]["aggregate_metrics"] == {"success": 1.0}
    assert document["aggregate_metrics"] == {"success": 1.0}


def test_invalid_plugin_contract_is_a_stack_error_without_starting_env(tmp_path) -> None:
    global _configured_agent_calls
    _configured_agent_calls = 0
    TrackingDummyEnvironment.total_start_calls = 0

    environment = tmp_path / "tracking-env.yaml"
    environment.write_text(
        "environment:\n"
        f"  factory: {__name__}:tracking_dummy_environment\n"
    )
    benchmark = tmp_path / "partially-invalid-stack-bench.yaml"
    benchmark.write_text(
        "benchmark:\n"
        "  factory: benches.dummy:DummyBenchmark\n"
        "  params:\n"
        "    split: fixture\n"
        "    cases:\n"
        "      - {task_id: invalid, instruction: Stay., target: 0}\n"
        "      - {task_id: valid, instruction: Stay., target: 0}\n"
        f"  environment: {environment}\n"
    )
    runner = tmp_path / "partially-invalid-stack-runner.yaml"
    runner.write_text(
        "runner:\n"
        f"  benches: [{benchmark}]\n"
        "  task_parallelism: 1\n"
        f"output:\n  root: {tmp_path / 'partially-invalid-stack-run'}\n"
    )
    agent = tmp_path / "partially-invalid-agent.yaml"
    agent.write_text(
        "agent:\n"
        f"  factory: {__name__}:partially_invalid_agent\n"
    )

    run_summary, manifest = asyncio.run(execute_runner(runner, agent))
    failed, completed = run_summary.benchmarks[0].records
    document = json.loads(manifest.read_text())
    serialized_failed, serialized_completed = document["benchmarks"][0]["records"]

    assert failed.error_stage == "stack"
    assert failed.result is None
    assert failed.error is not None and "InvalidConfiguredAgent" in failed.error
    assert completed.error is None
    assert completed.result is not None
    assert completed.result.terminal.status == "completed"
    assert completed.metrics == {"success": 1.0}
    assert TrackingDummyEnvironment.total_start_calls == 1
    assert serialized_failed["error_stage"] == "stack"
    assert serialized_failed["terminal"] is None
    assert serialized_completed["error_stage"] is None
    assert serialized_completed["terminal"]["status"] == "completed"

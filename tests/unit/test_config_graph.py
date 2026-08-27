from __future__ import annotations

from pathlib import Path

import pytest

from harness.config import (
    load_agent_config,
    load_benchmark_config,
    load_environment_config,
    load_runner_config,
)
from harness.errors import HarnessError


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_referenced_configs_resolve_ownership_and_relative_paths(tmp_path: Path) -> None:
    write(
        tmp_path / "envs/base.yaml",
        """
environment:
  factory: envs.dummy:from_episode
  params: {start_position: 0}
interactive: {scene_id: dummy_scene}
provenance: {simulator: dummy}
""",
    )
    write(
        tmp_path / "envs/shifted.yaml",
        """
extends: base.yaml
environment:
  params: {start_position: 2}
""",
    )
    write(
        tmp_path / "benches/base.yaml",
        """
benchmark:
  factory: benches.dummy:DummyBenchmark
  params:
    split: smoke
    cases: [{task_id: one, instruction: Stop., target: 0}]
  environment: ../envs/shifted.yaml
provenance: {dataset: fixture}
""",
    )
    bench = write(
        tmp_path / "profiles/bench.yaml",
        """
extends: ../benches/base.yaml
benchmark:
  params:
    split: derived
""",
    )
    runner = write(
        tmp_path / "runners/smoke.yaml",
        """
runner:
  benches: [../profiles/bench.yaml]
  bench_parallelism: 1
  task_parallelism: 2
output: {root: runs/smoke}
provenance: {run: smoke}
""",
    )
    write(
        tmp_path / "vln/dummy.yaml",
        """
vln:
  factory: vln.dummy:DummyVLNNavigator
  params: {max_steps: 10}
provenance: {model: dummy}
""",
    )
    write(
        tmp_path / "memory/dummy.yaml",
        """
memory:
  factory: memory.dummy_landmark:DummyLandmarkMemory
  params: {root: runs/memory, writeback: false}
""",
    )
    agent = write(
        tmp_path / "agents/normal.yaml",
        """
agent:
  factory: agents.normal_agent:NormalAgent
  params: {model: fixture}
  vln: ../vln/dummy.yaml
  memory: ../memory/dummy.yaml
provenance: {agent_protocol: native-tools}
""",
    )

    run_config = load_runner_config(runner)
    agent_config = load_agent_config(agent)

    assert run_config.settings == {
        "bench_parallelism": 1,
        "task_parallelism": 2,
    }
    assert len(run_config.benches) == 1
    resolved_bench = run_config.benches[0]
    assert resolved_bench.benchmark.params["split"] == "derived"
    assert resolved_bench.environment.component.params["start_position"] == 2
    assert resolved_bench.environment.interactive["scene_id"] == "dummy_scene"
    assert str(bench.resolve()) in run_config.sources
    assert run_config.data["provenance"] == {
        "simulator": "dummy",
        "dataset": "fixture",
        "run": "smoke",
    }
    assert agent_config.core.factory == "agents.normal_agent:NormalAgent"
    assert agent_config.vln is not None
    assert agent_config.vln.factory == "vln.dummy:DummyVLNNavigator"
    assert agent_config.memory is not None
    assert agent_config.memory.params["writeback"] is False
    assert agent_config.data["provenance"] == {
        "model": "dummy",
        "agent_protocol": "native-tools",
    }
    assert len(run_config.digest) == 64
    assert len(agent_config.digest) == 64


def test_config_extends_cycle_reports_the_complete_chain(tmp_path: Path) -> None:
    first = write(tmp_path / "first.yaml", "extends: second.yaml\n")
    write(tmp_path / "second.yaml", "extends: first.yaml\n")

    with pytest.raises(HarnessError, match="config extends cycle") as caught:
        load_agent_config(first)

    assert str(first.resolve()) in str(caught.value)
    assert str((tmp_path / "second.yaml").resolve()) in str(caught.value)


def test_missing_referenced_config_fails_at_the_owner_boundary(tmp_path: Path) -> None:
    agent = write(
        tmp_path / "agent.yaml",
        """
agent:
  factory: agents.passthrough:PassthroughVLNAgent
  vln: missing.yaml
""",
    )

    with pytest.raises(HarnessError, match="failed to load vln config"):
        load_agent_config(agent)


def test_runner_requires_at_least_one_bench_reference(tmp_path: Path) -> None:
    runner = write(tmp_path / "runner.yaml", "runner: {benches: []}\n")

    with pytest.raises(HarnessError, match="invalid runner config"):
        load_runner_config(runner)


@pytest.mark.parametrize(
    ("kind", "document", "loader"),
    (
        (
            "agent",
            "agent: {factory: agents.passthrough:PassthroughVLNAgent, scope: run}\n",
            load_agent_config,
        ),
        (
            "environment",
            "environment: {factory: envs.dummy:from_episode, scope: run}\n",
            load_environment_config,
        ),
    ),
)
def test_task_owned_components_reject_run_scope(
    tmp_path: Path, kind: str, document: str, loader
) -> None:
    config = write(tmp_path / f"{kind}.yaml", document)

    with pytest.raises(HarnessError, match=f"{kind} scope must be task"):
        loader(config)


def test_benchmark_rejects_run_scope(tmp_path: Path) -> None:
    write(
        tmp_path / "env.yaml",
        "environment: {factory: envs.dummy:from_episode}\n",
    )
    benchmark = write(
        tmp_path / "bench.yaml",
        """
benchmark:
  factory: benches.dummy:DummyBenchmark
  scope: run
  environment: env.yaml
""",
    )

    with pytest.raises(HarnessError, match="benchmark scope must be task"):
        load_benchmark_config(benchmark)

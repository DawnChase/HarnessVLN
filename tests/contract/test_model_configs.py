from pathlib import Path

import pytest

from benches.base import BenchmarkCase
from harness.config import (
    load_agent_config,
    load_benchmark_config,
    load_runner_config,
)
from harness.requirements import check_navigation_requirements
from schemas import NavGoal, NavTask


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("agent_profile", "model_name", "revision"),
    (
        (
            "passthrough_streamvln.yaml",
            "streamvln",
            "f1f76c66083c362ddfcd2610167f9c4e4a46c027",
        ),
        (
            "passthrough_janusvln.yaml",
            "janusvln",
            "33f932a4ea6bdc34afca9f5b79a8b4537cd02509",
        ),
        (
            "passthrough_dualvln.yaml",
            "dualvln",
            "a698a9e898b4001621a319e1bc89f02ec715cc86",
        ),
    ),
)
def test_agent_profile_owns_vln_configuration(
    agent_profile: str, model_name: str, revision: str
) -> None:
    config = load_agent_config(ROOT / "config/agents" / agent_profile)
    assert config.vln is not None
    navigator = config.vln.create()

    assert navigator.model_name == model_name
    assert config.vln.scope == "session"
    if model_name != "dualvln":
        assert config.vln.params["worker_options"]["local_files_only"] is True
    assert config.data["provenance"]["checkpoint_revision"] == revision
    assert navigator.requirements["camera"]["height"] == 480
    assert navigator.requirements["camera"]["width"] == 640
    assert navigator.requirements["camera"]["hfov_deg"] == 79


@pytest.mark.parametrize(
    "agent_profile",
    ("passthrough_streamvln.yaml", "passthrough_janusvln.yaml"),
)
def test_r2r_environment_composes_with_supported_vln(agent_profile: str) -> None:
    agent = load_agent_config(ROOT / "config/agents" / agent_profile)
    bench = load_benchmark_config(ROOT / "config/benches/r2r_ce.yaml")
    case = BenchmarkCase(
        "r2r_ce:fixture:1",
        NavTask("r2r:fixture", NavGoal("goal", "Go forward.")),
    )
    environment = bench.environment.component.create(
        episode=case.environment_episode
    )
    assert agent.vln is not None
    navigator = agent.vln.create()

    check_navigation_requirements(
        type(navigator).__name__, navigator.requirements, environment.profile
    )
    assert environment.profile.observation_channels == frozenset(
        {"rgb", "depth", "gps", "compass", "pose", "camera_intrinsics"}
    )


@pytest.mark.parametrize(
    ("agent_profile", "runner_profile"),
    (
        ("passthrough_streamvln.yaml", "r2r_streamvln.yaml"),
        ("passthrough_janusvln.yaml", "r2r_janusvln.yaml"),
        ("passthrough_dualvln.yaml", "r2r_dualvln.yaml"),
    ),
)
def test_full_r2r_run_configs_satisfy_model_requirements(
    agent_profile: str,
    runner_profile: str,
) -> None:
    runner = load_runner_config(ROOT / "config/runners" / runner_profile)
    agent = load_agent_config(ROOT / "config/agents" / agent_profile)
    bench = runner.benches[0]
    case = BenchmarkCase(
        "r2r_ce:fixture:1",
        NavTask("r2r:fixture", NavGoal("goal", "Go forward.")),
    )
    environment = bench.environment.component.create(episode=case.environment_episode)
    assert agent.vln is not None
    navigator = agent.vln.create()

    check_navigation_requirements(
        type(navigator).__name__, navigator.requirements, environment.profile
    )
    assert runner.settings["task_parallelism"] == 1
    assert "max_cases" not in runner.settings
    assert (
        bench.environment.component.params["native_factory_params"][
            "config_values"
        ]["habitat.task.measurements.success.success_distance"]
        == 3.0
    )


def test_smoke_override_bounds_a_full_model_run() -> None:
    resolved = load_runner_config(ROOT / "config/runners/smoke_one.yaml")

    assert resolved.settings["max_cases"] == 1
    assert resolved.data["provenance"]["evaluation_scope"] == "first-case-smoke"

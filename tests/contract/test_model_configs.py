from pathlib import Path

import pytest

from benches.base import BenchmarkCase
from harness.config import ComponentSpec, load_config
from harness.requirements import check_navigation_requirements
from schemas import NavGoal, NavTask


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("fragment", "model_name", "revision"),
    (
        (
            "streamvln.yaml",
            "streamvln",
            "f1f76c66083c362ddfcd2610167f9c4e4a46c027",
        ),
        (
            "janusvln.yaml",
            "janusvln",
            "33f932a4ea6bdc34afca9f5b79a8b4537cd02509",
        ),
        (
            "dualvln.yaml",
            "dualvln",
            "a698a9e898b4001621a319e1bc89f02ec715cc86",
        ),
    ),
)
def test_vln_fragment_composes_with_run_and_benchmark(
    fragment: str, model_name: str, revision: str
) -> None:
    resolved = load_config(
        (
            ROOT / "config/benches/dummy.yaml",
            ROOT / "config/agents/passthrough.yaml",
            ROOT / "config/envs/dummy.yaml",
            ROOT / "config/runs/dummy_passthrough.yaml",
            ROOT / "config/vln" / fragment,
        )
    )

    spec = resolved.data["stack"]["vln"]
    navigator = ComponentSpec.from_config(spec).create()

    assert navigator.model_name == model_name
    assert spec["scope"] == "run"
    if model_name != "dualvln":
        assert spec["params"]["worker_options"]["local_files_only"] is True
    assert resolved.data["provenance"]["checkpoint_revision"] == revision
    assert navigator.requirements["camera"]["height"] == 480
    assert navigator.requirements["camera"]["width"] == 640
    assert navigator.requirements["camera"]["hfov_deg"] == 79


@pytest.mark.parametrize("fragment", ("streamvln.yaml", "janusvln.yaml"))
def test_r2r_environment_composes_with_supported_vln(fragment: str) -> None:
    resolved = load_config(
        (
            ROOT / "config/benches/r2r_ce.yaml",
            ROOT / "config/agents/passthrough.yaml",
            ROOT / "config/envs/habitat_r2r.yaml",
            ROOT / "config/vln" / fragment,
            ROOT / "config/runs/dummy_passthrough.yaml",
        )
    )
    case = BenchmarkCase(
        "r2r_ce:fixture:1",
        NavTask("r2r:fixture", NavGoal("goal", "Go forward.")),
    )
    environment = ComponentSpec.from_config(
        resolved.data["stack"]["environment"]
    ).create(episode=case.environment_episode)
    navigator = ComponentSpec.from_config(resolved.data["stack"]["vln"]).create()

    check_navigation_requirements(
        type(navigator).__name__, navigator.requirements, environment.profile
    )
    assert resolved.data["runner"]["parallelism"] == 2
    assert environment.profile.observation_channels == frozenset(
        {"rgb", "depth", "gps", "compass", "pose", "camera_intrinsics"}
    )


@pytest.mark.parametrize(
    ("model_fragment", "run_fragment", "environment_overlays"),
    (
        ("streamvln.yaml", "r2r_streamvln.yaml", ()),
        ("janusvln.yaml", "r2r_janusvln.yaml", ()),
        (
            "dualvln.yaml",
            "r2r_dualvln.yaml",
            ("habitat_r2r_dualvln.yaml",),
        ),
    ),
)
def test_full_r2r_run_configs_satisfy_model_requirements(
    model_fragment: str,
    run_fragment: str,
    environment_overlays: tuple[str, ...],
) -> None:
    paths = [
        ROOT / "config/benches/r2r_ce.yaml",
        ROOT / "config/agents/passthrough.yaml",
        ROOT / "config/envs/habitat_r2r.yaml",
        *(ROOT / "config/envs" / name for name in environment_overlays),
        ROOT / "config/vln" / model_fragment,
        ROOT / "config/runs" / run_fragment,
    ]
    resolved = load_config(paths)
    case = BenchmarkCase(
        "r2r_ce:fixture:1",
        NavTask("r2r:fixture", NavGoal("goal", "Go forward.")),
    )
    environment = ComponentSpec.from_config(
        resolved.data["stack"]["environment"]
    ).create(episode=case.environment_episode)
    navigator = ComponentSpec.from_config(resolved.data["stack"]["vln"]).create()

    check_navigation_requirements(
        type(navigator).__name__, navigator.requirements, environment.profile
    )
    assert resolved.data["runner"]["parallelism"] == 1
    assert "max_cases" not in resolved.data["runner"]
    assert (
        resolved.data["stack"]["environment"]["params"]["native_factory_params"][
            "config_values"
        ]["habitat.task.measurements.success.success_distance"]
        == 3.0
    )


def test_smoke_override_bounds_a_full_model_run() -> None:
    resolved = load_config(
        (
            ROOT / "config/benches/r2r_ce.yaml",
            ROOT / "config/agents/passthrough.yaml",
            ROOT / "config/envs/habitat_r2r.yaml",
            ROOT / "config/vln/streamvln.yaml",
            ROOT / "config/runs/r2r_streamvln.yaml",
            ROOT / "config/runs/smoke_one.yaml",
        )
    )

    assert resolved.data["runner"]["max_cases"] == 1
    assert resolved.data["provenance"]["run_scope"] == "first-case-smoke"

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
    ),
)
def test_vln_fragment_composes_with_run_and_benchmark(
    fragment: str, model_name: str, revision: str
) -> None:
    resolved = load_config(
        (
            ROOT / "config/benches/dummy.yaml",
            ROOT / "config/runs/dummy_passthrough.yaml",
            ROOT / "config/vln" / fragment,
        )
    )

    spec = resolved.data["stack"]["vln"]
    navigator = ComponentSpec.from_config(spec).create()

    assert navigator.model_name == model_name
    assert spec["params"]["worker_options"]["local_files_only"] is True
    assert resolved.data["provenance"]["checkpoint_revision"] == revision
    assert navigator.requirements["camera"] == {
        "height": 480,
        "width": 640,
        "hfov_deg": 79,
    }


@pytest.mark.parametrize("fragment", ("streamvln.yaml", "janusvln.yaml"))
def test_r2r_environment_composes_with_supported_vln(fragment: str) -> None:
    resolved = load_config(
        (
            ROOT / "config/benches/r2r_ce.yaml",
            ROOT / "config/runs/dummy_passthrough.yaml",
            ROOT / "config/envs/habitat_r2r.yaml",
            ROOT / "config/vln" / fragment,
        )
    )
    case = BenchmarkCase(
        "r2r_ce:fixture:1",
        NavTask("r2r:fixture", NavGoal("goal", "Go forward.")),
    )
    environment = ComponentSpec.from_config(
        resolved.data["stack"]["environment"]
    ).create(case=case)
    navigator = ComponentSpec.from_config(resolved.data["stack"]["vln"]).create()

    check_navigation_requirements(
        type(navigator).__name__, navigator.requirements, environment.profile
    )
    assert resolved.data["runner"]["parallelism"] == 2
    assert environment.profile.observation_channels == frozenset(
        {"rgb", "depth", "gps", "compass", "pose", "camera_intrinsics"}
    )

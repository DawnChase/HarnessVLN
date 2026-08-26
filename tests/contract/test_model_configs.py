from pathlib import Path

import pytest

from harness.config import ComponentSpec, load_config


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

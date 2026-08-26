from pathlib import Path

from harness.config import load_config, load_symbol


ROOT = Path(__file__).resolve().parents[2]


def test_streamvln_fragment_composes_with_run_and_benchmark() -> None:
    resolved = load_config(
        (
            ROOT / "config/benches/dummy.yaml",
            ROOT / "config/runs/dummy_passthrough.yaml",
            ROOT / "config/vln/streamvln.yaml",
        )
    )

    spec = resolved.data["stack"]["vln"]
    navigator = load_symbol(spec["factory"])

    assert navigator.model_name == "streamvln"
    assert spec["params"]["worker_options"]["local_files_only"] is True
    assert resolved.data["provenance"]["checkpoint_revision"] == (
        "f1f76c66083c362ddfcd2610167f9c4e4a46c027"
    )

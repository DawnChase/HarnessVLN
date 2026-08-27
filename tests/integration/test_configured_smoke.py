from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness.app import run_config


ROOT = Path(__file__).resolve().parents[2]


def test_yaml_to_batch_manifest_smoke(tmp_path) -> None:
    override = tmp_path / "output.yaml"
    override.write_text(f"output:\n  root: {tmp_path / 'run'}\n")

    summary, manifest = asyncio.run(
        run_config(
            (
                ROOT / "config/benches/dummy.yaml",
                ROOT / "config/agents/passthrough.yaml",
                ROOT / "config/envs/dummy.yaml",
                ROOT / "config/vln/dummy.yaml",
                ROOT / "config/runs/dummy_passthrough.yaml",
                override,
            )
        )
    )
    document = json.loads(manifest.read_text())

    assert len(summary.records) == 2
    assert document["aggregate_metrics"] == {"success": 1.0}
    assert document["benchmark"]["validation_status"] == "contract"
    assert document["config_digest"]
    assert document["provenance"]["simulator"] == "dummy"
    assert all(record["terminal"]["actor"] == "agent" for record in document["records"])
    assert all(record["audit"][-1]["name"] == "nav.stop" for record in document["records"])

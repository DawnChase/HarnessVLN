from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_streamvln_real_trace_is_internally_consistent() -> None:
    path = ROOT / "docs/traces/streamvln-r2r-val-unseen-1.json"
    trace = json.loads(path.read_text(encoding="utf-8"))

    assert trace["schema_version"] == 1
    assert trace["case_id"] == "r2r_ce:val_unseen:1"
    assert trace["terminal"]["actor"] == "agent"
    assert trace["terminal"]["status"] == "completed"
    assert trace["metrics"] == {
        "sr": 1.0,
        "spl": 1.0,
        "ne": 0.5811688303947449,
        "os": 1.0,
    }
    assert len(trace["actions"]) == trace["event_counts"]["nav.move.discrete"]
    assert set(trace["actions"]) <= {"forward", "turn_left", "turn_right"}
    assert trace["event_counts"]["nav.goal.finish"] == 1
    assert trace["event_counts"]["nav.stop"] == 1
    assert trace["cleanup_errors"] == []

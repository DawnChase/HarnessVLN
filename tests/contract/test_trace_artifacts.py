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


def test_janusvln_real_trace_is_internally_consistent() -> None:
    path = ROOT / "docs/traces/janusvln-r2r-val-unseen-1.json"
    trace = json.loads(path.read_text(encoding="utf-8"))

    assert trace["schema_version"] == 1
    assert trace["case_id"] == "r2r_ce:val_unseen:1"
    assert trace["model"] == "JanusVLN"
    assert trace["terminal"]["actor"] == "agent"
    assert trace["terminal"]["status"] == "completed"
    assert trace["metrics"] == {
        "sr": 1.0,
        "spl": 1.0,
        "ne": 1.4598493576049805,
        "os": 1.0,
    }
    assert len(trace["actions"]) == trace["event_counts"]["nav.move.discrete"]
    assert set(trace["actions"]) <= {"forward", "turn_left", "turn_right"}
    assert trace["event_counts"]["nav.goal.finish"] == 1
    assert trace["event_counts"]["nav.stop"] == 1
    assert trace["cleanup_errors"] == []


def test_dualvln_real_trace_is_internally_consistent() -> None:
    path = ROOT / "docs/traces/dualvln-r2r-val-unseen-1.json"
    trace = json.loads(path.read_text(encoding="utf-8"))

    assert trace["schema_version"] == 1
    assert trace["case_id"] == "r2r_ce:val_unseen:1"
    assert trace["model"] == "InternVLA-N1-DualVLN"
    assert trace["terminal"]["actor"] == "agent"
    assert trace["terminal"]["status"] == "completed"
    assert trace["metrics"] == {
        "sr": 1.0,
        "spl": 0.9359975622939168,
        "ne": 0.06698722392320633,
        "os": 1.0,
    }
    assert len(trace["actions"]) == trace["event_counts"]["nav.move.discrete"]
    assert set(trace["actions"]) <= {
        "stand_still",
        "forward",
        "turn_left",
        "turn_right",
    }
    assert trace["actions"].count("stand_still") == 7
    assert trace["event_counts"]["nav.goal.finish"] == 1
    assert trace["event_counts"]["nav.stop"] == 1
    assert trace["cleanup_errors"] == []


def test_three_case_run_scope_traces_are_internally_consistent() -> None:
    expected_models = {
        "streamvln-r2r-val-unseen-run-scope-3.json": "StreamVLN",
        "janusvln-r2r-val-unseen-run-scope-3.json": "JanusVLN_Base",
        "dualvln-r2r-val-unseen-run-scope-3.json": "InternVLA-N1-DualVLN",
    }

    for filename, model in expected_models.items():
        trace = json.loads(
            (ROOT / "docs/traces" / filename).read_text(encoding="utf-8")
        )

        assert trace["schema_version"] == 1
        assert trace["model"] == model
        assert trace["validation_scope"] == "harness-real-run"
        assert trace["official_evaluator_comparison"] == "pending"
        assert trace["lifecycle"]["scope"] == "run"
        assert trace["lifecycle"]["tasks"] == 3
        assert trace["lifecycle"]["worker_processes"] == 1
        assert trace["lifecycle"]["observed_worker_pid"] > 0
        assert trace["lifecycle"]["worker_exited_after_run"] is True
        assert [record["case_id"] for record in trace["records"]] == [
            "r2r_ce:val_unseen:1",
            "r2r_ce:val_unseen:2",
            "r2r_ce:val_unseen:3",
        ]
        assert all(
            record["terminal_status"] == "completed" for record in trace["records"]
        )
        assert all(record["cleanup_errors"] == [] for record in trace["records"])
        for record in trace["records"]:
            assert record["event_counts"]["nav.goal.finish"] == 1
            assert record["event_counts"]["nav.stop"] == 1
            assert record["final_pose"]["frame"] == "habitat_episode"
            assert record["metrics"]["sr"] == 1.0
            assert record["metrics"]["os"] == 1.0
            assert record["action_trace"].endswith("S")
            assert set(record["action_trace"]) <= set(trace["action_encoding"])
            assert len(record["action_trace"]) == (
                record["event_counts"]["nav.move.discrete"] + 1
            )

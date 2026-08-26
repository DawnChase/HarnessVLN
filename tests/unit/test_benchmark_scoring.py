from __future__ import annotations

from types import SimpleNamespace

import pytest

from benches import HabitatObjectNavBenchmark, R2RCEBenchmark
from benches.base import BenchmarkCase
from harness.errors import HarnessError
from schemas import NavGoal, NavTask


def case() -> BenchmarkCase:
    goal = NavGoal("goal", "Go to the room.")
    return BenchmarkCase(
        "r2r:fixture",
        NavTask("r2r:fixture", goal),
        truth={"shortest_path_length": 2.0},
    )


def test_r2r_score_prefers_native_habitat_metrics() -> None:
    result = SimpleNamespace(
        environment={
            "success": 1.0,
            "spl": 0.4,
            "distance_to_goal": 0.75,
            "oracle_success": 1.0,
        }
    )

    assert R2RCEBenchmark("unused").score(case(), result) == {
        "sr": 1.0,
        "spl": 0.4,
        "ne": 0.75,
        "os": 1.0,
    }


def test_r2r_score_only_recomputes_spl_with_explicit_path_length() -> None:
    benchmark = R2RCEBenchmark("unused")
    fallback = SimpleNamespace(
        environment={"success": True, "path_length": 4.0, "distance_to_goal": 1.0}
    )
    missing = SimpleNamespace(environment={"success": True})

    assert benchmark.score(case(), fallback)["spl"] == 0.5
    with pytest.raises(HarnessError, match="neither native SPL nor path length"):
        benchmark.score(case(), missing)


def test_habitat_objectnav_score_prefers_native_metrics() -> None:
    result = SimpleNamespace(
        environment={"success": 1.0, "spl": 0.75, "distance_to_goal": 0.05}
    )
    benchmark = HabitatObjectNavBenchmark("unused", dataset="mp3d")

    assert benchmark.score(case(), result) == {
        "sr": 1.0,
        "spl": 0.75,
        "ne": 0.05,
    }

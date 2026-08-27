from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from benches.base import BenchmarkCase, MetricSet
from harness.runtime import NavigationResult
from schemas import NavGoal, NavTask


class DummyBenchmark:
    name = "dummy_navigation"
    validation_status = "contract"

    def __init__(self, cases: Sequence[dict[str, Any]], *, split: str = "smoke") -> None:
        self._definitions = tuple(cases)
        self.split = split

    def cases(self) -> Iterable[BenchmarkCase]:
        for index, definition in enumerate(self._definitions):
            task_id = str(definition.get("task_id", f"dummy-{index}"))
            instruction = str(definition.get("instruction", "navigate to the target"))
            goal = NavGoal(f"{task_id}:goal:0", instruction)
            target = int(definition.get("target", 0))
            yield BenchmarkCase(
                task_id,
                NavTask(task_id, goal, scene_id="dummy_scene"),
                {"targets": [target], "goal_stream": [goal]},
                {"target": target},
            )

    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet:
        reached = result.environment.get("position") == case.truth["target"]
        completed = result.terminal.status == "completed"
        return {"success": float(reached and completed)}

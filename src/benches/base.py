from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from harness.runtime import NavigationResult
from schemas import NavTask


MetricSet: TypeAlias = Mapping[str, float]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    task: NavTask
    setup: Mapping[str, Any] = field(default_factory=dict, repr=False)
    truth: Mapping[str, Any] = field(default_factory=dict, repr=False)


class Benchmark(Protocol):
    name: str
    split: str
    validation_status: str

    def cases(self) -> Iterable[BenchmarkCase]: ...

    def score(self, case: BenchmarkCase, result: NavigationResult) -> MetricSet: ...


def spl(success: bool, shortest_path: float, actual_path: float) -> float:
    if not success:
        return 0.0
    denominator = max(shortest_path, actual_path)
    return float(shortest_path / denominator) if denominator > 0 else 1.0

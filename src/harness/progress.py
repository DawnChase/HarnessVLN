from __future__ import annotations

import sys
from collections.abc import Hashable
from typing import Any, TextIO

from tqdm import tqdm

from benches.base import Benchmark
from harness.errors import HarnessError
from harness.runner import CaseRecord


class RunProgress:
    """Render one parent-owned progress line for all running benchmarks."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._stream = stream or sys.stderr
        self._enabled = _is_terminal(self._stream) if enabled is None else enabled
        self._labels: dict[Hashable, str] = {}
        self._totals: dict[Hashable, int | None] = {}
        self._failed = 0
        self._bar: Any | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def register(
        self,
        key: Hashable,
        benchmark: Benchmark,
        max_cases: int | None,
    ) -> None:
        if not self._enabled:
            return
        label = f"{benchmark.name}/{benchmark.split}"
        try:
            self._labels[key] = label
            self._totals[key] = benchmark_case_total(benchmark, max_cases)
            total = self._combined_total()
            if self._bar is None:
                self._bar = tqdm(
                    total=total,
                    desc=label,
                    unit="episode",
                    dynamic_ncols=True,
                    leave=True,
                    position=0,
                    mininterval=0.2,
                    file=self._stream,
                )
            else:
                self._bar.total = total
                self._bar.refresh()
        except Exception:
            self._disable()

    def advance(self, key: Hashable, record: CaseRecord) -> None:
        if self._bar is None:
            return
        try:
            if _case_failed(record):
                self._failed += 1
            self._bar.set_description_str(
                self._labels.get(key, "HarnessVLN"), refresh=False
            )
            if self._failed:
                self._bar.set_postfix_str(f"failed={self._failed}", refresh=False)
            self._bar.update(1)
        except Exception:
            self._disable()

    def close(self) -> None:
        if self._bar is None:
            return
        try:
            self._bar.close()
        finally:
            self._bar = None

    def _combined_total(self) -> int | None:
        if not self._totals or any(total is None for total in self._totals.values()):
            return None
        return sum(total for total in self._totals.values() if total is not None)

    def _disable(self) -> None:
        try:
            self.close()
        except Exception:
            pass
        self._enabled = False


def benchmark_case_total(
    benchmark: Benchmark,
    max_cases: int | None,
) -> int | None:
    value = getattr(benchmark, "case_count", None)
    if callable(value):
        value = value()
    if value is None:
        return max_cases
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessError("benchmark case_count must be a non-negative integer")
    return min(value, max_cases) if max_cases is not None else value


def _case_failed(record: CaseRecord) -> bool:
    return bool(
        record.error
        or record.output_errors
        or record.task_failed
        or (record.result is not None and record.result.cleanup_errors)
    )


def _is_terminal(stream: TextIO) -> bool:
    try:
        return stream.isatty()
    except (AttributeError, OSError):
        return False

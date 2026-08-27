from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import islice
from typing import Literal

from benches.base import Benchmark, BenchmarkCase, MetricSet
from harness.contracts import NavigationStack
from harness.errors import HarnessError
from harness.runtime import NavigationHarness, NavigationResult
from schemas import EnvironmentEpisode


StackFactory = Callable[[EnvironmentEpisode], NavigationStack]
CaseErrorStage = Literal["stack", "execution", "score"]


@dataclass(frozen=True, slots=True)
class CaseRecord:
    index: int
    case_id: str
    result: NavigationResult | None
    metrics: MetricSet
    error: str | None = None
    error_stage: CaseErrorStage | None = None

    def __post_init__(self) -> None:
        if (self.error is None) != (self.error_stage is None):
            raise ValueError("error and error_stage must be set together")


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    benchmark: str
    split: str
    validation_status: str
    records: tuple[CaseRecord, ...]
    error: str | None = None
    cleanup_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RunSummary:
    benchmarks: tuple[BenchmarkSummary, ...]


class BenchmarkExecutor:
    """Execute one benchmark's whole tasks; never observe, decide, or act."""

    def __init__(self, harness: NavigationHarness) -> None:
        self.harness = harness

    async def run(
        self,
        benchmark: Benchmark,
        stack_factory: StackFactory,
        *,
        parallelism: int = 1,
        max_cases: int | None = None,
    ) -> BenchmarkSummary:
        benchmark_error: BaseException | None = None
        summary: BenchmarkSummary | None = None
        cleanup_errors: list[str] = []
        try:
            if parallelism < 1:
                raise HarnessError("parallelism must be at least 1")
            if max_cases is not None and max_cases < 1:
                raise HarnessError("max_cases must be at least 1")
            if parallelism > 1 and getattr(stack_factory, "requires_serial", False):
                raise HarnessError("this stack requires serial task execution")

            queue: asyncio.Queue[tuple[int, BenchmarkCase] | None] = asyncio.Queue(
                maxsize=parallelism * 2
            )
            records: list[CaseRecord] = []

            async def produce() -> None:
                cases = benchmark.cases()
                selected = cases if max_cases is None else islice(cases, max_cases)
                for index, case in enumerate(selected):
                    await queue.put((index, case))
                for _ in range(parallelism):
                    await queue.put(None)

            async def consume() -> None:
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    index, case = item
                    records.append(
                        await self._execute_case(
                            index, case, benchmark, stack_factory
                        )
                    )

            producer = asyncio.create_task(produce(), name="benchmark-producer")
            workers = [
                asyncio.create_task(consume(), name=f"benchmark-worker-{index}")
                for index in range(parallelism)
            ]
            try:
                await asyncio.gather(producer, *workers)
            except BaseException:
                producer.cancel()
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(producer, *workers, return_exceptions=True)
                raise

            records.sort(key=lambda record: record.index)
            summary = BenchmarkSummary(
                benchmark.name,
                benchmark.split,
                benchmark.validation_status,
                tuple(records),
            )
        except BaseException as error:
            benchmark_error = error
            raise
        finally:
            close_session = getattr(stack_factory, "close_session", None)
            if callable(close_session):
                try:
                    result = close_session()
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except BaseException as cleanup_error:
                    if benchmark_error is None:
                        cleanup_errors.append(_error_text(cleanup_error))
                    else:
                        _attach_cleanup_error(benchmark_error, cleanup_error)
        if summary is None:  # pragma: no cover - exceptions leave via except
            raise AssertionError("benchmark run completed without a summary")
        return replace(summary, cleanup_errors=tuple(cleanup_errors))

    async def _execute_case(
        self,
        index: int,
        case: BenchmarkCase,
        benchmark: Benchmark,
        stack_factory: StackFactory,
    ) -> CaseRecord:
        try:
            stack = stack_factory(case.environment_episode)
        except Exception as error:
            return _failed_case(index, case, "stack", error)

        try:
            result = await self.harness.run_task(case.task, stack)
        except Exception as error:
            return _failed_case(index, case, "execution", error)

        try:
            metrics = benchmark.score(case, result)
        except Exception as error:
            return _failed_case(index, case, "score", error, result=result)
        return CaseRecord(index, case.case_id, result, metrics)


def _attach_cleanup_error(primary: BaseException, cleanup: BaseException) -> None:
    message = _error_text(cleanup)
    previous = getattr(primary, "_harness_cleanup_errors", ())
    try:
        setattr(primary, "_harness_cleanup_errors", (*previous, message))
    except (AttributeError, TypeError):
        pass
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        add_note(f"Benchmark session cleanup failed: {message}")


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _failed_case(
    index: int,
    case: BenchmarkCase,
    stage: CaseErrorStage,
    error: Exception,
    *,
    result: NavigationResult | None = None,
) -> CaseRecord:
    return CaseRecord(
        index=index,
        case_id=case.case_id,
        result=result,
        metrics={},
        error=_error_text(error),
        error_stage=stage,
    )

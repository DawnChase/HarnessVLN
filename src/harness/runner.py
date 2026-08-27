from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from itertools import islice
from numbers import Real
from typing import TYPE_CHECKING, Literal

from benches.base import Benchmark, BenchmarkCase, MetricSet
from harness.contracts import NavigationStack
from harness.errors import HarnessError
from harness.runtime import NavigationHarness, NavigationResult
from schemas import EnvironmentEpisode

if TYPE_CHECKING:
    from harness.output import BenchOutput, EpisodeOutput


StackFactory = Callable[[EnvironmentEpisode], NavigationStack]
CaseCompleted = Callable[["CaseRecord"], None]
CaseErrorStage = Literal["stack", "execution", "score"]


@dataclass(frozen=True, slots=True)
class CaseRecord:
    index: int
    case_id: str
    result: NavigationResult | None
    metrics: MetricSet
    error: str | None = None
    error_stage: CaseErrorStage | None = None
    output_errors: tuple[str, ...] = ()
    resources: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.error is None) != (self.error_stage is None):
            raise ValueError("error and error_stage must be set together")

    @property
    def task_failed(self) -> bool:
        return self.result is not None and self.result.terminal.status not in {
            "completed",
            "environment_terminal",
        }

    def output_record(self) -> dict[str, object]:
        result = self.result
        record = {
            "schema_version": 1,
            "index": self.index,
            "case_id": self.case_id,
            "error": self.error,
            "error_stage": self.error_stage,
            "metrics": dict(self.metrics),
            "execution_id": result.execution_id if result else None,
            "task_id": result.task_id if result else None,
            "terminal": asdict(result.terminal) if result else None,
            "environment_path": "environment.json" if result else None,
            "events_path": "events.jsonl" if result and result.audit else None,
            "cleanup_errors": list(result.cleanup_errors) if result else [],
            "output_errors": list(self.output_errors),
        }
        if self.resources:
            record["resources"] = dict(self.resources)
        return record


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    benchmark: str
    split: str
    validation_status: str
    records: tuple[CaseRecord, ...]
    error: str | None = None
    cleanup_errors: tuple[str, ...] = ()

    def output_record(self) -> dict[str, object]:
        metric_values: dict[str, list[float]] = {}
        for record in self.records:
            for name, value in record.metrics.items():
                metric_values.setdefault(name, []).append(float(value))
        return {
            "name": self.benchmark,
            "split": self.split,
            "validation_status": self.validation_status,
            "error": self.error,
            "cleanup_errors": list(self.cleanup_errors),
            "aggregate_metrics": _aggregate_metrics(metric_values),
            "case_counts": {
                "total": len(self.records),
                "task_failures": sum(record.task_failed for record in self.records),
                "case_errors": sum(record.error is not None for record in self.records),
                "cleanup_errors": len(self.cleanup_errors)
                + sum(
                    len(record.result.cleanup_errors)
                    for record in self.records
                    if record.result is not None
                ),
                "output_errors": sum(len(record.output_errors) for record in self.records),
            },
            "episodes": [
                {
                    "index": record.index,
                    "case_id": record.case_id,
                    "status": (
                        record.result.terminal.status if record.result else "error"
                    ),
                    "error": record.error,
                    "error_stage": record.error_stage,
                    "metrics": dict(record.metrics),
                    "output_errors": list(record.output_errors),
                    **(
                        {"resources": dict(record.resources)}
                        if record.resources
                        else {}
                    ),
                }
                for record in self.records
            ],
        }


@dataclass(frozen=True, slots=True)
class RunSummary:
    benchmarks: tuple[BenchmarkSummary, ...]

    @property
    def aggregate_metrics(self) -> dict[str, float]:
        values: dict[str, list[float]] = {}
        for benchmark in self.benchmarks:
            for record in benchmark.records:
                for name, value in record.metrics.items():
                    values.setdefault(name, []).append(float(value))
        return _aggregate_metrics(values)

    @property
    def failed(self) -> bool:
        return any(
            benchmark.error
            or benchmark.cleanup_errors
            or any(
                record.error
                or record.output_errors
                or record.task_failed
                or (record.result is not None and record.result.cleanup_errors)
                for record in benchmark.records
            )
            for benchmark in self.benchmarks
        )


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
        output: "BenchOutput | None" = None,
        on_case_complete: CaseCompleted | None = None,
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
                    record = await self._execute_case(
                        index, case, benchmark, stack_factory, output
                    )
                    records.append(record)
                    if on_case_complete is not None:
                        on_case_complete(record)

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
        output: "BenchOutput | None",
    ) -> CaseRecord:
        episode_output = (
            output.episode(index, case.case_id, case.output_record())
            if output is not None
            else None
        )
        return await execute_case(
            self.harness,
            index,
            case,
            benchmark,
            stack_factory,
            episode_output,
        )


async def execute_case(
    harness: NavigationHarness,
    index: int,
    case: BenchmarkCase,
    benchmark: Benchmark,
    stack_factory: StackFactory,
    output: "EpisodeOutput | None" = None,
    *,
    resources: Mapping[str, object] | None = None,
) -> CaseRecord:
    resource_record = dict(resources or {})
    try:
        stack = stack_factory(case.environment_episode)
    except Exception as error:
        return _finish_case_output(
            _failed_case(
                index, case, "stack", error, resources=resource_record
            ),
            output,
        )

    try:
        if output is None:
            result = await harness.run_task(case.task, stack)
        else:
            result = await harness.run_task(case.task, stack, output=output)
    except Exception as error:
        return _finish_case_output(
            _failed_case(
                index, case, "execution", error, resources=resource_record
            ),
            output,
        )

    try:
        metrics = _validate_metrics(benchmark.score(case, result))
    except Exception as error:
        return _finish_case_output(
            _failed_case(
                index,
                case,
                "score",
                error,
                result=result,
                resources=resource_record,
            ),
            output,
        )
    return _finish_case_output(
        CaseRecord(
            index,
            case.case_id,
            result,
            metrics,
            resources=resource_record,
        ),
        output,
    )


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


def _validate_metrics(metrics: object) -> dict[str, float]:
    if not isinstance(metrics, Mapping) or not metrics:
        raise HarnessError("benchmark score must return a non-empty metric mapping")
    normalized: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise HarnessError("benchmark metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, Real):
            raise HarnessError(f"benchmark metric {name!r} must be a real number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise HarnessError(f"benchmark metric {name!r} must be finite")
        normalized[name] = numeric
    return normalized


def _aggregate_metrics(values: Mapping[str, list[float]]) -> dict[str, float]:
    return {
        name: sum(items) / len(items)
        for name, items in values.items()
        if items
    }


def _finish_case_output(
    record: CaseRecord, output: "EpisodeOutput | None"
) -> CaseRecord:
    if output is None:
        return record
    errors = output.finish(record.output_record())
    return replace(record, output_errors=errors)


def _failed_case(
    index: int,
    case: BenchmarkCase,
    stage: CaseErrorStage,
    error: Exception,
    *,
    result: NavigationResult | None = None,
    resources: Mapping[str, object] | None = None,
) -> CaseRecord:
    return CaseRecord(
        index=index,
        case_id=case.case_id,
        result=result,
        metrics={},
        error=_error_text(error),
        error_stage=stage,
        resources=dict(resources or {}),
    )

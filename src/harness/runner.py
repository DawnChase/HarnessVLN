from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice

from benches.base import Benchmark, BenchmarkCase, MetricSet
from harness.contracts import NavigationStack
from harness.errors import HarnessError
from harness.runtime import NavigationHarness, NavigationResult


StackFactory = Callable[[BenchmarkCase], NavigationStack]


@dataclass(frozen=True, slots=True)
class CaseRecord:
    index: int
    case_id: str
    result: NavigationResult | None
    metrics: MetricSet
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RunSummary:
    benchmark: str
    split: str
    validation_status: str
    records: tuple[CaseRecord, ...]


class BenchRunner:
    """Bounded parallel scheduling of whole tasks; never observes or acts."""

    def __init__(self, harness: NavigationHarness) -> None:
        self.harness = harness

    async def run(
        self,
        benchmark: Benchmark,
        stack_factory: StackFactory,
        *,
        parallelism: int = 1,
        max_cases: int | None = None,
    ) -> RunSummary:
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
                try:
                    stack = stack_factory(case)
                    result = await self.harness.run_task(case.task, stack)
                    metrics = benchmark.score(case, result)
                    record = CaseRecord(index, case.case_id, result, metrics)
                except Exception as error:
                    record = CaseRecord(
                        index,
                        case.case_id,
                        None,
                        {},
                        f"{type(error).__name__}: {error}",
                    )
                records.append(record)

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
        return RunSummary(
            benchmark.name,
            benchmark.split,
            benchmark.validation_status,
            tuple(records),
        )

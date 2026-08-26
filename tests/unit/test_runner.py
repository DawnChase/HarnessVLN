from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from benches.base import BenchmarkCase
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.errors import HarnessError
from harness.runner import BenchRunner
from schemas import NavGoal, NavTask


class CasesBenchmark:
    name = "cases"
    split = "test"
    validation_status = "contract"

    def __init__(self, count: int, cases: Iterable[BenchmarkCase] | None = None):
        self.count = count
        self._cases = cases

    def cases(self):
        if self._cases is not None:
            yield from self._cases
            return
        for index in range(self.count):
            goal = NavGoal(f"goal-{index}", "wait")
            yield BenchmarkCase(str(index), NavTask(str(index), goal))

    def score(self, case, result):
        del case
        return {"completed": float(result.terminal.status == "completed")}


class StopAgent:
    required_tools: frozenset[str] = frozenset()

    async def run(self, context):
        await context.nav.stop("completed")


def stack_for(case: BenchmarkCase, agent=None):
    return NavigationStack(
        agent or StopAgent(),
        DummyNavigationEnvironment((case.task.goal,), targets=(0,)),
    )


def test_runner_bounds_whole_task_concurrency_and_preserves_order() -> None:
    class State:
        active = 0
        maximum = 0
        release = asyncio.Event()
        full = asyncio.Event()

    class BlockingAgent:
        required_tools: frozenset[str] = frozenset()

        async def run(self, context):
            State.active += 1
            State.maximum = max(State.maximum, State.active)
            if State.maximum == 3:
                State.full.set()
                State.release.set()
            await State.release.wait()
            State.active -= 1
            await context.nav.stop("completed")

    async def scenario():
        summary = await BenchRunner(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(9),
            lambda case: stack_for(case, BlockingAgent()),
            parallelism=3,
        )
        assert State.maximum == 3
        assert [record.case_id for record in summary.records] == [str(i) for i in range(9)]
        assert all(record.metrics == {"completed": 1.0} for record in summary.records)

    asyncio.run(scenario())


def test_runner_streams_cases_and_does_not_eagerly_consume_split() -> None:
    class State:
        produced = 0
        active = 0
        ready = asyncio.Event()
        release = asyncio.Event()

    def cases():
        for index in range(100):
            State.produced += 1
            goal = NavGoal(f"goal-{index}", "wait")
            yield BenchmarkCase(str(index), NavTask(str(index), goal))

    class BlockingAgent:
        required_tools: frozenset[str] = frozenset()

        async def run(self, context):
            State.active += 1
            if State.active == 2:
                State.ready.set()
            await State.release.wait()
            await context.nav.stop("completed")

    async def scenario():
        execution = asyncio.create_task(
            BenchRunner(NavigationHarness(timeout_s=2)).run(
                CasesBenchmark(100, cases()),
                lambda case: stack_for(case, BlockingAgent()),
                parallelism=2,
            )
        )
        await State.ready.wait()
        await asyncio.sleep(0)
        assert State.produced < 100
        assert State.produced <= 7
        State.release.set()
        summary = await execution
        assert len(summary.records) == 100

    asyncio.run(scenario())


def test_single_case_factory_failure_does_not_cancel_siblings() -> None:
    def factory(case):
        if case.case_id == "1":
            raise ValueError("bad fixture")
        return stack_for(case)

    async def scenario():
        summary = await BenchRunner(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(3), factory, parallelism=2
        )
        assert summary.records[0].result is not None
        assert summary.records[1].result is None
        assert summary.records[1].error == "ValueError: bad fixture"
        assert summary.records[2].result is not None

    asyncio.run(scenario())


def test_shared_writeback_stack_rejects_parallel_execution() -> None:
    class SerialFactory:
        requires_serial = True

        def __call__(self, case):
            return stack_for(case)

    async def scenario():
        with pytest.raises(HarnessError, match="requires serial"):
            await BenchRunner(NavigationHarness()).run(
                CasesBenchmark(1), SerialFactory(), parallelism=2
            )

    asyncio.run(scenario())

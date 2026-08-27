from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from benches.base import BenchmarkCase
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.errors import HarnessError
from harness.runner import BenchmarkExecutor
from schemas import EnvironmentEpisode, NavGoal, NavTask


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


def stack_for(episode: EnvironmentEpisode, agent=None):
    return NavigationStack(
        agent or StopAgent(),
        DummyNavigationEnvironment((episode.task.goal,), targets=(0,)),
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
        summary = await BenchmarkExecutor(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(9),
            lambda episode: stack_for(episode, BlockingAgent()),
            parallelism=3,
        )
        assert State.maximum == 3
        assert [record.case_id for record in summary.records] == [
            str(i) for i in range(9)
        ]
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
            BenchmarkExecutor(NavigationHarness(timeout_s=2)).run(
                CasesBenchmark(100, cases()),
                lambda episode: stack_for(episode, BlockingAgent()),
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


def test_runner_can_bound_a_smoke_run_without_consuming_an_extra_case() -> None:
    class State:
        produced = 0

    def cases():
        for index in range(10):
            State.produced += 1
            goal = NavGoal(f"goal-{index}", "stop")
            yield BenchmarkCase(str(index), NavTask(str(index), goal))

    async def scenario():
        summary = await BenchmarkExecutor(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(10, cases()),
            stack_for,
            parallelism=2,
            max_cases=3,
        )
        assert [record.case_id for record in summary.records] == ["0", "1", "2"]
        assert State.produced == 3

    asyncio.run(scenario())


def test_single_case_factory_failure_does_not_cancel_siblings() -> None:
    def factory(episode):
        if episode.task.task_id == "1":
            raise ValueError("bad fixture")
        return stack_for(episode)

    async def scenario():
        summary = await BenchmarkExecutor(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(3), factory, parallelism=2
        )
        assert summary.records[0].result is not None
        assert summary.records[1].result is None
        assert summary.records[1].error == "ValueError: bad fixture"
        assert summary.records[2].result is not None

    asyncio.run(scenario())


def test_runner_only_exposes_environment_episode_to_stack_factory() -> None:
    goal = NavGoal("goal", "stop")
    case = BenchmarkCase(
        "case",
        NavTask("case", goal),
        {"environment_value": "visible"},
        {"evaluation_secret": "hidden"},
    )
    received = []

    def factory(episode: EnvironmentEpisode):
        received.append(episode)
        return stack_for(episode)

    summary = asyncio.run(
        BenchmarkExecutor(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(1, (case,)), factory
        )
    )

    assert len(summary.records) == 1
    assert received == [case.environment_episode]
    assert received[0].setup == {"environment_value": "visible"}
    assert not hasattr(received[0], "truth")


def test_shared_writeback_stack_rejects_parallel_execution() -> None:
    class SerialFactory:
        requires_serial = True

        def __call__(self, case):
            return stack_for(case)

    async def scenario():
        with pytest.raises(HarnessError, match="requires serial"):
            await BenchmarkExecutor(NavigationHarness()).run(
                CasesBenchmark(1), SerialFactory(), parallelism=2
            )

    asyncio.run(scenario())


def test_executor_closes_session_scoped_factory_after_all_cases() -> None:
    class ClosableFactory:
        requires_serial = False

        def __init__(self):
            self.closed = False

        def __call__(self, case):
            return stack_for(case)

        async def close_session(self):
            self.closed = True

    async def scenario():
        factory = ClosableFactory()
        summary = await BenchmarkExecutor(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(2), factory, parallelism=1
        )

        assert len(summary.records) == 2
        assert factory.closed

    asyncio.run(scenario())


def test_runner_does_not_reflect_an_unrelated_factory_close_method() -> None:
    class Factory:
        def __init__(self):
            self.close_calls = 0

        def __call__(self, case):
            return stack_for(case)

        def close(self, required_argument):
            del required_argument
            self.close_calls += 1

    async def scenario():
        factory = Factory()
        await BenchmarkExecutor(NavigationHarness(timeout_s=1)).run(
            CasesBenchmark(1), factory
        )
        assert factory.close_calls == 0

    asyncio.run(scenario())


def test_executor_closes_session_scope_when_validation_fails() -> None:
    class SerialFactory:
        requires_serial = True

        def __init__(self):
            self.close_calls = 0

        def __call__(self, case):
            return stack_for(case)

        async def close_session(self):
            self.close_calls += 1

    async def scenario():
        factory = SerialFactory()
        with pytest.raises(HarnessError, match="requires serial"):
            await BenchmarkExecutor(NavigationHarness()).run(
                CasesBenchmark(1), factory, parallelism=2
            )
        assert factory.close_calls == 1

    asyncio.run(scenario())


def test_run_cleanup_error_after_success_is_reported_in_summary() -> None:
    class Factory:
        def __call__(self, episode):
            return stack_for(episode)

        async def close_session(self):
            raise RuntimeError("cleanup failed")

    async def scenario():
        summary = await BenchmarkExecutor(NavigationHarness()).run(
            CasesBenchmark(1), Factory()
        )

        assert len(summary.records) == 1
        assert summary.error is None
        assert summary.cleanup_errors == ("RuntimeError: cleanup failed",)

    asyncio.run(scenario())


def test_run_cleanup_error_does_not_mask_primary_runner_error() -> None:
    class BrokenBenchmark(CasesBenchmark):
        def cases(self):
            raise RuntimeError("case generation failed")
            yield

    class Factory:
        def __call__(self, case):
            return stack_for(case)

        async def close_session(self):
            raise RuntimeError("cleanup failed")

    async def scenario():
        with pytest.raises(RuntimeError, match="case generation failed") as caught:
            await BenchmarkExecutor(NavigationHarness()).run(
                BrokenBenchmark(0), Factory()
            )
        assert caught.value._harness_cleanup_errors == ("RuntimeError: cleanup failed",)

    asyncio.run(scenario())

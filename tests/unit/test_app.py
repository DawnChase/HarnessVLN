from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from harness.app import (
    ConfiguredStackFactory,
    InteractiveNavigationSession,
    _stack_factory,
    run_config,
)
from harness.config import (
    ComponentSpec,
    load_agent_config,
    load_environment_config,
)
from harness.errors import HarnessError
from harness.runtime import NavigationHarness


def spec(name: str, *, serial: bool = False) -> ComponentSpec:
    return ComponentSpec(name, {}, serial)


class GateHarness:
    def __init__(self) -> None:
        self._inner = NavigationHarness(timeout_s=1)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.active = 0
        self.maximum = 0

    async def run_task(self, task, stack):
        self.calls += 1
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.entered.set()
        try:
            await self.release.wait()
            return await self._inner.run_task(task, stack)
        finally:
            self.active -= 1


def test_stack_factory_propagates_component_serial_constraint() -> None:
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.isaac_vln_pe:from_episode", serial=True),
    )

    assert factory.requires_serial is True
    assert factory.global_serial_reasons == ("environment.serial",)


def test_stack_factory_allows_parallel_read_only_components() -> None:
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.dummy:from_episode"),
    )

    assert factory.requires_serial is False
    assert factory.global_serial_reasons == ()


def test_agent_and_memory_fragments_create_independent_plugins() -> None:
    factory = _stack_factory(
        load_agent_config("config/agents/normal_agent.yaml"),
        load_environment_config("config/envs/dummy.yaml"),
    )
    stack = factory(next(iter(factory_case_source().cases())).environment_episode)

    assert type(stack.agent).__name__ == "NormalAgent"
    assert type(stack.memory).__name__ == "DummyLandmarkMemory"
    assert factory.requires_serial is True
    assert factory.global_serial_reasons == (
        "memory.serial",
        "memory.writeback",
    )
    assert "spatial.search" in stack.agent.required_tools


def test_interactive_session_turns_each_instruction_into_an_agent_owned_task() -> None:
    async def scenario():
        session = InteractiveNavigationSession.from_configs(
            "config/envs/dummy.yaml", "config/agents/passthrough.yaml"
        )

        first = await session.navigate("Stay at the marker.")
        second = await session.navigate("Confirm the marker again.")
        await session.close()
        await session.close()

        assert first.task_id != second.task_id
        assert first.terminal.status == "completed"
        assert first.terminal.actor == "agent"
        assert [event.name for event in first.audit][-2:] == [
            "nav.goal.finish",
            "nav.stop",
        ]
        with pytest.raises(HarnessError, match="session is closed"):
            await session.navigate("Run after close.")

    asyncio.run(scenario())


def test_interactive_session_serializes_concurrent_instructions() -> None:
    async def scenario():
        session = InteractiveNavigationSession.from_configs(
            "config/envs/dummy.yaml", "config/agents/passthrough.yaml"
        )
        harness = GateHarness()
        session.harness = harness

        first = asyncio.create_task(session.navigate("First instruction."))
        await harness.entered.wait()
        second = asyncio.create_task(session.navigate("Second instruction."))
        await asyncio.sleep(0)

        assert harness.calls == 1
        harness.release.set()
        results = await asyncio.gather(first, second)
        await session.close()

        assert harness.calls == 2
        assert harness.maximum == 1
        assert results[0].task_id != results[1].task_id

    asyncio.run(scenario())


def test_interactive_close_waits_for_active_navigation() -> None:
    async def scenario():
        session = InteractiveNavigationSession.from_configs(
            "config/envs/dummy.yaml", "config/agents/passthrough.yaml"
        )
        harness = GateHarness()
        session.harness = harness

        navigation = asyncio.create_task(session.navigate("Active instruction."))
        await harness.entered.wait()
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)

        assert not closing.done()
        harness.release.set()
        result = await navigation
        await closing

        assert result.terminal.status == "completed"
        with pytest.raises(HarnessError, match="session is closed"):
            await session.navigate("After close.")

    asyncio.run(scenario())


def test_interactive_close_can_retry_after_factory_error() -> None:
    class FlakyFactory:
        def __init__(self) -> None:
            self.calls = 0

        async def close_run(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("close failed")

    async def scenario():
        session = InteractiveNavigationSession.from_configs(
            "config/envs/dummy.yaml", "config/agents/passthrough.yaml"
        )
        factory = FlakyFactory()
        session._factory = factory

        with pytest.raises(RuntimeError, match="close failed"):
            await session.close()
        assert session._closed is False

        await session.close()
        await session.close()
        assert factory.calls == 2
        assert session._closed is True

    asyncio.run(scenario())


def test_run_scoped_vln_is_shared_and_forces_serial_tasks(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("fixture")
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.dummy:from_episode"),
        vln=ComponentSpec(
            "vln.rpc:RPCVLNNavigator",
            {
                "command": [sys.executable, "worker.py"],
                "upstream_root": str(tmp_path),
                "checkpoint": str(checkpoint),
            },
            scope="run",
        ),
    )
    cases = list(factory_case_source().cases())
    first = factory(cases[0].environment_episode)
    second = factory(cases[0].environment_episode)

    assert first.vln is second.vln
    assert factory.requires_serial
    assert factory.global_serial_reasons == ()
    asyncio.run(factory.close_run())
    third = factory(cases[0].environment_episode)
    assert third.vln is not first.vln
    asyncio.run(factory.close_run())


def test_runner_rejects_cross_bench_global_resource_race(tmp_path) -> None:
    bench_config = Path("config/benches/dummy.yaml").resolve()
    runner = tmp_path / "runner.yaml"
    runner.write_text(
        "runner:\n"
        "  benches:\n"
        f"    - {bench_config}\n"
        f"    - {bench_config}\n"
        "  bench_parallelism: 2\n"
        "  task_parallelism: 1\n"
    )

    with pytest.raises(HarnessError, match="memory.writeback"):
        asyncio.run(run_config(runner, "config/agents/normal_agent.yaml"))


def test_run_scope_is_rejected_for_task_owned_components() -> None:
    with pytest.raises(HarnessError, match="run-scoped agent"):
        ConfiguredStackFactory(
            agent=ComponentSpec(
                "agents.passthrough:PassthroughVLNAgent", {}, scope="run"
            ),
            environment=spec("envs.dummy:from_episode"),
        )


def test_run_scoped_factory_closes_once_for_concurrent_callers() -> None:
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class Navigator:
            close_calls = 0

            async def close_run(self):
                self.close_calls += 1
                entered.set()
                await release.wait()

        factory = ConfiguredStackFactory(
            agent=spec("agents.passthrough:PassthroughVLNAgent"),
            environment=spec("envs.dummy:from_episode"),
        )
        navigator = Navigator()
        factory._run_vln = navigator
        first = asyncio.create_task(factory.close_run())
        await entered.wait()
        second = asyncio.create_task(factory.close_run())
        release.set()
        await asyncio.gather(first, second)

        assert navigator.close_calls == 1
        assert factory._run_vln is None
        assert factory._close_task is None
        await factory.close_run()
        assert navigator.close_calls == 1

    asyncio.run(scenario())


def test_run_scoped_factory_retains_handle_when_close_fails() -> None:
    class Navigator:
        close_calls = 0

        async def close_run(self):
            self.close_calls += 1
            raise RuntimeError("close failed")

    async def scenario():
        factory = ConfiguredStackFactory(
            agent=spec("agents.passthrough:PassthroughVLNAgent"),
            environment=spec("envs.dummy:from_episode"),
        )
        navigator = Navigator()
        factory._run_vln = navigator

        with pytest.raises(RuntimeError, match="close failed"):
            await factory.close_run()
        assert factory._run_vln is navigator
        assert factory._close_task is not None
        with pytest.raises(HarnessError, match="close is still"):
            factory(next(iter(factory_case_source().cases())).environment_episode)
        with pytest.raises(RuntimeError, match="close failed"):
            await factory.close_run()
        assert navigator.close_calls == 2

    asyncio.run(scenario())


def test_run_scoped_factory_preserves_cancellation_when_close_fails() -> None:
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class Navigator:
            async def close_run(self):
                entered.set()
                await release.wait()
                raise RuntimeError("cleanup failed")

        factory = ConfiguredStackFactory(
            agent=spec("agents.passthrough:PassthroughVLNAgent"),
            environment=spec("envs.dummy:from_episode"),
        )
        factory._run_vln = Navigator()
        closing = asyncio.create_task(factory.close_run())
        await entered.wait()
        closing.cancel()
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await closing
        assert factory._close_task is not None
        assert isinstance(factory._close_task.exception(), RuntimeError)
        assert factory._run_vln is not None

    asyncio.run(scenario())


def factory_case_source():
    from benches.dummy import DummyBenchmark

    return DummyBenchmark(
        split="fixture",
        cases=[{"task_id": "case", "instruction": "Go.", "target": 0}],
    )

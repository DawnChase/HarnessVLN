from __future__ import annotations

import asyncio
import sys

import pytest

from harness.app import ConfiguredStackFactory, _stack_factory
from harness.config import ComponentSpec
from harness.errors import HarnessError


def spec(name: str, *, serial: bool = False) -> ComponentSpec:
    return ComponentSpec(name, {}, serial)


def test_stack_factory_propagates_component_serial_constraint() -> None:
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.isaac_vln_pe:from_case", serial=True),
    )

    assert factory.requires_serial is True


def test_stack_factory_allows_parallel_read_only_components() -> None:
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.dummy:from_case"),
    )

    assert factory.requires_serial is False


def test_agent_and_memory_fragments_create_independent_plugins() -> None:
    from harness.config import load_config

    resolved = load_config(
        (
            "config/benches/dummy.yaml",
            "config/agents/normal_agent.yaml",
            "config/envs/dummy.yaml",
            "config/vln/dummy.yaml",
            "config/memory/dummy_landmark.yaml",
            "config/runs/dummy_normal_agent.yaml",
        )
    )
    factory = _stack_factory(resolved.data["stack"])
    stack = factory(next(iter(factory_case_source().cases())))

    assert type(stack.agent).__name__ == "NormalAgent"
    assert type(stack.memory).__name__ == "DummyLandmarkMemory"
    assert factory.requires_serial is True
    assert "spatial.search" in stack.agent.required_tools


def test_run_scoped_vln_is_shared_and_forces_serial_tasks(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("fixture")
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.dummy:from_case"),
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
    first = factory(cases[0])
    second = factory(cases[0])

    assert first.vln is second.vln
    assert factory.requires_serial
    asyncio.run(factory.close_run())
    third = factory(cases[0])
    assert third.vln is not first.vln
    asyncio.run(factory.close_run())


def test_run_scope_is_rejected_for_task_owned_components() -> None:
    with pytest.raises(HarnessError, match="run-scoped agent"):
        ConfiguredStackFactory(
            agent=ComponentSpec(
                "agents.passthrough:PassthroughVLNAgent", {}, scope="run"
            ),
            environment=spec("envs.dummy:from_case"),
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
            environment=spec("envs.dummy:from_case"),
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
            environment=spec("envs.dummy:from_case"),
        )
        navigator = Navigator()
        factory._run_vln = navigator

        with pytest.raises(RuntimeError, match="close failed"):
            await factory.close_run()
        assert factory._run_vln is navigator
        assert factory._close_task is not None
        with pytest.raises(HarnessError, match="close is still"):
            factory(next(iter(factory_case_source().cases())))
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
            environment=spec("envs.dummy:from_case"),
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

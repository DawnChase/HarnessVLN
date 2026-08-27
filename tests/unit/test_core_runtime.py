from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from harness.contracts import NavigationStack
from harness.errors import HarnessError, MissingToolError, ToolClosedError, ToolValidationError
from harness.runtime import NavigationHarness
from harness.tool_bus import Tool, ToolBus
from schemas import (
    EnvironmentTerminal,
    MotionProfile,
    NavGoal,
    NavigationProfile,
    NavTask,
)


def run(coroutine):
    return asyncio.run(coroutine)


def task() -> NavTask:
    return NavTask("task-1", NavGoal("goal-1", "move forward"), scene_id="scene")


class FakeEnvironment:
    profile = NavigationProfile(
        observation_channels=frozenset({"position"}),
        motion=MotionProfile(
            "nav.move.discrete", frozenset({"forward"}), frame="fixture"
        ),
    )

    def __init__(self) -> None:
        self.position = 0
        self.stopped = False
        self.failure: asyncio.Future[str] | None = None

    async def start(self, nav_task: NavTask) -> Sequence[Tool]:
        del nav_task
        self.failure = asyncio.get_running_loop().create_future()

        async def observe(actor, arguments):
            del actor, arguments
            return {"position": self.position}

        async def move(actor, arguments):
            del actor
            self.position += 1 if arguments["action"] == "forward" else 0
            return {"position": self.position}

        return (
            Tool(
                "nav.observe",
                "Observe.",
                {"type": "object", "additionalProperties": False},
                observe,
            ),
            Tool(
                "nav.move.discrete",
                "Move.",
                {
                    "type": "object",
                    "properties": {"action": {"enum": ["forward"]}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                move,
                writes=True,
            ),
        )

    async def stop(self, reason: str) -> None:
        del reason
        self.stopped = True

    async def wait_terminal(self) -> EnvironmentTerminal:
        assert self.failure is not None
        reason = await self.failure
        return EnvironmentTerminal("failed", reason)

    def result(self):
        return {"position": self.position, "stopped": self.stopped}


class LoopAgent:
    required_tools = frozenset({"nav.observe", "nav.move.discrete"})

    def __init__(self) -> None:
        self.run_calls = 0

    async def run(self, context) -> None:
        self.run_calls += 1
        while (await context.nav.observe())["position"] < 3:
            await context.nav.move_discrete("forward")
        await context.nav.stop("success", "reached")


def test_navigation_stack_rejects_invalid_agent_before_environment_start() -> None:
    class InvalidAgent:
        required_tools = frozenset()

    environment = FakeEnvironment()

    with pytest.raises(HarnessError, match="agent InvalidAgent.*callable methods: run"):
        NavigationStack(InvalidAgent(), environment)

    assert environment.failure is None


def test_navigation_stack_requires_an_environment_profile() -> None:
    class InvalidEnvironment(FakeEnvironment):
        profile = None

    with pytest.raises(HarnessError, match="must declare a NavigationProfile"):
        NavigationStack(LoopAgent(), InvalidEnvironment())


def test_navigation_stack_validates_vln_tool_ownership() -> None:
    class InvalidVLN:
        required_tools = frozenset({"nav.stop"})
        requirements = {}

        async def start(self, nav_task, tools):
            del nav_task, tools
            return ()

        async def stop(self, reason):
            del reason

    with pytest.raises(HarnessError, match="cannot require agent-owned nav.stop"):
        NavigationStack(LoopAgent(), FakeEnvironment(), vln=InvalidVLN())


def test_navigation_stack_requires_immutable_tool_declarations() -> None:
    class InvalidMemory:
        required_tools = {"nav.observe"}

        async def start(self, nav_task, tools):
            del nav_task, tools
            return ()

        async def stop(self, reason):
            del reason

    with pytest.raises(HarnessError, match="required_tools must be a frozenset"):
        NavigationStack(LoopAgent(), FakeEnvironment(), memory=InvalidMemory())


def test_agent_owns_complete_navigation_loop() -> None:
    async def scenario():
        agent = LoopAgent()
        environment = FakeEnvironment()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(agent, environment)
        )
        assert agent.run_calls == 1
        assert result.terminal.status == "success"
        assert result.terminal.actor == "agent"
        assert result.environment == {"position": 3, "stopped": True}
        assert [event.name for event in result.audit].count("nav.observe") == 4

    run(scenario())


def test_agent_return_without_stop_is_failure() -> None:
    class ReturningAgent:
        required_tools = frozenset()

        async def run(self, context):
            del context

    async def scenario():
        environment = FakeEnvironment()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(ReturningAgent(), environment)
        )
        assert result.terminal.status == "failed"
        assert "without calling nav.stop" in result.terminal.reason
        assert environment.stopped

    run(scenario())


def test_missing_agent_tool_fails_before_agent_runs() -> None:
    class InvalidAgent:
        required_tools = frozenset({"nav.move.velocity"})

        def __init__(self):
            self.run_calls = 0

        async def run(self, context):
            self.run_calls += 1

    async def scenario():
        agent = InvalidAgent()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(agent, FakeEnvironment())
        )
        assert agent.run_calls == 0
        assert result.terminal.status == "failed"
        assert "nav.move.velocity" in result.terminal.reason

    run(scenario())


def test_environment_is_stopped_when_its_start_partially_fails() -> None:
    class PartialEnvironment(FakeEnvironment):
        async def start(self, nav_task):
            del nav_task
            self.failure = asyncio.get_running_loop().create_future()
            raise RuntimeError("environment start failed")

    class UnusedAgent:
        required_tools = frozenset()

        def __init__(self):
            self.ran = False

        async def run(self, context):
            del context
            self.ran = True

    async def scenario():
        agent = UnusedAgent()
        environment = PartialEnvironment()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(agent, environment)
        )

        assert result.terminal.status == "failed"
        assert "environment start failed" in result.terminal.reason
        assert environment.stopped
        assert agent.ran is False
        assert result.cleanup_errors == ()

    run(scenario())


def test_memory_is_stopped_when_its_start_partially_fails() -> None:
    class PartialMemory:
        required_tools = frozenset()

        def __init__(self):
            self.stopped = False

        async def start(self, nav_task, tools):
            del nav_task, tools
            raise RuntimeError("memory start failed")

        async def stop(self, reason):
            del reason
            self.stopped = True

    class UnusedAgent:
        required_tools = frozenset()

        async def run(self, context):
            raise AssertionError(f"agent unexpectedly ran: {context}")

    async def scenario():
        environment = FakeEnvironment()
        memory = PartialMemory()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(UnusedAgent(), environment, memory=memory)
        )

        assert result.terminal.status == "failed"
        assert "memory start failed" in result.terminal.reason
        assert memory.stopped
        assert environment.stopped
        assert result.cleanup_errors == ()

    run(scenario())


def test_vln_is_stopped_when_its_start_partially_fails() -> None:
    class PartialVLN:
        required_tools = frozenset()
        requirements = {}

        def __init__(self):
            self.stopped = False

        async def start(self, nav_task, tools):
            del nav_task, tools
            raise RuntimeError("vln start failed")

        async def stop(self, reason):
            del reason
            self.stopped = True

    class UnusedAgent:
        required_tools = frozenset()

        async def run(self, context):
            raise AssertionError(f"agent unexpectedly ran: {context}")

    async def scenario():
        environment = FakeEnvironment()
        navigator = PartialVLN()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(UnusedAgent(), environment, vln=navigator)
        )

        assert result.terminal.status == "failed"
        assert "vln start failed" in result.terminal.reason
        assert navigator.stopped
        assert environment.stopped
        assert result.cleanup_errors == ()

    run(scenario())


def test_tool_bus_validates_schema_permissions_and_write_fence() -> None:
    async def scenario():
        bus = ToolBus()

        async def handler(actor, arguments):
            return {"actor": actor, "value": arguments["value"]}

        bus.register(
            (
                Tool(
                    "nav.write",
                    "Write.",
                    {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    handler,
                    writes=True,
                ),
            )
        )
        client = bus.client("agent", frozenset({"nav.write"}))
        assert [(spec.name, spec.description) for spec in client.specs] == [
            ("nav.write", "Write.")
        ]
        assert client.specs[0].input_schema["required"] == ["value"]
        assert await client.call("nav.write", value=1) == {"actor": "agent", "value": 1}
        with pytest.raises(ToolValidationError):
            await client.call("nav.write", value="bad")
        with pytest.raises(MissingToolError):
            await bus.client("vln", frozenset()).call("nav.write", value=2)
        bus.close_writes()
        with pytest.raises(ToolClosedError):
            await client.call("nav.write", value=3)

    run(scenario())


def test_external_cancellation_cleans_environment_then_propagates() -> None:
    class HangingAgent:
        required_tools = frozenset()

        def __init__(self):
            self.started = asyncio.Event()

        async def run(self, context):
            self.started.set()
            await context.cancelled.wait()
            await asyncio.Event().wait()

    async def scenario():
        environment = FakeEnvironment()
        agent = HangingAgent()
        execution = asyncio.create_task(
            NavigationHarness(timeout_s=10).run_task(
                task(), NavigationStack(agent, environment)
            )
        )
        await agent.started.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
        assert environment.stopped

    run(scenario())


def test_stop_acknowledgement_closes_motion_tools() -> None:
    class StopThenMoveAgent:
        required_tools = frozenset({"nav.move.discrete"})

        def __init__(self):
            self.move_rejected = False

        async def run(self, context):
            await context.nav.stop("success", "done")
            with pytest.raises(ToolClosedError):
                await context.nav.move_discrete("forward")
            self.move_rejected = True

    async def scenario():
        environment = FakeEnvironment()
        agent = StopThenMoveAgent()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(agent, environment)
        )
        assert result.terminal.status == "success"
        assert result.environment["position"] == 0
        assert agent.move_rejected

    run(scenario())


def test_cleanup_is_best_effort_and_preserves_terminal() -> None:
    class FailingStopEnvironment(FakeEnvironment):
        async def stop(self, reason: str) -> None:
            await super().stop(reason)
            raise RuntimeError("native close failed")

    class TrackingMemory:
        required_tools = frozenset()

        def __init__(self):
            self.stopped = False

        async def start(self, nav_task, tools):
            del nav_task, tools
            return ()

        async def stop(self, reason):
            del reason
            self.stopped = True

    class StoppingAgent:
        required_tools = frozenset()

        async def run(self, context):
            await context.nav.stop("success", "done")

    async def scenario():
        memory = TrackingMemory()
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(StoppingAgent(), FailingStopEnvironment(), memory=memory)
        )
        assert result.terminal.status == "success"
        assert memory.stopped
        assert result.cleanup_errors == (
            "environment: RuntimeError: native close failed",
        )

    run(scenario())


def test_stop_ack_fences_a_write_already_admitted_by_tool_bus() -> None:
    class FencedEnvironment(FakeEnvironment):
        def __init__(self):
            super().__init__()
            self.admitted = asyncio.Event()
            self.release = asyncio.Event()
            self.generation = 0

        async def start(self, nav_task):
            await super().start(nav_task)

            async def move(actor, arguments):
                del actor, arguments
                generation = self.generation
                self.admitted.set()
                await self.release.wait()
                if self.stopped or generation != self.generation:
                    raise ToolClosedError("native motion fence changed")
                self.position += 1
                return {"position": self.position}

            return (
                Tool(
                    "nav.move.discrete",
                    "Move.",
                    {
                        "type": "object",
                        "properties": {"action": {"enum": ["forward"]}},
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                    move,
                    writes=True,
                ),
            )

        async def stop(self, reason):
            del reason
            self.stopped = True
            self.generation += 1
            self.release.set()

    class RacingAgent:
        required_tools = frozenset({"nav.move.discrete"})

        def __init__(self, environment):
            self.environment = environment
            self.rejected = False

        async def run(self, context):
            move = asyncio.create_task(context.nav.move_discrete("forward"))
            await self.environment.admitted.wait()
            await context.nav.stop("success", "stop while move is queued")
            with pytest.raises(ToolClosedError):
                await move
            self.rejected = True

    async def scenario():
        environment = FencedEnvironment()
        agent = RacingAgent(environment)
        result = await NavigationHarness(timeout_s=1).run_task(
            task(), NavigationStack(agent, environment)
        )
        assert agent.rejected
        assert result.environment["position"] == 0
        assert result.terminal.status == "success"

    run(scenario())

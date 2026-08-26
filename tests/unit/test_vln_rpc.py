from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

from agents import PassthroughVLNAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.tool_bus import Tool, ToolBus
from schemas import NavGoal, NavTask
from vln.rpc import JsonLineProcess, RPCError, RPCVLNNavigator


WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "rpc_worker.py"
SDK_WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "sdk_worker.py"
ROOT = Path(__file__).resolve().parents[2]


def test_jsonl_worker_can_call_scoped_navigation_tools() -> None:
    async def scenario():
        calls = []

        async def observe(actor, arguments):
            calls.append((actor, arguments))
            return {"observation_id": "1", "channels": {"rgb": "ref://rgb"}}

        bus = ToolBus()
        bus.register(
            (
                Tool(
                    "nav.observe",
                    "Observe.",
                    {"type": "object", "additionalProperties": False},
                    observe,
                ),
            )
        )
        process = JsonLineProcess((sys.executable, str(WORKER)), request_timeout_s=1)
        hello = await process.start(
            bus.client("vln", frozenset({"nav.observe"})),
            {"protocol": 1, "model": "probe"},
        )
        result = await process.request(
            "probe_tool", {"name": "nav.observe", "arguments": {}}
        )
        await process.close()

        assert hello == {"protocol": 1, "model": "probe"}
        assert result["channels"]["rgb"] == "ref://rgb"
        assert calls == [("vln", {})]
        assert bus.audit[0].actor == "vln"

    asyncio.run(scenario())


def test_rpc_navigator_runs_through_standard_job_tools(tmp_path) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        goal = NavGoal("goal", "already at target")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        result = await NavigationHarness(timeout_s=2).run_task(
            NavTask("rpc", goal),
            NavigationStack(
                PassthroughVLNAgent(poll_period_s=0),
                DummyNavigationEnvironment((goal,), targets=(0,)),
                vln=navigator,
            ),
        )
        assert result.terminal.status == "completed"
        assert [event.name for event in result.audit].count("vln.navigate.start") == 1
        assert any(event.name == "vln.navigate.status" for event in result.audit)

    asyncio.run(scenario())


def test_worker_sdk_runs_model_owned_navigation_loop(tmp_path) -> None:
    class SDKNavigator(RPCVLNNavigator):
        model_name = "sdk-fixture"
        required_tools = frozenset({"nav.observe", "nav.move.discrete"})

    async def scenario():
        class ArrayEnvironment(DummyNavigationEnvironment):
            async def _observe(self, actor, arguments):
                observation = await super()._observe(actor, arguments)
                observation["channels"]["rgb"] = np.arange(
                    12, dtype=np.uint8
                ).reshape(2, 2, 3)
                return observation

        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        goal = NavGoal("goal", "move to the target")
        navigator = SDKNavigator(
            (sys.executable, str(SDK_WORKER)),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            env={"PYTHONPATH": str(ROOT / "src")},
            worker_options={"max_steps": 8},
            request_timeout_s=1,
        )
        result = await NavigationHarness(timeout_s=2).run_task(
            NavTask("sdk", goal),
            NavigationStack(
                PassthroughVLNAgent(poll_period_s=0),
                ArrayEnvironment((goal,), targets=(2,)),
                vln=navigator,
            ),
        )

        assert result.terminal.status == "completed"
        assert result.environment["position"] == 2
        assert [event.actor for event in result.audit if event.name == "nav.observe"] == [
            "vln",
            "vln",
            "vln",
        ]
        assert [
            event.name for event in result.audit if event.actor == "vln"
        ].count("nav.move.discrete") == 2

    asyncio.run(scenario())


def test_model_stdout_and_stderr_are_isolated_from_protocol() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--print-logs"), request_timeout_s=1
        )
        hello = await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )
        await process.close()
        assert hello == {"protocol": 1, "model": "x"}
        assert "model log leaked to protocol stdout" in process.stdout_tail
        assert "model diagnostic" in process.stderr_tail

    asyncio.run(scenario())


def test_protocol_rejects_invalid_socket_payload() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--bad-protocol"), request_timeout_s=1
        )
        with pytest.raises(RPCError, match="invalid JSONL"):
            await process.start(
                bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
            )
        assert process.returncode is not None

    asyncio.run(scenario())


def test_hello_timeout_reaps_process() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--delay-hello"), request_timeout_s=0.01
        )
        with pytest.raises(RPCError, match="timed out"):
            await process.start(
                bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
            )
        assert process.returncode is not None

    asyncio.run(scenario())


def test_navigator_handshake_mismatch_reaps_process(tmp_path) -> None:
    async def scenario():
        checkpoint = tmp_path / "checkpoint"
        checkpoint.write_text("fixture")
        navigator = RPCVLNNavigator(
            (sys.executable, str(WORKER), "--wrong-model"),
            upstream_root=tmp_path,
            checkpoint=checkpoint,
            request_timeout_s=1,
        )
        bus = ToolBus()
        with pytest.raises(RPCError, match="handshake mismatch"):
            await navigator.start(
                NavTask("mismatch", NavGoal("goal", "test")),
                bus.client("vln", frozenset()),
            )
        assert navigator._process is None

    asyncio.run(scenario())


def test_late_response_is_discarded_without_killing_reader() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER)), request_timeout_s=1
        )
        await process.start(
            bus.client("vln", frozenset()), {"protocol": 1, "model": "x"}
        )
        process.request_timeout_s = 0.01
        with pytest.raises(RPCError, match="timed out"):
            await process.request("slow", {})
        process.request_timeout_s = 1
        assert await process.request("ping", {}) == "pong"
        await process.close()

    asyncio.run(scenario())


def test_model_specific_adapters_declare_distinct_requirements() -> None:
    from vln import DualVLNNavigator, JanusVLNNavigator, StreamVLNNavigator

    assert "depth" in StreamVLNNavigator.requirements["observation_channels"]
    assert JanusVLNNavigator.requirements["observation_channels"] == ["rgb"]
    dual = DualVLNNavigator(
        ("worker",),
        upstream_root="upstream",
        checkpoint="checkpoint",
        motion_tool="nav.move.trajectory",
    )
    assert dual.required_tools == frozenset({"nav.observe", "nav.move.trajectory"})

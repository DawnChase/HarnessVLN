from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from agents import PassthroughVLNAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.tool_bus import Tool, ToolBus
from schemas import NavGoal, NavTask
from vln.rpc import JsonLineProcess, RPCError, RPCVLNNavigator


WORKER = Path(__file__).resolve().parents[1] / "fixtures" / "rpc_worker.py"


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


def test_protocol_rejects_stdout_log_pollution() -> None:
    async def scenario():
        bus = ToolBus()
        process = JsonLineProcess(
            (sys.executable, str(WORKER), "--bad-stdout"), request_timeout_s=1
        )
        with pytest.raises(RPCError, match="stdout"):
            await process.start(bus.client("vln", frozenset()), {"protocol": 1, "model": "x"})
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

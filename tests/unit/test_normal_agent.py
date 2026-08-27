from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from agents import NormalAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from schemas import NavGoal, NavTask


class ScriptedResponses:
    def __init__(self, outputs: list[list[Any]]) -> None:
        self.outputs = list(outputs)
        self.requests: list[dict[str, Any]] = []

    async def create(self, **request: Any) -> Any:
        captured = dict(request)
        captured["input"] = list(request["input"])
        self.requests.append(captured)
        return SimpleNamespace(output=self.outputs.pop(0))


def function_call(call_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


def test_normal_agent_runs_native_responses_tool_loop() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [function_call("call-1", "nav__observe", {})],
                [
                    function_call(
                        "call-2", "nav__move__discrete", {"action": "forward"}
                    )
                ],
                [
                    function_call(
                        "call-3", "nav__goal__finish", {"status": "completed"}
                    )
                ],
                [
                    function_call(
                        "call-4",
                        "nav__stop",
                        {"status": "completed", "reason": "done"},
                    )
                ],
            ]
        )
        client = SimpleNamespace(responses=responses)
        agent = NormalAgent(
            "test-model",
            ("nav.observe", "nav.move.discrete", "nav.goal.finish"),
            client=client,
        )
        goal = NavGoal("goal", "move to the marker")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("normal", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(1,)),
            ),
        )

        assert result.terminal.status == "completed"
        assert result.environment["position"] == 1
        assert [event.name for event in result.audit] == [
            "nav.observe",
            "nav.move.discrete",
            "nav.goal.finish",
            "nav.stop",
        ]
        assert all(event.actor == "agent" for event in result.audit)

        first_request = responses.requests[0]
        assert first_request["parallel_tool_calls"] is False
        assert first_request["tool_choice"] == "required"
        assert {tool["name"] for tool in first_request["tools"]} == {
            "nav__goal__finish",
            "nav__move__discrete",
            "nav__observe",
            "nav__stop",
        }
        assert all("function" not in tool for tool in first_request["tools"])
        assert all(tool["strict"] is False for tool in first_request["tools"])

        observation_output = responses.requests[1]["input"][-1]
        assert observation_output["type"] == "function_call_output"
        assert observation_output["call_id"] == "call-1"
        assert json.loads(observation_output["output"])["ok"] is True

    run(scenario())


def test_normal_agent_returns_tool_errors_to_the_model() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [
                [
                    function_call(
                        "bad-action", "nav__move__discrete", {"action": "fly"}
                    )
                ],
                [
                    function_call(
                        "stop",
                        "nav__stop",
                        {"status": "completed", "reason": "error handled"},
                    )
                ],
            ]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.move.discrete",),
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test recovery")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("recover", goal),
            NavigationStack(agent, DummyNavigationEnvironment((goal,), targets=(0,))),
        )

        assert result.terminal.status == "completed"
        error_output = json.loads(responses.requests[1]["input"][-1]["output"])
        assert error_output["ok"] is False
        assert error_output["error"]["type"] == "ToolValidationError"
        assert [event.outcome for event in result.audit] == ["invalid", "ok"]

    run(scenario())


def test_normal_agent_stops_when_iteration_budget_is_exhausted() -> None:
    async def scenario() -> None:
        responses = ScriptedResponses(
            [[function_call("observe", "nav__observe", {})]]
        )
        agent = NormalAgent(
            "test-model",
            ("nav.observe",),
            max_iterations=1,
            client=SimpleNamespace(responses=responses),
        )
        goal = NavGoal("goal", "test budget")
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("budget", goal),
            NavigationStack(agent, DummyNavigationEnvironment((goal,), targets=(0,))),
        )

        assert result.terminal.status == "failed"
        assert result.terminal.reason == "agent iteration budget reached"
        assert [event.name for event in result.audit] == ["nav.observe", "nav.stop"]

    run(scenario())

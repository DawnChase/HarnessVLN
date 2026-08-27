from __future__ import annotations

import asyncio

import pytest

from benches.base import BenchmarkCase
from envs.isaac import IsaacNavigationEnvironment
from envs.isaac_vln_pe import environment as isaac_vln_pe
from envs.isaac_vlnverse import environment as isaac_vlnverse
from harness import NavigationHarness, NavigationStack
from harness.requirements import check_navigation_requirements
from schemas import NavGoal, NavTask
from vln import DualVLNNavigator


class FakeIsaacSession:
    def __init__(self) -> None:
        self.actions: list[object] = []
        self.closed = False
        self.action_ticks: dict[str, int] = {}

    async def reset(self):
        return ([{"h1": {"finish_action": False}}], [{"path_key": "fixture"}])

    async def step(self, action):
        self.actions.append(action)
        action_name = next(iter(action[0]["h1"]))
        ticks = self.action_ticks.get(action_name, 0) + 1
        self.action_ticks[action_name] = ticks
        required = {"stand_still": 2, "move_by_discrete": 3, "stop": 2}[action_name]
        finished = ticks % required == 0
        terminated = action_name == "stop" and finished
        robot_observation = {
            "finish_action": finished,
            "rgb": f"rgb-{action_name}-{ticks}",
            "depth": f"depth-{action_name}-{ticks}",
            "globalgps": [1.0, 2.0, 3.0],
            "fail_reason": "private",
        }
        if terminated:
            robot_observation["metrics"] = {
                "fixture": [{"success": 1.0, "spl": 0.75}]
            }
        return ([{"h1": robot_observation}], [0.0], [terminated], [False], [{}])

    async def close(self):
        self.closed = True


def case() -> BenchmarkCase:
    goal = NavGoal("goal:0", "Walk through the hallway.")
    task = NavTask("vln_pe:fixture", goal, scene_id="fixture.usd")
    return BenchmarkCase("vln_pe:fixture", task)


def test_isaac_action_barrier_and_private_native_metrics() -> None:
    class Agent:
        required_tools = frozenset(
            {"nav.observe", "nav.move.discrete", "nav.goal.finish"}
        )

        async def run(self, context):
            initial = await context.nav.observe()
            assert initial["channels"] == {
                "rgb": "rgb-stand_still-2",
                "depth": "depth-stand_still-2",
            }
            assert "fail_reason" not in initial["channels"]
            movement = await context.nav.move_discrete("forward")
            assert movement["native_ticks"] == 3
            transition = await context.nav.finish_goal("completed")
            assert transition == {"done": True, "accepted": True, "native_ticks": 2}
            await context.nav.stop("completed")

    async def scenario():
        fixture = case()
        session = FakeIsaacSession()
        environment = IsaacNavigationEnvironment(
            fixture.environment_episode,
            session_factory=lambda _: session,
            native_actions={
                "forward": {"h1": {"move_by_discrete": [1]}},
                "turn_left": {"h1": {"move_by_discrete": [2]}},
                "turn_right": {"h1": {"move_by_discrete": [3]}},
            },
            warmup_action={"h1": {"stand_still": []}},
            goal_finish_action={"h1": {"stop": []}},
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(Agent(), environment),
        )

        assert result.terminal.status == "completed"
        assert result.terminal.actor == "agent"
        assert result.environment["action_count"] == 2
        assert result.environment["native_tick_count"] == 7
        assert result.environment["native_metrics"] == {
            "fixture": [{"success": 1.0, "spl": 0.75}]
        }
        assert session.closed
        assert all(isinstance(action, list) and len(action) == 1 for action in session.actions)

    asyncio.run(scenario())


def test_isaac_unexpected_native_terminal_preempts_agent() -> None:
    class TerminalSession(FakeIsaacSession):
        async def step(self, action):
            value = await super().step(action)
            action_name = next(iter(action[0]["h1"]))
            if action_name == "move_by_discrete":
                observations, rewards, _, truncated, info = value
                return observations, rewards, [True], truncated, info
            return value

    class Agent:
        required_tools = frozenset({"nav.move.discrete"})

        async def run(self, context):
            await context.nav.move_discrete("forward")
            await asyncio.sleep(10)

    async def scenario():
        fixture = case()
        environment = IsaacNavigationEnvironment(
            fixture.environment_episode,
            session_factory=lambda _: TerminalSession(),
            native_actions={"forward": {"h1": {"move_by_discrete": [1]}}},
            warmup_action={"h1": {"stand_still": []}},
            goal_finish_action={"h1": {"stop": []}},
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(Agent(), environment),
        )
        assert result.terminal.status == "environment_terminal"
        assert result.terminal.actor == "environment"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "module, expected_controller",
    [(isaac_vln_pe, "move_by_discrete"), (isaac_vlnverse, "move_by_flash")],
)
def test_dualvln_isaac_entries_declare_standstill_and_camera(
    monkeypatch, module, expected_controller
) -> None:
    monkeypatch.setattr(module, "load_symbol", lambda _: lambda case: FakeIsaacSession())

    environment = module.from_episode(
        case().environment_episode, session_factory="fixture:factory"
    )

    assert environment.native_actions["stand_still"] == {"h1": {"stand_still": []}}
    assert environment.native_actions["forward"] == {
        "h1": {expected_controller: [1]}
    }
    assert environment.profile.camera == {
        "height": 480,
        "width": 640,
        "hfov_deg": 79,
        "pitch_deg": -30,
    }
    check_navigation_requirements(
        "DualVLNNavigator", DualVLNNavigator.requirements, environment.profile
    )

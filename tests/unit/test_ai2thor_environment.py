from __future__ import annotations

import asyncio

from benches.base import BenchmarkCase
from envs.ai2thor import RoboTHOREnvironment
from harness import NavigationHarness, NavigationStack
from schemas import NavGoal, NavTask


class FakeEvent:
    def __init__(self, x=0.0, *, success=True, visible=True):
        self.frame = "rgb-frame"
        self.depth_frame = "depth-frame"
        self.metadata = {
            "lastActionSuccess": success,
            "errorMessage": "blocked" if not success else "",
            "agent": {
                "position": {"x": x, "y": 0.9, "z": 0.0},
                "rotation": {"y": 30.0},
                "cameraHorizon": 0.0,
            },
            "objects": [{"objectType": "AlarmClock", "visible": visible}],
        }


class FakeController:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.initialization_parameters = {}
        self.last_event = FakeEvent()
        self.actions = []
        self.stopped = False
        self.scene = None
        self.instances.append(self)

    def reset(self, scene):
        self.scene = scene

    def step(self, action, **kwargs):
        self.actions.append((action, kwargs))
        x = self.last_event.metadata["agent"]["position"]["x"]
        if action == "TeleportFull":
            x = kwargs["x"]
        if action == "MoveAhead":
            x += 0.25
        self.last_event = FakeEvent(x, success=action != "RotateLeft")
        return self.last_event

    def stop(self):
        self.stopped = True


def case() -> BenchmarkCase:
    goal = NavGoal("goal", "Find the AlarmClock.", "object", {"category": "AlarmClock"})
    task = NavTask("robothor:fixture", goal, scene_id="FloorPlan_Val1_1")
    return BenchmarkCase(
        "robothor:fixture",
        task,
        {
            "scene": "FloorPlan_Val1_1",
            "initial_position": {"x": 1.0, "y": 0.9, "z": 0.0},
            "initial_orientation": 30,
            "initial_horizon": 0,
            "object_type": "AlarmClock",
        },
        {"shortest_path_length": 0.25},
    )


def test_robothor_maps_reset_observe_actions_stop_and_result() -> None:
    class Agent:
        required_tools = frozenset(
            {"nav.observe", "nav.move.discrete", "nav.goal.finish"}
        )

        async def run(self, context):
            observation = await context.nav.observe()
            assert observation["channels"]["rgb"] == "rgb-frame"
            assert observation["channels"]["object_goal"] == "AlarmClock"
            assert set(observation["channels"]) == {"rgb", "object_goal"}
            assert "pose" not in observation
            assert observation["extras"] == {}
            failed_turn = await context.nav.move_discrete("turn_left")
            assert failed_turn == {
                "action": "turn_left",
                "accepted": True,
                "action_count": 1,
            }
            await context.nav.move_discrete("forward")
            transition = await context.nav.finish_goal("completed")
            assert transition == {"done": True, "accepted": True, "success": True}
            await context.nav.stop("completed")

    async def scenario():
        fixture = case()
        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(
                Agent(),
                RoboTHOREnvironment(
                    fixture,
                    controller_kwargs={"width": 640, "height": 480},
                    controller_factory=FakeController,
                ),
            ),
        )
        controller = FakeController.instances[-1]
        assert controller.scene == "FloorPlan_Val1_1"
        assert controller.initialization_parameters["robothorChallengeEpisodeId"] == (
            "robothor:fixture"
        )
        assert [action for action, _ in controller.actions] == [
            "TeleportFull",
            "RotateLeft",
            "MoveAhead",
            "Stop",
        ]
        assert controller.stopped
        assert result.terminal.status == "completed"
        assert result.environment["success"] is True
        assert result.environment["path_length"] == 0.25
        assert len(result.environment["actions_taken"]) == 3
        assert result.environment["actions_taken"][0]["success"] is False

    asyncio.run(scenario())


def test_robothor_optional_depth_pose_and_feedback_are_declared() -> None:
    class Agent:
        required_tools = frozenset({"nav.observe", "nav.move.discrete"})

        async def run(self, context):
            observation = await context.nav.observe()
            assert observation["channels"]["depth"] == "depth-frame"
            assert observation["channels"]["pose"]["frame"] == "thor_world"
            assert observation["extras"]["last_action_success"] is True
            movement = await context.nav.move_discrete("turn_left")
            assert movement["native_success"] is False
            await context.nav.stop("completed")

    async def scenario():
        fixture = case()
        environment = RoboTHOREnvironment(
            fixture,
            controller_factory=FakeController,
            render_depth=True,
            expose_pose=True,
            expose_action_feedback=True,
        )
        assert environment.profile.observation_channels == frozenset(
            {"rgb", "depth", "pose", "object_goal"}
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(Agent(), environment),
        )
        assert result.terminal.status == "completed"

    asyncio.run(scenario())


def test_failed_native_action_is_not_environment_terminal() -> None:
    class Agent:
        required_tools = frozenset({"nav.move.discrete"})

        async def run(self, context):
            result = await context.nav.move_discrete("turn_left")
            assert result["accepted"] is True
            assert "native_success" not in result
            await context.nav.stop("completed", "failed action handled by agent")

    async def scenario():
        fixture = case()
        result = await NavigationHarness(timeout_s=1).run_task(
            fixture.task,
            NavigationStack(
                Agent(),
                RoboTHOREnvironment(fixture, controller_factory=FakeController),
            ),
        )
        assert result.terminal.status == "completed"
        assert result.terminal.actor == "agent"
        assert result.environment["actions_taken"][0]["success"] is False

    asyncio.run(scenario())

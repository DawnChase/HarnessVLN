from __future__ import annotations

import asyncio

import numpy as np

from benches.base import BenchmarkCase
from envs.goat import GOATHabitatEnvironment
from harness import NavigationHarness, NavigationStack
from harness.config import ComponentSpec, load_config
from schemas import NavGoal, NavTask


class FakeSimulator:
    def __init__(self) -> None:
        self.goal_renders = 0

    def get_agent_state(self):
        return type("State", (), {"position": [0.0, 1.0, 0.0]})()

    def get_observations_at(self, **kwargs):
        assert kwargs["keep_agent_at_new_pose"] is False
        self.goal_renders += 1
        return {"rgb": np.full((4, 3, 3), 7, dtype=np.uint8)}


class FakeGOATSession:
    def __init__(self) -> None:
        self.sim = FakeSimulator()
        self.actions: list[str] = []
        self.goal_index = 0
        self.closed = False
        self.episode_over = False

    def reset(self):
        return self._observation()

    def step(self, action):
        self.actions.append(action)
        if action == "subtask_stop":
            self.goal_index += 1
            self.episode_over = self.goal_index == 2
        return self._observation()

    def get_metrics(self):
        success = [1.0 if index < self.goal_index else 0.0 for index in range(2)]
        spl = [0.8 if index < self.goal_index else 0.0 for index in range(2)]
        return {
            "distance_to_goal": {
                "distance_to_target": 2.0,
                "prev_distance_to_target": 0.1,
            },
            "success": {"subtask_success": success},
            "spl": {"spl_by_subtask": spl},
        }

    def close(self):
        self.closed = True

    @staticmethod
    def _observation():
        return {
            "rgb": np.zeros((4, 3, 3), dtype=np.uint8),
            "gps": [0.0, 0.0],
            "compass": [0.0],
        }


def goat_case() -> BenchmarkCase:
    goals = (
        NavGoal("goal:0", "Find the chair.", "object"),
        NavGoal("goal:1", "Find the shown object.", "image"),
    )
    return BenchmarkCase(
        "goat:fixture:0",
        NavTask("goat:fixture:0", goals[0]),
        {
            "goal_stream": goals,
            "goal_specs": (
                {"modality": "object", "category": "chair", "object_id": None},
                {
                    "modality": "image",
                    "category": "lamp",
                    "object_id": "lamp_1",
                    "image_goal": {
                        "position": [1.0, 2.0, 3.0],
                        "rotation": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            ),
        },
    )


def test_goat_environment_keeps_session_and_normalizes_per_goal_metrics() -> None:
    class Agent:
        required_tools = frozenset({"nav.observe", "nav.goal.finish"})

        async def run(self, context):
            first = await context.nav.observe()
            assert first["channels"]["goal_image"] is None
            transition = await context.nav.finish_goal("completed")
            assert transition["done"] is False
            assert transition["goal"]["modality"] == "image"

            second = await context.nav.observe()
            assert second["channels"]["goal_image"].shape == (4, 3, 3)
            cached = await context.nav.observe()
            assert np.array_equal(
                cached["channels"]["goal_image"], second["channels"]["goal_image"]
            )
            assert (await context.nav.finish_goal("completed"))["done"] is True
            await context.nav.stop("completed")

    async def scenario():
        case = goat_case()
        session = FakeGOATSession()
        environment = GOATHabitatEnvironment(
            case.environment_episode,
            native_factory=lambda _: session,
            native_actions={"forward": "move_forward"},
            goal_finish_action="subtask_stop",
            observation_channels=("rgb", "gps", "compass"),
        )
        result = await NavigationHarness(timeout_s=1).run_task(
            case.task, NavigationStack(Agent(), environment)
        )

        assert result.terminal.status == "completed"
        assert session.actions == ["subtask_stop", "subtask_stop"]
        assert session.sim.goal_renders == 1
        assert session.closed
        assert result.environment["goal_results"] == [
            {
                "goal_id": "goal:0",
                "modality": "object",
                "success": 1.0,
                "spl": 0.8,
                "distance_to_goal": 0.1,
            },
            {
                "goal_id": "goal:1",
                "modality": "image",
                "success": 1.0,
                "spl": 0.8,
                "distance_to_goal": 0.1,
            },
        ]

    asyncio.run(scenario())


def test_goat_config_composes_benchmark_and_environment() -> None:
    resolved = load_config(
        (
            "config/benches/goat.yaml",
            "config/agents/passthrough.yaml",
            "config/vln/streamvln.yaml",
            "config/envs/habitat_goat.yaml",
        )
    )
    benchmark = ComponentSpec.from_config(resolved.data["benchmark"]).create()
    environment = ComponentSpec.from_config(
        resolved.data["stack"]["environment"]
    ).create(episode=goat_case().environment_episode)

    assert benchmark.name == "goat_bench"
    assert benchmark.split == "val_unseen"
    assert resolved.data["runner"]["parallelism"] == 2
    assert environment.profile.observation_channels == frozenset(
        {"rgb", "gps", "compass", "pose", "goal_image"}
    )
    assert environment.profile.motion.actions == frozenset(
        {"forward", "turn_left", "turn_right", "look_up", "look_down"}
    )

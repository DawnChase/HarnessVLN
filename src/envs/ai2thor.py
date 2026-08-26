from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from benches.base import BenchmarkCase
from harness.errors import HarnessError, ToolClosedError
from harness.tool_bus import Tool
from schemas import EnvironmentTerminal, MotionProfile, NavigationProfile, Observation, Pose


ControllerFactory = Callable[..., Any]


class RoboTHOREnvironment:
    profile = NavigationProfile(
        observation_channels=frozenset({"rgb", "depth", "pose", "object_goal"}),
        motion=MotionProfile(
            tool="nav.move.discrete",
            actions=frozenset(
                {"forward", "turn_left", "turn_right", "look_up", "look_down"}
            ),
            frame="thor_world",
            units="meters_degrees",
            forward_m=0.25,
            turn_deg=30.0,
        ),
        camera={"height": 480, "width": 640},
    )

    ACTIONS = {
        "forward": "MoveAhead",
        "turn_left": "RotateLeft",
        "turn_right": "RotateRight",
        "look_up": "LookUp",
        "look_down": "LookDown",
    }

    def __init__(
        self,
        case: BenchmarkCase,
        *,
        controller_kwargs: Mapping[str, Any] | None = None,
        controller_factory: ControllerFactory | None = None,
        max_actions: int = 500,
    ) -> None:
        self.case = case
        self.controller_kwargs = dict(controller_kwargs or {})
        self.controller_factory = controller_factory
        self.max_actions = max_actions
        self._controller: Any = None
        self._event: Any = None
        self._running = False
        self._generation = 0
        self._lock = asyncio.Lock()
        self._terminal: asyncio.Future[EnvironmentTerminal] | None = None
        self._observation_id = 0
        self._actions: list[dict[str, Any]] = []
        self._trajectory: list[dict[str, float]] = []
        self._path_length = 0.0
        self._success = False
        self._goal_finished = False

    async def start(self, task) -> Sequence[Tool]:
        del task
        if self._running or self._controller is not None:
            raise HarnessError("RoboTHOR environment instances are single-use")
        factory = self.controller_factory or self._load_controller_factory()
        self._controller = factory(**self.controller_kwargs)
        setup = self.case.setup
        self._controller.initialization_parameters["robothorChallengeEpisodeId"] = (
            self.case.case_id
        )
        self._controller.reset(setup["scene"])
        self._event = self._controller.step(
            action="TeleportFull",
            **setup["initial_position"],
            rotation={"x": 0, "y": setup["initial_orientation"], "z": 0},
            horizon=setup["initial_horizon"],
            standing=True,
        )
        if not self._event.metadata.get("lastActionSuccess", False):
            raise HarnessError(
                f"RoboTHOR TeleportFull failed: {self._event.metadata.get('errorMessage', '')}"
            )
        self._running = True
        self._terminal = asyncio.get_running_loop().create_future()
        self._trajectory.append(self._agent_state())
        return (
            Tool(
                "nav.observe",
                "Get the current RoboTHOR RGB-D observation.",
                {"type": "object", "additionalProperties": False},
                self._observe,
            ),
            Tool(
                "nav.move.discrete",
                "Execute one RoboTHOR discrete action.",
                {
                    "type": "object",
                    "properties": {"action": {"enum": sorted(self.ACTIONS)}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                self._move,
                writes=True,
            ),
            Tool(
                "nav.goal.finish",
                "Issue RoboTHOR Stop for the current ObjectNav goal.",
                {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["status"],
                    "additionalProperties": False,
                },
                self._finish_goal,
                writes=True,
            ),
        )

    async def _observe(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor, arguments
        self._ensure_running()
        self._observation_id += 1
        now = time.time()
        state = self._agent_state()
        observation = Observation(
            str(self._observation_id),
            now,
            now,
            "thor_world",
            {
                "rgb": self._event.frame,
                "depth": self._event.depth_frame,
                "object_goal": self.case.setup["object_type"],
            },
            Pose(
                "thor_world",
                state["x"],
                state["z"],
                state["y"],
                yaw=state["rotation"],
                pitch=state["horizon"],
            ),
            {
                "last_action_success": self._event.metadata.get(
                    "lastActionSuccess", True
                ),
                "error_message": self._event.metadata.get("errorMessage", ""),
            },
        )
        return observation.as_dict()

    async def _move(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        generation = self._generation
        async with self._lock:
            self._ensure_running()
            if generation != self._generation:
                raise ToolClosedError("stale RoboTHOR motion generation")
            action = arguments["action"]
            previous = self._agent_state()
            self._event = self._controller.step(action=self.ACTIONS[action])
            current = self._agent_state()
            self._path_length += math.sqrt(
                (current["x"] - previous["x"]) ** 2
                + (current["y"] - previous["y"]) ** 2
                + (current["z"] - previous["z"]) ** 2
            )
            self._trajectory.append(current)
            self._actions.append(
                {
                    "action": self.ACTIONS[action],
                    "success": bool(self._event.metadata.get("lastActionSuccess", False)),
                }
            )
            if len(self._actions) >= self.max_actions:
                self._publish_terminal("completed", "RoboTHOR action budget reached")
            return {
                "action": action,
                "accepted": True,
                "native_success": self._actions[-1]["success"],
                "action_count": len(self._actions),
            }

    async def _finish_goal(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        del actor, arguments
        async with self._lock:
            self._ensure_running()
            self._event = self._controller.step(action="Stop")
            self._actions.append(
                {
                    "action": "Stop",
                    "success": bool(self._event.metadata.get("lastActionSuccess", False)),
                }
            )
            category = self.case.setup["object_type"]
            target = next(
                (
                    item
                    for item in self._event.metadata.get("objects", ())
                    if item.get("objectType") == category
                ),
                None,
            )
            self._success = bool(target and target.get("visible", False))
            self._goal_finished = True
            return {"done": True, "accepted": True, "success": self._success}

    async def wait_terminal(self) -> EnvironmentTerminal:
        if self._terminal is None:
            raise HarnessError("RoboTHOR environment is not started")
        return await self._terminal

    async def stop(self, reason: str) -> None:
        del reason
        async with self._lock:
            self._generation += 1
            self._running = False
            if self._controller is not None:
                self._controller.stop()

    def result(self) -> dict[str, Any]:
        return {
            "success": self._success,
            "goal_finished": self._goal_finished,
            "path_length": self._path_length,
            "trajectory": list(self._trajectory),
            "actions_taken": list(self._actions),
            "stopped": not self._running,
        }

    def _agent_state(self) -> dict[str, float]:
        agent = self._event.metadata["agent"]
        return {
            "x": float(agent["position"]["x"]),
            "y": float(agent["position"]["y"]),
            "z": float(agent["position"]["z"]),
            "rotation": float(agent["rotation"]["y"]),
            "horizon": float(agent["cameraHorizon"]),
        }

    def _publish_terminal(
        self, kind: Literal["completed", "failed"], reason: str
    ) -> None:
        if self._terminal is not None and not self._terminal.done():
            self._terminal.set_result(EnvironmentTerminal(kind, reason))

    def _ensure_running(self) -> None:
        if not self._running:
            raise ToolClosedError("RoboTHOR environment is stopped")

    @staticmethod
    def _load_controller_factory() -> ControllerFactory:
        try:
            from ai2thor.controller import Controller
        except ImportError as error:
            raise HarnessError(
                "RoboTHOR requires ai2thor==2.7.2 in the environment process"
            ) from error
        return Controller

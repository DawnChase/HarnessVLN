from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from benches.base import BenchmarkCase
from harness.config import load_symbol
from harness.errors import HarnessError, ToolClosedError
from harness.tool_bus import Tool
from schemas import EnvironmentTerminal, MotionProfile, NavigationProfile, Observation, Pose


NativeFactory = Callable[[BenchmarkCase], Any]


class HabitatEnvironment:
    def __init__(
        self,
        case: BenchmarkCase,
        *,
        native_factory: NativeFactory,
        native_actions: Mapping[str, Any] | None = None,
        goal_finish_action: Any = 0,
        observation_channels: Sequence[str] = (
            "rgb",
            "depth",
            "gps",
            "compass",
        ),
        static_channels: Mapping[str, Any] | None = None,
        expose_pose: bool = True,
        forward_m: float = 0.25,
        turn_deg: float = 15.0,
    ) -> None:
        self.case = case
        self.native_factory = native_factory
        self.native_actions = dict(
            native_actions
            or {"forward": 1, "turn_left": 2, "turn_right": 3, "look_up": 4, "look_down": 5}
        )
        self.goal_finish_action = goal_finish_action
        self.observation_channels = tuple(observation_channels)
        self.static_channels = dict(static_channels or {})
        self.expose_pose = expose_pose
        provided_channels = set(self.observation_channels) | set(self.static_channels)
        if expose_pose:
            provided_channels.add("pose")
        self.profile = NavigationProfile(
            observation_channels=frozenset(provided_channels),
            motion=MotionProfile(
                "nav.move.discrete",
                frozenset(self.native_actions),
                frame="habitat_episode",
                units="meters_degrees",
                forward_m=forward_m,
                turn_deg=turn_deg,
            ),
        )
        self._session: Any = None
        self._observation: Mapping[str, Any] = {}
        self._running = False
        self._generation = 0
        self._lock = asyncio.Lock()
        self._terminal: asyncio.Future[EnvironmentTerminal] | None = None
        self._goal_stream = tuple(case.setup.get("goal_stream", (case.task.goal,)))
        self._goal_index = 0
        self._actions_this_goal = 0
        self._action_count = 0
        self._observation_id = 0
        self._metrics: dict[str, Any] = {}

    async def start(self, task) -> Sequence[Tool]:
        del task
        if self._session is not None:
            raise HarnessError("Habitat environment instances are single-use")
        self._session = self.native_factory(self.case)
        self._observation = self._session.reset()
        self._running = True
        self._terminal = asyncio.get_running_loop().create_future()
        self._capture_metrics()
        return (
            Tool(
                "nav.observe",
                "Get the current normalized Habitat navigation observation.",
                {"type": "object", "additionalProperties": False},
                self._observe,
            ),
            Tool(
                "nav.move.discrete",
                "Execute one Habitat discrete action.",
                {
                    "type": "object",
                    "properties": {"action": {"enum": sorted(self.native_actions)}},
                    "required": ["action"],
                    "additionalProperties": False,
                },
                self._move,
                writes=True,
            ),
            Tool(
                "nav.goal.finish",
                "Finish the current Habitat navigation goal.",
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
        pose = self._pose()
        channels = {
            name: self._observation[name]
            for name in self.observation_channels
            if name in self._observation
        }
        channels.update(self.static_channels)
        if pose is not None:
            channels.setdefault("pose", pose.as_dict())
        return Observation(
            str(self._observation_id),
            now,
            now,
            "habitat_episode",
            channels,
            pose,
            {"goal_id": self._goal_stream[self._goal_index].goal_id},
        ).as_dict()

    async def _move(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        generation = self._generation
        async with self._lock:
            self._ensure_running()
            if generation != self._generation:
                raise ToolClosedError("stale Habitat motion generation")
            self._observation = self._session.step(self.native_actions[arguments["action"]])
            self._action_count += 1
            self._actions_this_goal += 1
            self._capture_metrics()
            self._capture_native_terminal()
            return {
                "action": arguments["action"],
                "action_count": self._action_count,
                "goal_action_count": self._actions_this_goal,
                "native_terminal": self._native_terminal(),
            }

    async def _finish_goal(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        del actor
        async with self._lock:
            self._ensure_running()
            if arguments["status"] != "completed":
                return {"done": True, "accepted": False}
            self._observation = self._session.step(self.goal_finish_action)
            self._capture_metrics()
            if self._goal_index + 1 >= len(self._goal_stream):
                return {"done": True, "accepted": True}
            self._goal_index += 1
            self._actions_this_goal = 0
            goal = self._goal_stream[self._goal_index]
            return {
                "done": False,
                "accepted": True,
                "goal": {
                    "goal_id": goal.goal_id,
                    "instruction": goal.instruction,
                    "modality": goal.modality,
                    "public": dict(goal.public),
                },
            }

    async def wait_terminal(self) -> EnvironmentTerminal:
        if self._terminal is None:
            raise HarnessError("Habitat environment is not started")
        return await self._terminal

    async def stop(self, reason: str) -> None:
        del reason
        async with self._lock:
            self._generation += 1
            self._running = False
            self._capture_metrics()
            if self._session is not None:
                self._session.close()

    def result(self) -> dict[str, Any]:
        return {
            **self._metrics,
            "action_count": self._action_count,
            "goal_index": self._goal_index,
            "goal_count": len(self._goal_stream),
            "stopped": not self._running,
        }

    def _pose(self) -> Pose | None:
        if not self.expose_pose:
            return None
        if "gps" in self._observation:
            gps = self._observation["gps"]
            compass: Any = self._observation.get("compass", 0.0)
            try:
                yaw = float(compass[0])
            except (IndexError, TypeError):
                yaw = float(compass)
            return Pose(
                "habitat_episode",
                float(gps[0]),
                float(gps[1]),
                yaw=yaw,
            )
        sim = getattr(self._session, "sim", None)
        if sim is not None and hasattr(sim, "get_agent_state"):
            state = sim.get_agent_state()
            position = state.position
            return Pose(
                "habitat_world",
                float(position[0]),
                float(position[2]),
                float(position[1]),
            )
        return None

    def _capture_metrics(self) -> None:
        if self._session is not None and hasattr(self._session, "get_metrics"):
            value = self._session.get_metrics()
            if isinstance(value, Mapping):
                self._metrics = dict(value)

    def _native_terminal(self) -> bool:
        return bool(getattr(self._session, "episode_over", False))

    def _capture_native_terminal(self) -> None:
        if self._native_terminal() and self._terminal is not None and not self._terminal.done():
            self._terminal.set_result(
                EnvironmentTerminal("completed", "Habitat episode ended")
            )

    def _ensure_running(self) -> None:
        if not self._running:
            raise ToolClosedError("Habitat environment is stopped")


def from_case(
    case: BenchmarkCase,
    *,
    native_factory: str,
    native_factory_params: Mapping[str, Any] | None = None,
    **adapter_params: Any,
) -> HabitatEnvironment:
    factory = load_symbol(native_factory)

    def build(private_case: BenchmarkCase) -> Any:
        return factory(private_case, **dict(native_factory_params or {}))

    return HabitatEnvironment(case, native_factory=build, **adapter_params)


def create_native_session(
    case: BenchmarkCase,
    *,
    config_path: str | Path,
    config_loader: str,
    config_options: Sequence[Any] | None = None,
) -> Any:
    """Create a one-episode Habitat session inside a Habitat-enabled process."""

    try:
        import habitat
    except ImportError as error:
        raise HarnessError("Habitat adapter requires habitat-lab and habitat-sim") from error
    loader = load_symbol(config_loader)
    config = loader(str(config_path), list(config_options or ()))
    task_config = getattr(config, "TASK_CONFIG", None)
    if task_config is not None:
        dataset_config = task_config.DATASET
        dataset = habitat.make_dataset(dataset_config.TYPE, config=dataset_config)
        env_config = task_config
    elif hasattr(config, "habitat"):
        dataset_config = config.habitat.dataset
        dataset = habitat.make_dataset(dataset_config.type, config=dataset_config)
        env_config = config
    else:
        raise HarnessError("unsupported Habitat config shape")
    episode_id = case.case_id.rsplit(":", 1)[-1]
    matches = [
        episode
        for episode in dataset.episodes
        if str(episode.episode_id) == episode_id
        and (
            case.task.scene_id is None
            or str(episode.scene_id).replace("//", "/")
            == case.task.scene_id.replace("//", "/")
        )
    ]
    if len(matches) != 1:
        raise HarnessError(
            f"expected one Habitat episode for {case.case_id}, found {len(matches)}"
        )
    dataset.episodes = matches
    return habitat.Env(config=env_config, dataset=dataset)

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol, cast

from harness.errors import HarnessError, ToolClosedError
from harness.output import ModuleOutput, NULL_MODULE_OUTPUT
from harness.tool_bus import Tool
from schemas import (
    EnvironmentEpisode,
    EnvironmentTerminal,
    MotionProfile,
    NavigationProfile,
    Observation,
)


class IsaacSession(Protocol):
    def reset(self) -> Any | Awaitable[Any]: ...

    def step(self, action: list[Mapping[str, Any]]) -> Any | Awaitable[Any]: ...

    def close(self) -> Any | Awaitable[Any]: ...


SessionFactory = Callable[[EnvironmentEpisode], IsaacSession | Awaitable[IsaacSession]]


class IsaacNavigationEnvironment:
    """One-lane adapter that hides Isaac physics ticks behind navigation tools."""

    def __init__(
        self,
        episode: EnvironmentEpisode,
        *,
        session_factory: SessionFactory,
        native_actions: Mapping[str, Mapping[str, Any]],
        warmup_action: Mapping[str, Any],
        goal_finish_action: Mapping[str, Any],
        observation_channels: Mapping[str, str] | None = None,
        frame: str = "isaac_world",
        forward_m: float = 0.25,
        turn_deg: float = 15.0,
        camera: Mapping[str, Any] | None = None,
        max_native_ticks_per_action: int = 2000,
    ) -> None:
        if max_native_ticks_per_action <= 0:
            raise ValueError("max_native_ticks_per_action must be positive")
        self.episode = episode
        self.session_factory = session_factory
        self.native_actions = dict(native_actions)
        self.warmup_action = dict(warmup_action)
        self.goal_finish_action = dict(goal_finish_action)
        self.observation_channels = dict(
            observation_channels or {"rgb": "rgb", "depth": "depth"}
        )
        self.frame = frame
        self.max_native_ticks_per_action = max_native_ticks_per_action
        self.profile = NavigationProfile(
            observation_channels=frozenset(self.observation_channels),
            motion=MotionProfile(
                "nav.move.discrete",
                frozenset(self.native_actions),
                frame=frame,
                units="meters_degrees",
                forward_m=forward_m,
                turn_deg=turn_deg,
            ),
            camera=dict(camera or {}),
        )
        self._session: IsaacSession | None = None
        self._observation: Mapping[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._running = False
        self._generation = 0
        self._lock = asyncio.Lock()
        self._terminal: asyncio.Future[EnvironmentTerminal] | None = None
        self._observation_id = 0
        self._action_count = 0
        self._native_tick_count = 0
        self._output = NULL_MODULE_OUTPUT

    async def start(
        self, task, output: ModuleOutput = NULL_MODULE_OUTPUT
    ) -> Sequence[Tool]:
        if self._session is not None:
            raise HarnessError("Isaac environment instances are single-use")
        self._output = output
        if task.goal.goal_id != self.episode.task.goal.goal_id:
            raise HarnessError("task initial goal does not match Isaac episode")
        try:
            self._session = await _resolve(self.session_factory(self.episode))
            reset_value = await _resolve(self._session.reset())
            self._observation, terminated = self._normalize_reset(reset_value)
            if terminated:
                raise HarnessError("Isaac episode terminated during reset")
            self._record_main_camera("reset")
            self._running = True
            self._terminal = asyncio.get_running_loop().create_future()
            _, terminated = await self._run_native_action(
                self.warmup_action, publish_terminal=False
            )
            if terminated:
                raise HarnessError("Isaac episode terminated during warmup")
            output.record({"profile": self.profile.as_dict()})
        except BaseException:
            self._running = False
            if self._session is not None:
                await _resolve(self._session.close())
            raise
        return (
            Tool(
                "nav.observe",
                "Get the latest completed Isaac navigation observation.",
                {"type": "object", "additionalProperties": False},
                self._observe,
            ),
            Tool(
                "nav.move.discrete",
                "Execute one completed high-level Isaac navigation action.",
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
                "Submit the native Isaac stop action for the navigation goal.",
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
        channels = {
            public_name: self._observation[native_name]
            for public_name, native_name in self.observation_channels.items()
            if native_name in self._observation
        }
        return Observation(
            str(self._observation_id),
            now,
            now,
            self.frame,
            channels,
            extras={"goal_id": self.episode.task.goal.goal_id},
        ).as_dict()

    async def _move(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        generation = self._generation
        async with self._lock:
            self._ensure_running()
            if generation != self._generation:
                raise ToolClosedError("stale Isaac motion generation")
            ticks, _ = await self._run_native_action(
                self.native_actions[arguments["action"]], publish_terminal=True
            )
            self._action_count += 1
            return {
                "action": arguments["action"],
                "accepted": True,
                "action_count": self._action_count,
                "native_ticks": ticks,
            }

    async def _finish_goal(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        del actor
        async with self._lock:
            self._ensure_running()
            if arguments["status"] != "completed":
                return {"done": True, "accepted": False}
            ticks, _ = await self._run_native_action(
                self.goal_finish_action, publish_terminal=False
            )
            self._action_count += 1
            return {"done": True, "accepted": True, "native_ticks": ticks}

    async def _run_native_action(
        self, action: Mapping[str, Any], *, publish_terminal: bool
    ) -> tuple[int, bool]:
        assert self._session is not None
        for tick in range(1, self.max_native_ticks_per_action + 1):
            value = await _resolve(self._session.step([action]))
            observation, terminated = self._normalize_step(value)
            self._observation = observation
            self._native_tick_count += 1
            self._record_main_camera("native_tick")
            self._capture_metrics(observation)
            if terminated and publish_terminal:
                self._publish_terminal("completed", "Isaac episode ended")
            if terminated or bool(observation.get("finish_action", False)):
                return tick, terminated
        raise HarnessError(
            f"Isaac action did not finish within {self.max_native_ticks_per_action} ticks"
        )

    async def wait_terminal(self) -> EnvironmentTerminal:
        if self._terminal is None:
            raise HarnessError("Isaac environment is not started")
        return await self._terminal

    async def stop(self, reason: str) -> None:
        del reason
        async with self._lock:
            if not self._running and self._session is None:
                return
            self._generation += 1
            self._running = False
            session, self._session = self._session, None
            if session is not None:
                await _resolve(session.close())

    def result(self) -> dict[str, Any]:
        return {
            "native_metrics": dict(self._metrics),
            "action_count": self._action_count,
            "native_tick_count": self._native_tick_count,
            "stopped": not self._running,
        }

    def _normalize_reset(self, value: Any) -> tuple[Mapping[str, Any], bool]:
        observation = value[0] if isinstance(value, tuple) else value
        return self._unwrap_observation(observation), False

    def _normalize_step(self, value: Any) -> tuple[Mapping[str, Any], bool]:
        if not isinstance(value, tuple) or len(value) < 3:
            raise HarnessError("Isaac step must return a vector environment tuple")
        observation = self._unwrap_observation(value[0])
        terminated = _first_bool(value[2])
        truncated = _first_bool(value[3]) if len(value) > 3 else False
        return observation, terminated or truncated

    @staticmethod
    def _unwrap_observation(value: Any) -> Mapping[str, Any]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
            if len(value) != 1:
                raise HarnessError("Isaac v0.1 requires a list-of-one observation")
            value = value[0]
        if not isinstance(value, Mapping):
            raise HarnessError("Isaac observation must be an object")
        if "h1" in value:
            value = value["h1"]
        if not isinstance(value, Mapping):
            raise HarnessError("Isaac robot observation must be an object")
        return cast(Mapping[str, Any], value)

    def _capture_metrics(self, observation: Mapping[str, Any]) -> None:
        metrics = observation.get("metrics")
        if isinstance(metrics, Mapping):
            self._metrics = dict(metrics)

    def _record_main_camera(self, stage: str) -> None:
        native_name = self.observation_channels.get("rgb")
        if native_name is None or native_name not in self._observation:
            self._output.unavailable(
                "main_camera", "Isaac observation has no configured rgb channel"
            )
            return
        self._output.frame(
            "main_camera",
            self._observation[native_name],
            {
                "source_time": time.time(),
                "stage": stage,
                "action_index": self._action_count,
                "native_tick_index": self._native_tick_count,
            },
        )

    def _publish_terminal(
        self, kind: Literal["completed", "failed"], reason: str
    ) -> None:
        if self._terminal is not None and not self._terminal.done():
            self._terminal.set_result(EnvironmentTerminal(kind, reason))

    def _ensure_running(self) -> None:
        if not self._running:
            raise ToolClosedError("Isaac environment is stopped")


async def _resolve(value: Any | Awaitable[Any]) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _first_bool(value: Any) -> bool:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return bool(value[0]) if value else False
    return bool(value)

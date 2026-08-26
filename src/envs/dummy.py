from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from harness.errors import HarnessError, ToolClosedError
from harness.tool_bus import Tool
from schemas import EnvironmentTerminal, NavGoal, NavTask, Observation, Pose


class DummyNavigationEnvironment:
    """Small deterministic adapter used to exercise real Harness contracts."""

    def __init__(
        self,
        goals: Sequence[NavGoal],
        *,
        targets: Sequence[int],
        start_position: int = 0,
        max_actions_per_goal: int = 500,
    ) -> None:
        if not goals or len(goals) != len(targets):
            raise ValueError("goals and targets must have the same non-zero length")
        self.goals = tuple(goals)
        self.targets = tuple(targets)
        self.position = start_position
        self.max_actions_per_goal = max_actions_per_goal
        self.goal_index = 0
        self.actions_this_goal = 0
        self.action_count = 0
        self.start_count = 0
        self.goal_transitions = 0
        self._running = False
        self._generation = 0
        self._motion_lock = asyncio.Lock()
        self._terminal: asyncio.Future[EnvironmentTerminal] | None = None
        self._observation_id = 0

    async def start(self, task: NavTask) -> Sequence[Tool]:
        if self._running or self.start_count:
            raise HarnessError("environment instances are single-use")
        if task.goal.goal_id != self.goals[0].goal_id:
            raise HarnessError("task initial goal does not match environment setup")
        self.start_count += 1
        self._running = True
        self._terminal = asyncio.get_running_loop().create_future()
        return (
            Tool(
                "nav.observe",
                "Get the current normalized navigation observation.",
                {"type": "object", "additionalProperties": False},
                self._observe,
            ),
            Tool(
                "nav.move.discrete",
                "Execute one discrete navigation action.",
                {
                    "type": "object",
                    "properties": {
                        "action": {
                            "enum": ["forward", "backward", "turn_left", "turn_right"]
                        }
                    },
                    "required": ["action"],
                    "additionalProperties": False,
                },
                self._move,
                writes=True,
            ),
            Tool(
                "nav.goal.finish",
                "Finish the current goal and reveal the next goal, if any.",
                {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "minLength": 1},
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
        target = self.targets[self.goal_index]
        observation = Observation(
            observation_id=str(self._observation_id),
            source_time=now,
            received_time=now,
            frame="dummy_world",
            channels={"target_delta": target - self.position},
            pose=Pose("dummy_world", float(self.position), 0.0, yaw=0.0),
            extras={"goal_id": self.goals[self.goal_index].goal_id},
        )
        return observation.as_dict()

    async def _move(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        generation = self._generation
        async with self._motion_lock:
            self._ensure_running()
            if generation != self._generation:
                raise ToolClosedError("stale motion generation")
            action = arguments["action"]
            if action == "forward":
                self.position += 1
            elif action == "backward":
                self.position -= 1
            self.action_count += 1
            self.actions_this_goal += 1
            if self.actions_this_goal >= self.max_actions_per_goal:
                await self._advance_goal()
            return {
                "position": self.position,
                "action_count": self.action_count,
                "goal_id": self.goals[self.goal_index].goal_id,
            }

    async def _finish_goal(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        del actor
        self._ensure_running()
        if arguments["status"] != "completed":
            return {"done": True, "accepted": False, "reason": arguments.get("reason", "")}
        return await self._advance_goal()

    async def _advance_goal(self) -> dict[str, Any]:
        if self.goal_index + 1 >= len(self.goals):
            return {"done": True, "accepted": True}
        self.goal_index += 1
        self.goal_transitions += 1
        self.actions_this_goal = 0
        goal = self.goals[self.goal_index]
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

    async def stop(self, reason: str) -> None:
        del reason
        async with self._motion_lock:
            self._generation += 1
            self._running = False

    async def wait_terminal(self) -> EnvironmentTerminal:
        if self._terminal is None:
            raise HarnessError("environment is not started")
        return await self._terminal

    def result(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "action_count": self.action_count,
            "goal_index": self.goal_index,
            "goal_transitions": self.goal_transitions,
            "start_count": self.start_count,
            "stopped": not self._running,
        }

    def _ensure_running(self) -> None:
        if not self._running:
            raise ToolClosedError("environment is stopped")

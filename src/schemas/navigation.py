from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Pose:
    """A metric pose whose frame and units are explicit."""

    frame: str
    x: float
    y: float
    z: float = 0.0
    yaw: float | None = None
    pitch: float | None = None

    def as_dict(self) -> JsonObject:
        value: JsonObject = {
            "frame": self.frame,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }
        if self.yaw is not None:
            value["yaw"] = self.yaw
        if self.pitch is not None:
            value["pitch"] = self.pitch
        return value


@dataclass(frozen=True, slots=True)
class NavGoal:
    goal_id: str
    instruction: str
    modality: str = "language"
    public: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.goal_id:
            raise ValueError("goal_id must not be empty")
        if not self.instruction:
            raise ValueError("instruction must not be empty")


@dataclass(frozen=True, slots=True)
class NavTask:
    """Public task data visible to an agent.

    For a compound task this contains only the current goal. Future goals stay in
    the environment/benchmark adapter and are revealed by ``nav.goal.finish``.
    """

    task_id: str
    goal: NavGoal
    scene_id: str | None = None
    public: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")

    @property
    def instruction(self) -> str:
        return self.goal.instruction


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    source_time: float
    received_time: float
    frame: str
    channels: Mapping[str, Any]
    pose: Pose | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> JsonObject:
        value: JsonObject = {
            "observation_id": self.observation_id,
            "source_time": self.source_time,
            "received_time": self.received_time,
            "frame": self.frame,
            "channels": dict(self.channels),
            "extras": dict(self.extras),
        }
        if self.pose is not None:
            value["pose"] = self.pose.as_dict()
        return value

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MotionProfile:
    tool: str
    actions: frozenset[str] = frozenset()
    frame: str | None = None
    units: str | None = None
    forward_m: float | None = None
    turn_deg: float | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "tool": self.tool,
            "actions": sorted(self.actions),
        }
        for name in ("frame", "units", "forward_m", "turn_deg"):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        return value


@dataclass(frozen=True, slots=True)
class NavigationProfile:
    observation_channels: frozenset[str]
    motion: MotionProfile
    camera: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_channels": sorted(self.observation_channels),
            "motion": self.motion.as_dict(),
            "camera": dict(self.camera),
        }

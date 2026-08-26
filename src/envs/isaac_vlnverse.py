from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benches.base import BenchmarkCase
from envs.isaac import IsaacNavigationEnvironment
from harness.config import load_symbol


def from_case(
    case: BenchmarkCase,
    *,
    session_factory: str,
    session_params: Mapping[str, Any] | None = None,
    flash: bool = True,
    **adapter_params: Any,
) -> IsaacNavigationEnvironment:
    factory = load_symbol(session_factory)

    def build(private_case: BenchmarkCase):
        return factory(private_case, **dict(session_params or {}))

    controller = "move_by_flash" if flash else "move_by_discrete"
    actions: dict[str, Mapping[str, Any]] = {
        "stand_still": {"h1": {"stand_still": []}},
        "forward": {"h1": {controller: [1]}},
        "turn_left": {"h1": {controller: [2]}},
        "turn_right": {"h1": {controller: [3]}},
    }
    adapter_params.setdefault(
        "camera", {"height": 480, "width": 640, "hfov_deg": 79, "pitch_deg": -30}
    )
    return IsaacNavigationEnvironment(
        case,
        session_factory=build,
        native_actions=actions,
        warmup_action={"h1": {"stand_still": []}},
        goal_finish_action={"h1": {"stop": []}},
        **adapter_params,
    )

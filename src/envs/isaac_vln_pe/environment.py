from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from envs.isaac import IsaacNavigationEnvironment
from harness.config import load_symbol
from schemas import EnvironmentEpisode


def from_episode(
    episode: EnvironmentEpisode,
    *,
    session_factory: str = "envs.internutopia:create_vln_pe_session",
    session_params: Mapping[str, Any] | None = None,
    **adapter_params: Any,
) -> IsaacNavigationEnvironment:
    factory = load_symbol(session_factory)

    def build(private_episode: EnvironmentEpisode):
        return factory(private_episode, **dict(session_params or {}))

    actions: dict[str, Mapping[str, Any]] = {
        "stand_still": {"h1": {"stand_still": []}},
        "forward": {"h1": {"move_by_discrete": [1]}},
        "turn_left": {"h1": {"move_by_discrete": [2]}},
        "turn_right": {"h1": {"move_by_discrete": [3]}},
    }
    adapter_params.setdefault(
        "camera", {"height": 480, "width": 640, "hfov_deg": 79, "pitch_deg": -30}
    )
    return IsaacNavigationEnvironment(
        episode,
        session_factory=build,
        native_actions=actions,
        warmup_action={"h1": {"stand_still": []}},
        goal_finish_action={"h1": {"stop": []}},
        **adapter_params,
    )

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from vln.rpc import RPCVLNNavigator


class DualVLNNavigator(RPCVLNNavigator):
    model_name = "dualvln"

    def __init__(
        self,
        command: Sequence[str],
        *,
        upstream_root: str | Path,
        checkpoint: str | Path,
        motion_tool: str = "nav.move.trajectory",
        motion_requirements: Mapping[str, object] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            command,
            upstream_root=upstream_root,
            checkpoint=checkpoint,
            **kwargs,
        )
        self.required_tools = frozenset({"nav.observe", motion_tool})
        self.requirements = {
            "observation_channels": ["rgb", "depth"],
            "motion": {"tool": motion_tool, **dict(motion_requirements or {})},
        }

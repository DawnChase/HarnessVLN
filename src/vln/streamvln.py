from __future__ import annotations

from vln.rpc import RPCVLNNavigator


class StreamVLNNavigator(RPCVLNNavigator):
    model_name = "streamvln"
    required_tools = frozenset({"nav.observe", "nav.move.discrete"})
    requirements = {
        "observation_channels": ["rgb", "depth", "pose", "camera_intrinsics"],
        "motion": {
            "tool": "nav.move.discrete",
            "actions": ["forward", "turn_left", "turn_right"],
            "forward_m": 0.25,
            "turn_deg": 15.0,
        },
    }

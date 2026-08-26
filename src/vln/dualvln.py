from __future__ import annotations

from vln.rpc import RPCVLNNavigator


class DualVLNNavigator(RPCVLNNavigator):
    model_name = "dualvln"
    required_tools = frozenset({"nav.observe", "nav.move.discrete"})
    requirements = {
        "observation_channels": ["rgb", "depth"],
        "motion": {
            "tool": "nav.move.discrete",
            "actions": ["stand_still", "forward", "turn_left", "turn_right"],
            "forward_m": 0.25,
            "turn_deg": 15.0,
        },
        "camera": {"height": 480, "width": 640, "hfov_deg": 79, "pitch_deg": -30},
    }

from __future__ import annotations

import os
import time
from pathlib import Path

from vln.worker import run_worker


class FixtureBackend:
    model_name = "sdk-fixture"

    def load(self, hello):
        assert hello["checkpoint"]
        assert hello["options"]["max_steps"] == 8

    def navigate(self, instruction, options, tools, cancelled):
        del options
        while not cancelled.is_set():
            observation = tools.observe()
            rgb = observation["channels"].get("rgb")
            if rgb is not None:
                assert rgb.shape == (2, 2, 3)
                assert int(rgb[1, 1, 2]) == 11
            delta = observation["channels"]["target_delta"]
            if delta == 0:
                return f"completed: {instruction}"
            tools.move_discrete("forward" if delta > 0 else "backward")
        return "cancelled"

    def close(self):
        if os.environ.get("HARNESS_CLOSE_ERROR"):
            raise RuntimeError("fixture backend close failed")
        marker = os.environ.get("HARNESS_CLOSE_MARKER")
        if marker:
            time.sleep(0.25)
            Path(marker).write_text("closed", encoding="utf-8")


run_worker(FixtureBackend())

from __future__ import annotations

from vln.worker import run_worker


class FixtureBackend:
    model_name = "sdk-fixture"

    def load(self, hello):
        assert hello["checkpoint"]

    def navigate(self, instruction, options, tools, cancelled):
        del options
        while not cancelled.is_set():
            observation = tools.observe()
            delta = observation["channels"]["target_delta"]
            if delta == 0:
                return f"completed: {instruction}"
            tools.move_discrete("forward" if delta > 0 else "backward")
        return "cancelled"

    def close(self):
        pass


run_worker(FixtureBackend())

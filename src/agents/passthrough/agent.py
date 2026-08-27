from __future__ import annotations

from agents._jobs import run_vln_job
from harness.runtime import NavContext


class PassthroughVLNAgent:
    """Pass each benchmark goal unchanged to one complete VLN job."""

    required_tools = frozenset(
        {
            "vln.navigate.start",
            "vln.navigate.status",
            "vln.navigate.cancel",
            "nav.goal.finish",
        }
    )

    def __init__(self, *, poll_period_s: float = 0.01) -> None:
        self.poll_period_s = poll_period_s

    async def run(self, context: NavContext) -> None:
        context.output.record(
            {
                "agent": type(self).__name__,
                "mode": "vln_passthrough",
                "poll_period_s": self.poll_period_s,
                "required_tools": sorted(self.required_tools),
            }
        )
        instruction = context.task.instruction
        while True:
            status = await run_vln_job(
                context, instruction, poll_period_s=self.poll_period_s
            )
            if status["state"] != "succeeded":
                await context.nav.stop("failed", status.get("reason", "VLN job failed"))
                return
            transition = await context.nav.finish_goal("completed", status.get("reason", ""))
            if transition["done"]:
                await context.nav.stop("completed", "all navigation goals completed")
                return
            instruction = transition["goal"]["instruction"]

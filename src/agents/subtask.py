from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from agents._jobs import run_vln_job
from harness.runtime import NavContext


@dataclass(frozen=True, slots=True)
class PlanStep:
    instruction: str


Decomposer = Callable[[str], Sequence[PlanStep] | Awaitable[Sequence[PlanStep]]]


class SentenceDecomposer:
    """Deterministic baseline; an LLM decomposer can implement the same callable."""

    def __call__(self, instruction: str) -> Sequence[PlanStep]:
        parts = [part.strip() for part in re.split(r"\bthen\b|[.;]", instruction) if part.strip()]
        return tuple(PlanStep(part) for part in parts) or (PlanStep(instruction),)


class SubtaskNavigationAgent:
    def __init__(
        self,
        decomposer: Decomposer | None = None,
        *,
        use_memory: bool = True,
        poll_period_s: float = 0.01,
    ) -> None:
        self.decomposer = decomposer or SentenceDecomposer()
        self.use_memory = use_memory
        self.poll_period_s = poll_period_s
        tools = {
            "vln.navigate.start",
            "vln.navigate.status",
            "vln.navigate.cancel",
            "nav.goal.finish",
        }
        if use_memory:
            tools.update({"spatial.search", "spatial.remember"})
        self.required_tools = frozenset(tools)

    async def run(self, context: NavContext) -> None:
        instruction = context.task.instruction
        while True:
            steps = self.decomposer(instruction)
            if inspect.isawaitable(steps):
                steps = await steps
            if not steps:
                await context.nav.stop("failed", "decomposer returned no plan steps")
                return
            for step in steps:
                if self.use_memory:
                    await context.spatial.search(step.instruction, top_k=3)
                status = await run_vln_job(
                    context, step.instruction, poll_period_s=self.poll_period_s
                )
                if status["state"] != "succeeded":
                    await context.nav.stop("failed", status.get("reason", "VLN job failed"))
                    return
                if self.use_memory:
                    await context.spatial.remember(
                        step.instruction,
                        frame=str(context.task.public.get("memory_frame", "world")),
                    )
            transition = await context.nav.finish_goal("completed", "plan completed")
            if transition["done"]:
                await context.nav.stop("completed", "all navigation goals completed")
                return
            instruction = transition["goal"]["instruction"]

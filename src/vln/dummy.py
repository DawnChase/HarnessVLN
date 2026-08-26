from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from harness.errors import HarnessError
from harness.tool_bus import Tool, ToolClient
from schemas import NavTask


@dataclass(slots=True)
class _Job:
    job_id: str
    instruction: str
    state: str = "running"
    steps: int = 0
    reason: str = ""
    task: asyncio.Task[None] | None = None


class DummyVLNNavigator:
    """A complete asynchronous VLN job, not a one-step policy."""

    required_tools = frozenset({"nav.observe", "nav.move.discrete"})
    requirements = {
        "observation_channels": ["target_delta", "pose"],
        "motion": {
            "tool": "nav.move.discrete",
            "actions": ["forward", "backward"],
            "frame": "dummy_world",
            "units": "meters_degrees",
            "forward_m": 1.0,
        },
    }

    def __init__(self, *, max_steps: int = 100, inference_period_s: float = 0.0) -> None:
        self.max_steps = max_steps
        self.inference_period_s = inference_period_s
        self._tools: ToolClient | None = None
        self._jobs: dict[str, _Job] = {}
        self._closed = False

    async def start(self, task: NavTask, tools: ToolClient):
        del task
        self._tools = tools
        return (
            Tool(
                "vln.navigate.start",
                "Start a complete VLN navigation job.",
                {
                    "type": "object",
                    "properties": {
                        "instruction": {"type": "string", "minLength": 1},
                        "options": {"type": "object"},
                    },
                    "required": ["instruction", "options"],
                    "additionalProperties": False,
                },
                self._start_job,
                writes=True,
            ),
            Tool(
                "vln.navigate.status",
                "Read the state of a VLN navigation job.",
                {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
                self._status_job,
            ),
            Tool(
                "vln.navigate.cancel",
                "Cancel a VLN job without ending the complete task.",
                {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                    "additionalProperties": False,
                },
                self._cancel_job,
                writes=True,
            ),
        )

    async def _start_job(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        if self._closed:
            raise HarnessError("navigator is stopped")
        job_id = uuid.uuid4().hex
        job = _Job(job_id, arguments["instruction"])
        self._jobs[job_id] = job
        job.task = asyncio.create_task(self._run_job(job), name=f"dummy-vln-{job_id}")
        return {"job_id": job_id}

    async def _run_job(self, job: _Job) -> None:
        assert self._tools is not None
        try:
            while job.steps < self.max_steps:
                observation = await self._tools.call("nav.observe")
                delta = int(observation["channels"]["target_delta"])
                if delta == 0:
                    job.state = "succeeded"
                    job.reason = "target reached"
                    return
                action = "forward" if delta > 0 else "backward"
                await self._tools.call("nav.move.discrete", action=action)
                job.steps += 1
                await asyncio.sleep(self.inference_period_s)
            job.state = "failed"
            job.reason = "maximum VLN steps reached"
        except asyncio.CancelledError:
            job.state = "cancelled"
            job.reason = "job cancelled"
            raise
        except Exception as error:
            job.state = "failed"
            job.reason = f"{type(error).__name__}: {error}"

    async def _status_job(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        job = self._get_job(arguments["job_id"])
        return {
            "job_id": job.job_id,
            "state": job.state,
            "steps": job.steps,
            "reason": job.reason,
        }

    async def _cancel_job(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        job = self._get_job(arguments["job_id"])
        if job.task is not None and not job.task.done():
            job.state = "cancelled"
            job.reason = "job cancelled"
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        return {"job_id": job.job_id, "state": job.state}

    def _get_job(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as error:
            raise HarnessError(f"unknown VLN job: {job_id}") from error

    async def stop(self, reason: str) -> None:
        del reason
        self._closed = True
        tasks = []
        for job in self._jobs.values():
            if job.task is not None and not job.task.done():
                job.state = "cancelled"
                job.reason = "navigator stopped"
                job.task.cancel()
                tasks.append(job.task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

from __future__ import annotations

import asyncio

from harness.runtime import NavContext


async def run_vln_job(
    context: NavContext,
    instruction: str,
    *,
    poll_period_s: float,
) -> dict:
    job_id = await context.vln.start(instruction)
    try:
        while True:
            status = await context.vln.status(job_id)
            if status["state"] != "running":
                return status
            if context.cancelled.is_set():
                await context.vln.cancel(job_id)
                return await context.vln.status(job_id)
            await asyncio.sleep(poll_period_s)
    except asyncio.CancelledError:
        await asyncio.shield(context.vln.cancel(job_id))
        raise

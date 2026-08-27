from __future__ import annotations

import asyncio

from harness.runtime import NavContext


async def run_vln_job(
    context: NavContext,
    instruction: str,
    *,
    poll_period_s: float,
) -> dict:
    if context.cancelled.is_set():
        raise asyncio.CancelledError()
    job_id = await context.vln.start(instruction)
    try:
        while True:
            if context.cancelled.is_set():
                raise asyncio.CancelledError()
            status = await context.vln.status(job_id)
            if context.cancelled.is_set():
                raise asyncio.CancelledError()
            if status["state"] != "running":
                return status
            await asyncio.sleep(poll_period_s)
    except asyncio.CancelledError as cancellation:
        if context.cancelled.is_set():
            raise
        try:
            await asyncio.shield(context.vln.cancel(job_id))
        except BaseException as cleanup_error:
            raise cancellation from cleanup_error
        raise

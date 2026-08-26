from __future__ import annotations

import io
import threading
import time

import pytest

from vln.worker import WorkerRuntime


class Backend:
    model_name = "fixture"

    def load(self, hello):
        pass

    def navigate(self, instruction, options, tools, cancelled):
        return None

    def close(self):
        pass


def test_shutdown_wakes_worker_thread_blocked_on_tool_result() -> None:
    runtime = WorkerRuntime(Backend())
    runtime._writer = io.StringIO()
    errors = []

    def call_tool():
        try:
            runtime.call_tool("job", "nav.observe", {})
        except Exception as error:
            errors.append(str(error))

    thread = threading.Thread(target=call_tool)
    thread.start()
    deadline = time.monotonic() + 1
    while not runtime._tool_results and time.monotonic() < deadline:
        time.sleep(0.001)

    runtime._cancel_all()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert errors == ["worker is shutting down"]


def test_cancelled_job_keeps_worker_exclusive_until_thread_exits() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingBackend(Backend):
        def navigate(self, instruction, options, tools, cancelled):
            started.set()
            assert release.wait(timeout=1)
            return "backend returned after cancellation"

    runtime = WorkerRuntime(BlockingBackend())
    first = runtime._start_job({"instruction": "first", "options": {}})
    assert started.wait(timeout=1)

    cancelling = runtime._cancel_job({"job_id": first["job_id"]})
    assert cancelling["state"] == "cancelling"
    assert cancelling["reason"] == "cancel requested"
    with pytest.raises(RuntimeError, match="already has a running navigation job"):
        runtime._start_job({"instruction": "second", "options": {}})

    release.set()
    thread = runtime._job_threads[first["job_id"]]
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert runtime._job_status({"job_id": first["job_id"]}) == {
        "job_id": first["job_id"],
        "state": "cancelled",
        "reason": "backend returned after cancellation",
    }

    second = runtime._start_job({"instruction": "second", "options": {}})
    runtime._job_threads[second["job_id"]].join(timeout=1)
    assert runtime._job_status({"job_id": second["job_id"]})["state"] == "succeeded"

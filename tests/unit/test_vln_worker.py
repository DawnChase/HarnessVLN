from __future__ import annotations

import io
import threading
import time

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

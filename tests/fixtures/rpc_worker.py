from __future__ import annotations

import json
import sys


def send(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


if "--bad-stdout" in sys.argv:
    print("model log leaked to protocol stdout", flush=True)

jobs = {}
for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "tool_result":
        continue
    request_id = message["id"]
    method = message["method"]
    params = message["params"]
    if method == "hello":
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": True,
                "result": {
                    "protocol": params["protocol"],
                    "model": params["model"],
                },
            }
        )
    elif method == "probe_tool":
        call_id = "probe-call"
        send(
            {
                "type": "tool_call",
                "id": call_id,
                "name": params["name"],
                "arguments": params.get("arguments", {}),
            }
        )
        while True:
            tool_result = json.loads(sys.stdin.readline())
            if tool_result.get("type") == "tool_result" and tool_result.get("id") == call_id:
                break
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": tool_result["ok"],
                "result": tool_result.get("result"),
                "error": tool_result.get("error"),
            }
        )
    elif method == "navigate.start":
        jobs["job-1"] = {"job_id": "job-1", "state": "succeeded", "reason": "done"}
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": True,
                "result": {"job_id": "job-1"},
            }
        )
    elif method == "navigate.status":
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": True,
                "result": jobs[params["job_id"]],
            }
        )
    elif method == "navigate.cancel":
        job = jobs[params["job_id"]]
        job["state"] = "cancelled"
        send({"type": "response", "id": request_id, "ok": True, "result": job})
    elif method == "shutdown":
        send({"type": "response", "id": request_id, "ok": True, "result": {}})
        break
    else:
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": False,
                "error": f"unknown method: {method}",
            }
        )

from __future__ import annotations

import json
import os
import signal
import socket
import sys
import time


protocol_socket = socket.socket(fileno=int(os.environ["HARNESS_VLN_RPC_FD"]))
protocol_reader = protocol_socket.makefile("r", encoding="utf-8")
protocol_writer = protocol_socket.makefile("w", encoding="utf-8", buffering=1)


def send(value):
    protocol_writer.write(json.dumps(value, separators=(",", ":")) + "\n")
    protocol_writer.flush()


if "--print-logs" in sys.argv:
    print("model log leaked to protocol stdout", flush=True)
    print("model diagnostic", file=sys.stderr, flush=True)
if "--bad-protocol" in sys.argv:
    protocol_writer.write("not-json\n")
    protocol_writer.flush()
if "--ignore-shutdown" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

jobs = {}
next_job = 1
dropped_start = False
malformed_start = False
for line in protocol_reader:
    message = json.loads(line)
    if message["type"] == "tool_result":
        continue
    request_id = message["id"]
    method = message["method"]
    params = message["params"]
    if method == "hello":
        if "--delay-hello" in sys.argv:
            time.sleep(1)
        result = {
            "protocol": params["protocol"],
            "model": "wrong-model" if "--wrong-model" in sys.argv else params["model"],
        }
        if params["protocol"] >= 2:
            result["capabilities"] = ["navigate.release"]
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": True,
                "result": result,
            }
        )
        if "--hang-after-hello" in sys.argv:
            while True:
                time.sleep(1)
    elif method == "probe_tool":
        call_id = "probe-call"
        send(
            {
                "type": "tool_call",
                "id": call_id,
                "job_id": "probe-job",
                "name": params["name"],
                "arguments": params.get("arguments", {}),
            }
        )
        while True:
            tool_result = json.loads(protocol_reader.readline())
            if (
                tool_result.get("type") == "tool_result"
                and tool_result.get("id") == call_id
            ):
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
        job_id = f"job-{next_job}"
        next_job += 1
        jobs[job_id] = {"job_id": job_id, "state": "succeeded", "reason": "done"}
        if "--drop-first-start" in sys.argv and not dropped_start:
            dropped_start = True
            continue
        if "--malformed-first-start" in sys.argv and not malformed_start:
            malformed_start = True
            send(
                {
                    "type": "response",
                    "id": request_id,
                    "ok": True,
                    "result": {"job_id": 123},
                }
            )
            continue
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": True,
                "result": {"job_id": job_id},
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
    elif method == "navigate.release":
        job_id = params["job_id"]
        jobs.pop(job_id)
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": True,
                "result": {"job_id": job_id},
            }
        )
        if "--late-tool-after-release" in sys.argv:
            send(
                {
                    "type": "tool_call",
                    "id": f"late-{job_id}",
                    "job_id": job_id,
                    "name": "nav.observe",
                    "arguments": {},
                }
            )
    elif method == "shutdown":
        if "--ignore-shutdown" in sys.argv:
            continue
        send({"type": "response", "id": request_id, "ok": True, "result": {}})
        break
    elif method == "slow":
        time.sleep(0.1)
        send({"type": "response", "id": request_id, "ok": True, "result": "late"})
    elif method == "ping":
        send({"type": "response", "id": request_id, "ok": True, "result": "pong"})
    else:
        send(
            {
                "type": "response",
                "id": request_id,
                "ok": False,
                "error": f"unknown method: {method}",
            }
        )

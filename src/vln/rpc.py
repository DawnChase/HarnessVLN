from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.errors import HarnessError
from harness.tool_bus import Tool, ToolClient
from schemas import NavTask


class RPCError(HarnessError):
    pass


class JsonLineProcess:
    """Bidirectional JSONL transport with reverse tool calls from a worker."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        request_timeout_s: float = 30.0,
    ) -> None:
        if not command:
            raise ValueError("worker command must not be empty")
        self.command = tuple(command)
        self.cwd = Path(cwd).resolve() if cwd else None
        self.env = dict(env or {})
        self.request_timeout_s = request_timeout_s
        self.stderr_tail: deque[str] = deque(maxlen=100)
        self._process: asyncio.subprocess.Process | None = None
        self._tools: ToolClient | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._write_lock = asyncio.Lock()
        self._closed = False

    async def start(self, tools: ToolClient, hello: Mapping[str, Any]) -> dict[str, Any]:
        if self._process is not None:
            raise RPCError("worker is already started")
        environment = os.environ.copy()
        environment.update(self.env)
        self._tools = tools
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise RPCError(f"failed to start worker {self.command[0]}: {error}") from error
        self._reader_task = asyncio.create_task(self._read_stdout(), name="vln-rpc-reader")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="vln-rpc-stderr")
        response = await self.request("hello", dict(hello))
        if not isinstance(response, dict):
            raise RPCError("worker hello response must be an object")
        return response

    async def request(self, method: str, params: Mapping[str, Any]) -> Any:
        if self._closed or self._process is None:
            raise RPCError("worker is not active")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "type": "request",
                    "id": request_id,
                    "method": method,
                    "params": dict(params),
                }
            )
            return await asyncio.wait_for(future, timeout=self.request_timeout_s)
        except asyncio.TimeoutError as error:
            raise RPCError(f"worker request timed out: {method}") from error
        finally:
            self._pending.pop(request_id, None)

    async def close(self) -> None:
        if self._closed:
            return
        process = self._process
        if process is not None and process.returncode is None:
            try:
                await self.request("shutdown", {})
            except (RPCError, asyncio.CancelledError):
                pass
        self._closed = True
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        for tool_task in tuple(self._tool_tasks):
            tool_task.cancel()
        if self._tool_tasks:
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
        for stream_task in (self._reader_task, self._stderr_task):
            if stream_task is not None and not stream_task.done():
                stream_task.cancel()
        tasks = [
            stream_task
            for stream_task in (self._reader_task, self._stderr_task)
            if stream_task is not None
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._fail_pending(RPCError("worker closed"))

    async def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            while line := await self._process.stdout.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RPCError(
                        "worker stdout must contain only JSONL protocol messages"
                    ) from error
                await self._dispatch(message)
            returncode = await self._process.wait()
            raise RPCError(
                f"worker exited with code {returncode}; stderr: {self._stderr_summary()}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._fail_pending(error if isinstance(error, RPCError) else RPCError(str(error)))

    async def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            raise RPCError("worker message must be an object")
        message_type = message.get("type")
        if message_type == "response":
            request_id = message.get("id")
            if not isinstance(request_id, str):
                raise RPCError("worker response id must be a string")
            future = self._pending.get(request_id)
            if future is None or future.done():
                raise RPCError(f"response for unknown request: {request_id}")
            if message.get("ok") is True:
                future.set_result(message.get("result"))
            else:
                future.set_exception(RPCError(str(message.get("error", "worker error"))))
            return
        if message_type == "tool_call":
            task = asyncio.create_task(self._handle_tool_call(message), name="vln-tool-call")
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)
            return
        raise RPCError(f"unknown worker message type: {message_type!r}")

    async def _handle_tool_call(self, message: dict[str, Any]) -> None:
        call_id = message.get("id")
        name = message.get("name")
        arguments = message.get("arguments", {})
        if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(
            arguments, dict
        ):
            raise RPCError("malformed worker tool_call")
        assert self._tools is not None
        try:
            result = await self._tools.call(name, arguments)
            response = {"type": "tool_result", "id": call_id, "ok": True, "result": result}
        except Exception as error:
            response = {
                "type": "tool_result",
                "id": call_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        await self._send(response)

    async def _send(self, message: Mapping[str, Any]) -> None:
        assert self._process is not None and self._process.stdin is not None
        try:
            payload = json.dumps(message, separators=(",", ":")).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            raise RPCError(f"protocol value is not JSON serializable: {error}") from error
        async with self._write_lock:
            self._process.stdin.write(payload)
            try:
                await self._process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as error:
                raise RPCError(f"worker pipe closed; stderr: {self._stderr_summary()}") from error

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        while line := await self._process.stderr.readline():
            self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    def _stderr_summary(self) -> str:
        return " | ".join(self.stderr_tail) or "<empty>"

    def _fail_pending(self, error: BaseException) -> None:
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)


class RPCVLNNavigator:
    protocol_version = 1
    model_name = "rpc"
    required_tools: frozenset[str] = frozenset()
    requirements: dict[str, Any] = {}

    def __init__(
        self,
        command: Sequence[str],
        *,
        upstream_root: str | Path,
        checkpoint: str | Path,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        request_timeout_s: float = 300.0,
    ) -> None:
        self.command = tuple(command)
        self.upstream_root = Path(upstream_root)
        self.checkpoint = Path(checkpoint)
        self.cwd = Path(cwd) if cwd else self.upstream_root
        self.env = dict(env or {})
        self.request_timeout_s = request_timeout_s
        self._task: NavTask | None = None
        self._process: JsonLineProcess | None = None

    async def start(self, task: NavTask, tools: ToolClient):
        self._validate_resources()
        self._task = task
        self._process = JsonLineProcess(
            self.command,
            cwd=self.cwd,
            env=self.env,
            request_timeout_s=self.request_timeout_s,
        )
        hello = await self._process.start(
            tools,
            {
                "protocol": self.protocol_version,
                "model": self.model_name,
                "upstream_root": str(self.upstream_root.resolve()),
                "checkpoint": str(self.checkpoint.resolve()),
            },
        )
        if hello.get("protocol") != self.protocol_version:
            raise RPCError(f"worker protocol mismatch: {hello!r}")
        if hello.get("model") != self.model_name:
            raise RPCError(f"worker model mismatch: {hello!r}")
        return (
            Tool(
                "vln.navigate.start",
                "Start a complete external VLN navigation job.",
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
                "Get external VLN job state.",
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
                "Cancel an external VLN job.",
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

    async def _start_job(self, actor: str, arguments: dict[str, Any]) -> Any:
        del actor
        assert self._process is not None and self._task is not None
        return await self._process.request(
            "navigate.start",
            {
                "task_id": self._task.task_id,
                "instruction": arguments["instruction"],
                "options": arguments["options"],
            },
        )

    async def _status_job(self, actor: str, arguments: dict[str, Any]) -> Any:
        del actor
        assert self._process is not None
        return await self._process.request("navigate.status", arguments)

    async def _cancel_job(self, actor: str, arguments: dict[str, Any]) -> Any:
        del actor
        assert self._process is not None
        return await self._process.request("navigate.cancel", arguments)

    async def stop(self, reason: str) -> None:
        del reason
        if self._process is not None:
            await self._process.close()

    def _validate_resources(self) -> None:
        if not self.upstream_root.is_dir():
            raise HarnessError(f"{self.model_name} upstream root not found: {self.upstream_root}")
        if not self.checkpoint.exists():
            raise HarnessError(f"{self.model_name} checkpoint not found: {self.checkpoint}")

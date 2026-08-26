from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.errors import HarnessError
from harness.media import FileArrayStore
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
        self.stdout_tail: deque[str] = deque(maxlen=100)
        self.stderr_tail: deque[str] = deque(maxlen=100)
        self._process: asyncio.subprocess.Process | None = None
        self._tools: ToolClient | None = None
        self._pending: dict[str, asyncio.Future[Any]] = {}
        self._expired_ids: set[str] = set()
        self._expired_order: deque[str] = deque()
        self._reader_task: asyncio.Task[None] | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._protocol_reader: asyncio.StreamReader | None = None
        self._protocol_writer: asyncio.StreamWriter | None = None
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._job_tool_tasks: dict[str, set[asyncio.Task[None]]] = {}
        self._closed_jobs: set[str] = set()
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._failure: RPCError | None = None
        self._media = FileArrayStore()

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process is not None else None

    async def start(self, tools: ToolClient, hello: Mapping[str, Any]) -> dict[str, Any]:
        if self._process is not None:
            raise RPCError("worker is already started")
        environment = os.environ.copy()
        environment.update(self.env)
        self._tools = tools
        parent_socket, child_socket = socket.socketpair()
        parent_socket.setblocking(False)
        child_socket.set_inheritable(True)
        environment["HARNESS_VLN_RPC_FD"] = str(child_socket.fileno())
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                pass_fds=(child_socket.fileno(),),
                start_new_session=True,
            )
        except OSError as error:
            parent_socket.close()
            child_socket.close()
            raise RPCError(f"failed to start worker {self.command[0]}: {error}") from error
        finally:
            child_socket.close()
        self._protocol_reader, self._protocol_writer = await asyncio.open_connection(
            sock=parent_socket
        )
        self._reader_task = asyncio.create_task(
            self._read_protocol(), name="vln-rpc-reader"
        )
        self._stdout_task = asyncio.create_task(
            self._read_log("stdout"), name="vln-rpc-stdout"
        )
        self._stderr_task = asyncio.create_task(
            self._read_log("stderr"), name="vln-rpc-stderr"
        )
        try:
            response = await self.request("hello", dict(hello))
        except BaseException:
            await self.close()
            raise
        if not isinstance(response, dict):
            raise RPCError("worker hello response must be an object")
        return response

    async def request(self, method: str, params: Mapping[str, Any]) -> Any:
        if self._closed or self._process is None:
            raise RPCError("worker is not active")
        if self._failure is not None:
            raise self._failure
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
            self._remember_expired(request_id)
            raise RPCError(f"worker request timed out: {method}") from error
        except asyncio.CancelledError:
            self._remember_expired(request_id)
            raise
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
            self._signal_process_group(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._signal_process_group(process, signal.SIGKILL)
                await process.wait()
        for tool_task in tuple(self._tool_tasks):
            tool_task.cancel()
        if self._tool_tasks:
            await asyncio.gather(*self._tool_tasks, return_exceptions=True)
        if self._protocol_writer is not None:
            self._protocol_writer.close()
            try:
                await self._protocol_writer.wait_closed()
            except (BrokenPipeError, ConnectionResetError):
                pass
        tasks = [
            stream_task
            for stream_task in (self._reader_task, self._stdout_task, self._stderr_task)
            if stream_task is not None
        ]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=1.0
                )
            except asyncio.TimeoutError:
                for stream_task in tasks:
                    if not stream_task.done():
                        stream_task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        self._fail_pending(RPCError("worker closed"))
        self._media.close()

    async def close_job(self, job_id: str) -> None:
        self._closed_jobs.add(job_id)
        tasks = tuple(self._job_tool_tasks.get(job_id, ()))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_protocol(self) -> None:
        assert self._process is not None and self._protocol_reader is not None
        try:
            while line := await self._protocol_reader.readline():
                try:
                    message = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RPCError("worker sent invalid JSONL protocol data") from error
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
            if request_id in self._expired_ids:
                self._expired_ids.discard(request_id)
                return
            future = self._pending.get(request_id)
            if future is None or future.done():
                raise RPCError(f"response for unknown request: {request_id}")
            if message.get("ok") is True:
                future.set_result(message.get("result"))
            else:
                future.set_exception(RPCError(str(message.get("error", "worker error"))))
            return
        if message_type == "tool_call":
            job_id = message.get("job_id")
            if not isinstance(job_id, str):
                raise RPCError("worker tool_call job_id must be a string")
            if job_id in self._closed_jobs:
                raise RPCError(f"tool_call arrived after job closed: {job_id}")
            task = asyncio.create_task(self._handle_tool_call(message), name="vln-tool-call")
            self._tool_tasks.add(task)
            self._job_tool_tasks.setdefault(job_id, set()).add(task)

            def discard(done: asyncio.Task[None]) -> None:
                self._tool_tasks.discard(done)
                job_tasks = self._job_tool_tasks.get(job_id)
                if job_tasks is not None:
                    job_tasks.discard(done)
                    if not job_tasks:
                        self._job_tool_tasks.pop(job_id, None)

            task.add_done_callback(discard)
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
        assert self._process is not None and self._protocol_writer is not None
        try:
            payload = json.dumps(
                self._media.encode(message), separators=(",", ":")
            ).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as error:
            raise RPCError(f"protocol value is not JSON serializable: {error}") from error
        async with self._write_lock:
            self._protocol_writer.write(payload)
            try:
                await self._protocol_writer.drain()
            except (BrokenPipeError, ConnectionResetError) as error:
                raise RPCError(f"worker pipe closed; stderr: {self._stderr_summary()}") from error

    async def _read_log(self, stream_name: str) -> None:
        assert self._process is not None
        stream = getattr(self._process, stream_name)
        assert stream is not None
        target = self.stdout_tail if stream_name == "stdout" else self.stderr_tail
        while line := await stream.readline():
            target.append(line.decode("utf-8", errors="replace").rstrip())

    def _stderr_summary(self) -> str:
        return " | ".join(self.stderr_tail) or "<empty>"

    def _fail_pending(self, error: BaseException) -> None:
        if isinstance(error, RPCError):
            self._failure = error
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(error)

    def _remember_expired(self, request_id: str) -> None:
        if len(self._expired_order) >= 1024:
            oldest = self._expired_order.popleft()
            self._expired_ids.discard(oldest)
        self._expired_order.append(request_id)
        self._expired_ids.add(request_id)

    @staticmethod
    def _signal_process_group(
        process: asyncio.subprocess.Process, process_signal: signal.Signals
    ) -> None:
        try:
            os.killpg(process.pid, process_signal)
        except ProcessLookupError:
            pass


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
        try:
            hello = await self._process.start(
                tools,
                {
                    "protocol": self.protocol_version,
                    "model": self.model_name,
                    "upstream_root": str(self.upstream_root.resolve()),
                    "checkpoint": str(self.checkpoint.resolve()),
                },
            )
        except BaseException:
            await self._process.close()
            self._process = None
            raise
        if (
            hello.get("protocol") != self.protocol_version
            or hello.get("model") != self.model_name
        ):
            await self._process.close()
            self._process = None
            raise RPCError(f"worker handshake mismatch: {hello!r}")
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
        result = await self._process.request("navigate.cancel", arguments)
        await self._process.close_job(arguments["job_id"])
        return result

    async def stop(self, reason: str) -> None:
        del reason
        if self._process is not None:
            await self._process.close()

    def _validate_resources(self) -> None:
        if not self.upstream_root.is_dir():
            raise HarnessError(f"{self.model_name} upstream root not found: {self.upstream_root}")
        if not self.checkpoint.exists():
            raise HarnessError(f"{self.model_name} checkpoint not found: {self.checkpoint}")

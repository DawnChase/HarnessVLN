from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from harness.errors import (
    DuplicateToolError,
    MissingToolError,
    ToolClosedError,
    ToolValidationError,
)


JsonObject = dict[str, Any]
ToolHandler = Callable[[str, JsonObject], Awaitable[Any]]


def _audit_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value if not isinstance(value, str) or len(value) <= 512 else value[:509] + "..."
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {str(key): _audit_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return {"type": type(value).__name__, "size": len(value)}
        return [_audit_value(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": list(shape)}
    return {"type": type(value).__name__}


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: JsonObject
    handler: ToolHandler
    output_schema: JsonObject | None = None
    writes: bool = False

    def __post_init__(self) -> None:
        if not self.name or any(character.isspace() for character in self.name):
            raise ValueError("tool name must be non-empty and contain no whitespace")
        try:
            Draft202012Validator.check_schema(self.input_schema)
            if self.output_schema is not None:
                Draft202012Validator.check_schema(self.output_schema)
        except SchemaError as error:
            raise ValueError(f"invalid schema for tool {self.name}: {error.message}") from error


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Provider-neutral description exposed to an agent model."""

    name: str
    description: str
    input_schema: JsonObject


@dataclass(frozen=True, slots=True)
class ToolEvent:
    sequence: int
    monotonic_time: float
    actor: str
    name: str
    arguments: Mapping[str, Any]
    outcome: str
    error: str | None = None


class ToolBus:
    """Single dispatch path for typed calls and model function calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._writes_open = True
        self._events: list[ToolEvent] = []
        self._next_sequence = 1
        self._active_writes: set[asyncio.Task[Any]] = set()
        self._writes_drained = asyncio.Event()
        self._writes_drained.set()

    @property
    def audit(self) -> tuple[ToolEvent, ...]:
        return tuple(self._events)

    def register(self, tools: Sequence[Tool]) -> None:
        names = [tool.name for tool in tools]
        duplicate_batch = sorted({name for name in names if names.count(name) > 1})
        duplicate_existing = sorted(set(names) & self._tools.keys())
        duplicates = duplicate_batch + duplicate_existing
        if duplicates:
            raise DuplicateToolError(f"duplicate tools: {sorted(set(duplicates))}")
        self._tools.update((tool.name, tool) for tool in tools)

    def require(self, owner: str, names: set[str] | frozenset[str]) -> None:
        missing = sorted(set(names) - self._tools.keys())
        if missing:
            raise MissingToolError(f"{owner} requires unavailable tools: {missing}")

    def client(
        self, actor: str, allowed: set[str] | frozenset[str] | None = None
    ) -> ToolClient:
        return ToolClient(self, actor, None if allowed is None else frozenset(allowed))

    def close_writes(self) -> None:
        self._writes_open = False

    async def drain_writes(self) -> None:
        await self._writes_drained.wait()

    def specs(self, allowed: frozenset[str] | None = None) -> tuple[ToolSpec, ...]:
        names = self._tools.keys() if allowed is None else self._tools.keys() & allowed
        return tuple(
            ToolSpec(
                name=self._tools[name].name,
                description=self._tools[name].description,
                input_schema=self._tools[name].input_schema,
            )
            for name in sorted(names)
        )

    async def call(
        self,
        actor: str,
        name: str,
        arguments: JsonObject,
        allowed: frozenset[str] | None,
    ) -> Any:
        sequence = self._next_sequence
        self._next_sequence += 1
        started_at = time.monotonic()
        if name not in self._tools or (allowed is not None and name not in allowed):
            self._record(sequence, started_at, actor, name, arguments, "denied")
            raise MissingToolError(f"{actor} cannot call tool: {name}")
        tool = self._tools[name]
        if tool.writes and not self._writes_open:
            self._record(
                sequence, started_at, actor, name, arguments, "closed", "writes are closed"
            )
            raise ToolClosedError(f"write tools are closed: {name}")
        try:
            self._validate(name, tool.input_schema, arguments, "input")
        except ToolValidationError:
            self._record(sequence, started_at, actor, name, arguments, "invalid")
            raise

        current = asyncio.current_task()
        if tool.writes and current is not None:
            self._active_writes.add(current)
            self._writes_drained.clear()
        try:
            result = await tool.handler(actor, arguments)
            if tool.output_schema is not None:
                self._validate(name, tool.output_schema, result, "output")
        except BaseException as error:
            self._record(
                sequence,
                started_at,
                actor,
                name,
                arguments,
                "error",
                type(error).__name__,
            )
            raise
        else:
            self._record(sequence, started_at, actor, name, arguments, "ok")
            return result
        finally:
            if tool.writes and current is not None:
                self._active_writes.discard(current)
                if not self._active_writes:
                    self._writes_drained.set()

    def _record(
        self,
        sequence: int,
        started_at: float,
        actor: str,
        name: str,
        arguments: JsonObject,
        outcome: str,
        error: str | None = None,
    ) -> None:
        self._events.append(
            ToolEvent(
                sequence,
                started_at,
                actor,
                name,
                _audit_value(arguments),
                outcome,
                error,
            )
        )

    @staticmethod
    def _validate(name: str, schema: JsonObject, value: Any, direction: str) -> None:
        try:
            Draft202012Validator(schema).validate(value)
        except ValidationError as error:
            location = ".".join(str(part) for part in error.absolute_path) or "<root>"
            raise ToolValidationError(
                f"{name} {direction} at {location}: {error.message}"
            ) from error


class ToolClient:
    def __init__(
        self,
        bus: ToolBus,
        actor: str,
        allowed: frozenset[str] | None,
    ) -> None:
        self._bus = bus
        self.actor = actor
        self._allowed = allowed

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return self._bus.specs(self._allowed)

    async def call(
        self, name: str, arguments: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> Any:
        payload = dict(arguments or {})
        payload.update(kwargs)
        return await self._bus.call(self.actor, name, payload, self._allowed)

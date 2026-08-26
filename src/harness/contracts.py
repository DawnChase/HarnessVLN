from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from harness.tool_bus import Tool, ToolClient
from schemas import NavTask


if TYPE_CHECKING:
    from harness.runtime import NavContext


JsonObject = dict[str, Any]


class NavigationAgent(Protocol):
    required_tools: frozenset[str]

    async def run(self, context: "NavContext") -> None: ...


class Environment(Protocol):
    async def start(self, task: NavTask) -> Sequence[Tool]: ...

    async def stop(self, reason: str) -> None: ...

    async def wait_failure(self) -> str: ...

    def result(self) -> JsonObject: ...


class VLNNavigator(Protocol):
    required_tools: frozenset[str]

    async def start(self, task: NavTask, tools: ToolClient) -> Sequence[Tool]: ...

    async def stop(self, reason: str) -> None: ...


class SpatialMemory(Protocol):
    required_tools: frozenset[str]

    async def start(self, task: NavTask, tools: ToolClient) -> Sequence[Tool]: ...

    async def stop(self, reason: str) -> None: ...


@dataclass(slots=True)
class NavigationStack:
    agent: NavigationAgent
    environment: Environment
    vln: VLNNavigator | None = None
    memory: SpatialMemory | None = None

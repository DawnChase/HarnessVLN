from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from harness.errors import HarnessError
from harness.tool_bus import Tool, ToolClient
from schemas import EnvironmentTerminal, NavigationProfile, NavTask


if TYPE_CHECKING:
    from harness.runtime import NavContext


JsonObject = dict[str, Any]


class NavigationAgent(Protocol):
    required_tools: frozenset[str]

    async def run(self, context: "NavContext") -> None: ...


class Environment(Protocol):
    profile: NavigationProfile

    async def start(self, task: NavTask) -> Sequence[Tool]: ...

    async def stop(self, reason: str) -> None: ...

    async def wait_terminal(self) -> EnvironmentTerminal: ...

    def result(self) -> JsonObject: ...


class VLNNavigator(Protocol):
    required_tools: frozenset[str]
    requirements: dict[str, Any]

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

    def __post_init__(self) -> None:
        _require_methods("agent", self.agent, ("run",))
        _require_tools("agent", self.agent)
        _require_methods(
            "environment",
            self.environment,
            ("start", "stop", "wait_terminal", "result"),
        )
        if not isinstance(getattr(self.environment, "profile", None), NavigationProfile):
            raise HarnessError(
                f"environment {type(self.environment).__name__} must declare "
                "a NavigationProfile"
            )
        if self.vln is not None:
            _require_methods("vln", self.vln, ("start", "stop"))
            tools = _require_tools("vln", self.vln)
            if "nav.stop" in tools:
                raise HarnessError("VLNNavigator cannot require agent-owned nav.stop")
            if not isinstance(getattr(self.vln, "requirements", None), Mapping):
                raise HarnessError(
                    f"vln {type(self.vln).__name__} requirements must be a mapping"
                )
        if self.memory is not None:
            _require_methods("memory", self.memory, ("start", "stop"))
            _require_tools("memory", self.memory)


def _require_methods(role: str, component: Any, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not callable(getattr(component, name, None))]
    if missing:
        raise HarnessError(
            f"{role} {type(component).__name__} must implement callable methods: "
            f"{', '.join(missing)}"
        )


def _require_tools(role: str, component: Any) -> frozenset[str]:
    tools = getattr(component, "required_tools", None)
    if not isinstance(tools, frozenset) or any(
        not isinstance(name, str) or not name or any(char.isspace() for char in name)
        for name in tools
    ):
        raise HarnessError(
            f"{role} {type(component).__name__} required_tools must be "
            "a frozenset of non-empty tool names"
        )
    return tools

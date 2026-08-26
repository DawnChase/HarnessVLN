from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from harness.errors import HarnessError
from harness.tool_bus import Tool, ToolClient
from schemas import NavTask


class DummyLandmarkMemory:
    required_tools: frozenset[str] = frozenset()

    def __init__(self, root: str | Path, *, writeback: bool = True) -> None:
        self.root = Path(root)
        self.writeback = writeback
        self.path = self.root / "landmarks.json"
        self._task_id: str | None = None
        self._items: list[dict[str, Any]] = []
        self._started = False
        self._stopped = False

    async def start(self, task: NavTask, tools: ToolClient):
        del tools
        if self._started:
            raise HarnessError("memory instances are single-use")
        self._started = True
        self._task_id = task.task_id
        if self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, list):
                raise HarnessError(f"invalid landmark file: {self.path}")
            self._items = value
        return (
            Tool(
                "spatial.remember",
                "Store a navigation landmark.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1},
                        "frame": {"type": "string", "minLength": 1},
                        "pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 3,
                        },
                    },
                    "required": ["text", "frame"],
                    "additionalProperties": False,
                },
                self._remember,
                writes=True,
            ),
            Tool(
                "spatial.search",
                "Search stored navigation landmarks.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "frame": {"type": "string"},
                        "near_pose": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 3,
                        },
                        "top_k": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query", "top_k"],
                    "additionalProperties": False,
                },
                self._search,
            ),
        )

    async def _remember(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        self._ensure_open()
        identifier = self._next_id()
        item: dict[str, Any] = {
            "id": identifier,
            "source_task_id": self._task_id,
            "frame": arguments["frame"],
            "text": arguments["text"],
        }
        if "pose" in arguments:
            item["pose"] = list(arguments["pose"])
        self._items.append(item)
        return dict(item)

    async def _search(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del actor
        self._ensure_open()
        query = arguments["query"].casefold()
        frame = arguments.get("frame")
        near_pose = arguments.get("near_pose")
        matches = [
            item
            for item in self._items
            if (not query or query in item["text"].casefold())
            and (frame is None or item["frame"] == frame)
        ]

        def key(item: dict[str, Any]) -> tuple[float, str]:
            if near_pose is None or "pose" not in item or item["frame"] != frame:
                return math.inf, item["id"]
            pose = item["pose"]
            return math.hypot(pose[0] - near_pose[0], pose[1] - near_pose[1]), item["id"]

        matches.sort(key=key)
        return {"items": [dict(item) for item in matches[: arguments["top_k"]]]}

    async def stop(self, reason: str) -> None:
        del reason
        if not self._started or self._stopped:
            return
        self._stopped = True
        if self.writeback:
            self._write_atomic()

    def _write_atomic(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=self.root, prefix=".landmarks-", suffix=".json.tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self._items, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _next_id(self) -> str:
        numeric = []
        for item in self._items:
            identifier = str(item.get("id", ""))
            if identifier.startswith("landmark-") and identifier[9:].isdigit():
                numeric.append(int(identifier[9:]))
        return f"landmark-{max(numeric, default=0) + 1:08d}"

    def _ensure_open(self) -> None:
        if not self._started or self._stopped:
            raise HarnessError("memory is not active")

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from harness.runtime import NavContext
from harness.tool_bus import ToolSpec


DEFAULT_INSTRUCTIONS = """You are the decision core of a navigation agent.
Keep control of the task by calling exactly one tool at a time. You may observe and
move directly, delegate a complete navigation job to a VLN tool, or use spatial
memory. Finish each current goal with the goal-finish tool. Finish the complete task
only with the stop tool. Do not end a turn with plain text."""


class NormalAgent:
    """Minimal Responses API agent loop over Harness navigation tools."""

    def __init__(
        self,
        model: str,
        tools: Sequence[str],
        *,
        instructions: str = DEFAULT_INSTRUCTIONS,
        max_iterations: int = 80,
        client: Any | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if not tools:
            raise ValueError("tools must not be empty")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self.model = model
        self.instructions = instructions
        self.max_iterations = max_iterations
        self.required_tools = frozenset(tools)
        self._client = client

    async def run(self, context: NavContext) -> None:
        context.output.record(
            {
                "agent": type(self).__name__,
                "mode": "free_agent_loop",
                "model": self.model,
                "max_iterations": self.max_iterations,
                "required_tools": sorted(self.required_tools),
            }
        )
        model_tools, tool_names = _responses_tools(context.tools.specs)
        input_items: list[Any] = [
            {
                "role": "user",
                "content": "Navigate this task:\n" + _json(_task_data(context)),
            }
        ]

        for _ in range(self.max_iterations):
            if context.cancelled.is_set():
                return
            response = await self._responses().create(
                model=self.model,
                instructions=self.instructions,
                input=input_items,
                tools=model_tools,
                tool_choice="required",
                parallel_tool_calls=False,
            )
            output = list(response.output)
            input_items.extend(output)
            calls = [item for item in output if item.type == "function_call"]
            if len(calls) != 1:
                await context.nav.stop(
                    "failed", f"model returned {len(calls)} tool calls"
                )
                return

            call = calls[0]
            canonical_name: str | None = tool_names.get(call.name)
            try:
                if canonical_name is None:
                    raise ValueError(f"unknown model tool: {call.name}")
                arguments = json.loads(call.arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
                result = await context.tools.call(canonical_name, arguments)
                tool_output = {"ok": True, "result": result}
            except Exception as error:
                tool_output = {
                    "ok": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": _json(tool_output),
                }
            )
            if canonical_name == "nav.stop" and tool_output["ok"]:
                return

        await context.nav.stop("failed", "agent iteration budget reached")

    def _responses(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        responses = getattr(self._client, "responses", None)
        if responses is None:
            raise RuntimeError("the installed OpenAI SDK does not expose Responses API")
        return responses


def _responses_tools(
    specs: Sequence[ToolSpec],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tools: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for spec in specs:
        model_name = spec.name.replace(".", "__")
        if model_name in names:
            raise ValueError(f"tool name collision after provider mapping: {spec.name}")
        names[model_name] = spec.name
        tools.append(
            {
                "type": "function",
                "name": model_name,
                "description": spec.description,
                "parameters": spec.input_schema,
                "strict": False,
            }
        )
    return tools, names


def _task_data(context: NavContext) -> dict[str, Any]:
    task = context.task
    return {
        "task_id": task.task_id,
        "scene_id": task.scene_id,
        "goal": {
            "goal_id": task.goal.goal_id,
            "instruction": task.goal.instruction,
            "modality": task.goal.modality,
            "public": dict(task.goal.public),
        },
        "public": dict(task.public),
    }


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    shape = getattr(value, "shape", None)
    if shape is not None:
        size = getattr(value, "size", None)
        tolist = getattr(value, "tolist", None)
        if isinstance(size, int) and size <= 64 and callable(tolist):
            return tolist()
        return {
            "type": "array",
            "shape": [int(item) for item in shape],
            "dtype": str(getattr(value, "dtype", "unknown")),
        }
    return {"type": type(value).__name__}

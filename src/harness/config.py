from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from harness.errors import HarnessError


CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "benchmark": {"$ref": "#/$defs/component"},
        "stack": {
            "type": "object",
            "properties": {
                "agent": {"$ref": "#/$defs/component"},
                "environment": {"$ref": "#/$defs/component"},
                "vln": {"anyOf": [{"$ref": "#/$defs/component"}, {"type": "null"}]},
                "memory": {
                    "anyOf": [{"$ref": "#/$defs/component"}, {"type": "null"}]
                },
            },
            "required": ["agent", "environment"],
            "additionalProperties": False,
        },
        "runner": {
            "type": "object",
            "properties": {
                "parallelism": {"type": "integer", "minimum": 1},
                "max_cases": {"type": "integer", "minimum": 1},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0},
                "shutdown_timeout_s": {"type": "number", "exclusiveMinimum": 0},
                "seed": {"type": "integer"},
            },
            "additionalProperties": False,
        },
        "output": {"type": "object"},
        "provenance": {"type": "object"},
    },
    "required": ["benchmark", "stack", "runner"],
    "additionalProperties": False,
    "$defs": {
        "component": {
            "type": "object",
            "properties": {
                "factory": {"type": "string", "pattern": "^[^:]+:[^:]+$"},
                "params": {"type": "object"},
                "serial": {"type": "boolean"},
            },
            "required": ["factory"],
            "additionalProperties": False,
        }
    },
}


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    data: dict[str, Any]
    sources: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    factory: str
    params: Mapping[str, Any]
    serial: bool = False

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ComponentSpec":
        return cls(
            str(value["factory"]),
            dict(value.get("params", {})),
            bool(value.get("serial", False)),
        )

    def create(self, **runtime: Any) -> Any:
        target = load_symbol(self.factory)
        kwargs = dict(self.params)
        overlap = sorted(kwargs.keys() & runtime.keys())
        if overlap:
            raise HarnessError(
                f"factory {self.factory} has duplicate configured/runtime arguments: {overlap}"
            )
        kwargs.update(runtime)
        try:
            return target(**kwargs)
        except Exception as error:
            raise HarnessError(
                f"factory {self.factory} failed: {type(error).__name__}: {error}"
            ) from error


def load_config(paths: Sequence[str | Path]) -> ResolvedConfig:
    if not paths:
        raise HarnessError("at least one YAML config path is required")
    merged: dict[str, Any] = {}
    sources: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise HarnessError(f"failed to load config {path}: {error}") from error
        if not isinstance(value, Mapping):
            raise HarnessError(f"config {path} must contain a mapping")
        merged = overlay(merged, value)
        sources.append(str(path))
    try:
        Draft202012Validator(CONFIG_SCHEMA).validate(merged)
    except ValidationError as error:
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise HarnessError(f"invalid config at {location}: {error.message}") from error
    canonical = json.dumps(merged, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ResolvedConfig(merged, tuple(sources), digest)


def overlay(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        previous = result.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            result[key] = (
                overlay({}, value)
                if _changes_factory(previous, value)
                else overlay(previous, value)
            )
        else:
            result[key] = value
    return result


def _changes_factory(
    previous: Mapping[str, Any], replacement: Mapping[str, Any]
) -> bool:
    old_factory = previous.get("factory")
    new_factory = replacement.get("factory")
    return (
        isinstance(old_factory, str)
        and isinstance(new_factory, str)
        and old_factory != new_factory
    )


def load_symbol(uri: str) -> Any:
    if uri.count(":") != 1:
        raise HarnessError(f"factory must use module:object syntax: {uri}")
    module_name, object_path = uri.split(":")
    try:
        value: Any = importlib.import_module(module_name)
        for part in object_path.split("."):
            value = getattr(value, part)
    except (ImportError, AttributeError) as error:
        raise HarnessError(f"cannot load factory {uri}: {error}") from error
    if not callable(value):
        raise HarnessError(f"factory is not callable: {uri}")
    if inspect.isabstract(value):
        raise HarnessError(f"factory is abstract: {uri}")
    return value

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from harness.errors import HarnessError


def _component_schema(
    extra_properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "factory": {"type": "string", "pattern": "^[^:]+:[^:]+$"},
            "params": {"type": "object"},
            "serial": {"type": "boolean"},
            "scope": {"enum": ["task", "run"]},
            **dict(extra_properties or {}),
        },
        "required": ["factory"],
        "additionalProperties": False,
    }


_REFERENCE = {"type": ["string", "null"]}

AGENT_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "agent": _component_schema(
            {"vln": _REFERENCE, "memory": _REFERENCE}
        ),
        "provenance": {"type": "object"},
    },
    "required": ["agent"],
    "additionalProperties": False,
}

ENVIRONMENT_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "environment": _component_schema(),
        "interactive": {"type": "object"},
        "provenance": {"type": "object"},
    },
    "required": ["environment"],
    "additionalProperties": False,
}

VLN_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "vln": _component_schema(),
        "provenance": {"type": "object"},
    },
    "required": ["vln"],
    "additionalProperties": False,
}

MEMORY_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "memory": _component_schema(),
        "provenance": {"type": "object"},
    },
    "required": ["memory"],
    "additionalProperties": False,
}

BENCHMARK_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "benchmark": _component_schema(
            {"environment": {"type": "string", "minLength": 1}}
        ),
        "provenance": {"type": "object"},
    },
    "required": ["benchmark"],
    "additionalProperties": False,
}

RUNNER_CONFIG_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "runner": {
            "type": "object",
            "properties": {
                "benches": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "bench_parallelism": {"type": "integer", "minimum": 1},
                "task_parallelism": {"type": "integer", "minimum": 1},
                "max_cases": {"type": "integer", "minimum": 1},
                "timeout_s": {"type": "number", "exclusiveMinimum": 0},
                "shutdown_timeout_s": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                },
                "seed": {"type": "integer"},
            },
            "required": ["benches"],
            "additionalProperties": False,
        },
        "output": {"type": "object"},
        "provenance": {"type": "object"},
    },
    "required": ["runner"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    component: "ComponentSpec"
    interactive: Mapping[str, Any]
    data: Mapping[str, Any]
    sources: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class AgentConfig:
    core: "ComponentSpec"
    vln: "ComponentSpec | None"
    memory: "ComponentSpec | None"
    data: Mapping[str, Any]
    sources: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    benchmark: "ComponentSpec"
    environment: EnvironmentConfig
    data: Mapping[str, Any]
    sources: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    settings: Mapping[str, Any]
    benches: tuple[BenchmarkConfig, ...]
    output: Mapping[str, Any]
    data: Mapping[str, Any]
    sources: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class _ConfigDocument:
    path: Path
    data: dict[str, Any]
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    factory: str
    params: Mapping[str, Any]
    serial: bool = False
    scope: str = "task"

    def __post_init__(self) -> None:
        if self.scope not in {"task", "run"}:
            raise ValueError(f"invalid component scope: {self.scope}")

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "ComponentSpec":
        return cls(
            str(value["factory"]),
            dict(value.get("params", {})),
            bool(value.get("serial", False)),
            str(value.get("scope", "task")),
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


def load_agent_config(path: str | Path) -> AgentConfig:
    document = _load_config_document(path, "agent")
    agent_value = document.data["agent"]
    component_value = _without_keys(agent_value, "vln", "memory")
    vln_document = _load_optional_reference(agent_value.get("vln"), "vln")
    memory_document = _load_optional_reference(
        agent_value.get("memory"), "memory"
    )
    vln_value = vln_document.data["vln"] if vln_document else None
    memory_value = memory_document.data["memory"] if memory_document else None
    provenance = _merged_provenance(
        *(item for item in (vln_document, memory_document, document) if item)
    )
    data = {
        "agent": component_value,
        "vln": vln_value,
        "memory": memory_value,
        "provenance": provenance,
    }
    sources = _ordered_sources(
        document,
        *(item for item in (vln_document, memory_document) if item),
    )
    core = ComponentSpec.from_config(component_value)
    vln = ComponentSpec.from_config(vln_value) if vln_value else None
    memory = ComponentSpec.from_config(memory_value) if memory_value else None
    _require_task_scope("agent", core)
    if memory is not None:
        _require_task_scope("memory", memory)
    return AgentConfig(
        core=core,
        vln=vln,
        memory=memory,
        data=data,
        sources=sources,
        digest=_config_digest(data),
    )


def load_environment_config(path: str | Path) -> EnvironmentConfig:
    document = _load_config_document(path, "environment")
    data = dict(document.data)
    component = ComponentSpec.from_config(data["environment"])
    _require_task_scope("environment", component)
    return EnvironmentConfig(
        component=component,
        interactive=dict(data.get("interactive", {})),
        data=data,
        sources=document.sources,
        digest=_config_digest(data),
    )


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    document = _load_config_document(path, "benchmark")
    benchmark_value = document.data["benchmark"]
    environment = load_environment_config(benchmark_value["environment"])
    component_value = _without_keys(benchmark_value, "environment")
    provenance = overlay(
        environment.data.get("provenance", {}),
        document.data.get("provenance", {}),
    )
    data = {
        "benchmark": component_value,
        "environment": dict(environment.data),
        "provenance": provenance,
    }
    sources = _ordered_sources(document, environment)
    benchmark = ComponentSpec.from_config(component_value)
    _require_task_scope("benchmark", benchmark)
    return BenchmarkConfig(
        benchmark=benchmark,
        environment=environment,
        data=data,
        sources=sources,
        digest=_config_digest(data),
    )


def load_runner_config(path: str | Path) -> RunnerConfig:
    document = _load_config_document(path, "runner")
    runner_value = document.data["runner"]
    benches = tuple(
        load_benchmark_config(reference) for reference in runner_value["benches"]
    )
    settings = _without_keys(runner_value, "benches")
    provenance: dict[str, Any] = {}
    for bench in benches:
        provenance = overlay(provenance, bench.data.get("provenance", {}))
    provenance = overlay(provenance, document.data.get("provenance", {}))
    output = dict(document.data.get("output", {}))
    data = {
        "runner": {**settings, "benches": [dict(bench.data) for bench in benches]},
        "output": output,
        "provenance": provenance,
    }
    sources = _ordered_sources(document, *benches)
    return RunnerConfig(
        settings=settings,
        benches=benches,
        output=output,
        data=data,
        sources=sources,
        digest=_config_digest(data),
    )


_CONFIG_SCHEMAS = {
    "agent": AGENT_CONFIG_SCHEMA,
    "environment": ENVIRONMENT_CONFIG_SCHEMA,
    "vln": VLN_CONFIG_SCHEMA,
    "memory": MEMORY_CONFIG_SCHEMA,
    "benchmark": BENCHMARK_CONFIG_SCHEMA,
    "runner": RUNNER_CONFIG_SCHEMA,
}


def _load_optional_reference(
    value: Any, kind: str
) -> _ConfigDocument | None:
    if value is None:
        return None
    return _load_config_document(str(value), kind)


def _load_config_document(
    raw_path: str | Path,
    kind: str,
    trail: tuple[Path, ...] = (),
) -> _ConfigDocument:
    path = Path(raw_path).expanduser().resolve()
    if path in trail:
        cycle = " -> ".join(str(item) for item in (*trail, path))
        raise HarnessError(f"config extends cycle: {cycle}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HarnessError(f"failed to load {kind} config {path}: {error}") from error
    if not isinstance(loaded, Mapping):
        raise HarnessError(f"{kind} config {path} must contain a mapping")
    local = dict(loaded)
    extends = local.pop("extends", None)
    local = _normalize_references(kind, local, path)
    sources: tuple[str, ...]
    if extends is None:
        data = local
        sources = (str(path),)
    else:
        if not isinstance(extends, str) or not extends:
            raise HarnessError(f"{kind} config {path} extends must be a path string")
        base = _load_config_document(
            _reference_path(path, extends), kind, (*trail, path)
        )
        data = overlay(base.data, local)
        sources = (*base.sources, str(path))
    try:
        Draft202012Validator(_CONFIG_SCHEMAS[kind]).validate(data)
    except ValidationError as error:
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise HarnessError(
            f"invalid {kind} config {path} at {location}: {error.message}"
        ) from error
    return _ConfigDocument(path, data, tuple(dict.fromkeys(sources)))


def _normalize_references(
    kind: str, data: Mapping[str, Any], owner: Path
) -> dict[str, Any]:
    value = dict(data)
    if kind == "agent" and isinstance(value.get("agent"), Mapping):
        component = dict(value["agent"])
        for name in ("vln", "memory"):
            reference = component.get(name)
            if isinstance(reference, str):
                component[name] = str(_reference_path(owner, reference))
        value["agent"] = component
    elif kind == "benchmark" and isinstance(value.get("benchmark"), Mapping):
        component = dict(value["benchmark"])
        reference = component.get("environment")
        if isinstance(reference, str):
            component["environment"] = str(_reference_path(owner, reference))
        value["benchmark"] = component
    elif kind == "runner" and isinstance(value.get("runner"), Mapping):
        runner = dict(value["runner"])
        references = runner.get("benches")
        if isinstance(references, list):
            runner["benches"] = [
                str(_reference_path(owner, reference))
                if isinstance(reference, str)
                else reference
                for reference in references
            ]
        value["runner"] = runner
    return value


def _reference_path(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve()


def _without_keys(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    excluded = frozenset(keys)
    return {key: item for key, item in value.items() if key not in excluded}


def _require_task_scope(name: str, component: ComponentSpec) -> None:
    if component.scope != "task":
        raise HarnessError(f"{name} scope must be task")


def _merged_provenance(*documents: _ConfigDocument) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for document in documents:
        value = overlay(value, document.data.get("provenance", {}))
    return value


def _ordered_sources(*configs: Any) -> tuple[str, ...]:
    sources: list[str] = []
    for config in configs:
        for source in config.sources:
            if source not in sources:
                sources.append(source)
    return tuple(sources)


def _config_digest(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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

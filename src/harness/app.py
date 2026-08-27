from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.config import (
    AgentConfig,
    BenchmarkConfig,
    ComponentSpec,
    EnvironmentConfig,
    RunnerConfig,
    load_agent_config,
    load_environment_config,
    load_runner_config,
    overlay,
)
from harness.contracts import NavigationStack
from harness.errors import HarnessError
from harness.runner import BenchRunner, CaseRecord, RunSummary, SuiteSummary
from harness.runtime import NavigationHarness, NavigationResult
from schemas import EnvironmentEpisode, NavGoal, NavTask


@dataclass(slots=True)
class ConfiguredStackFactory:
    agent: ComponentSpec
    environment: ComponentSpec
    vln: ComponentSpec | None = None
    memory: ComponentSpec | None = None
    _run_vln: Any = field(init=False, default=None, repr=False)
    _close_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        for name, spec in (
            ("agent", self.agent),
            ("environment", self.environment),
            ("memory", self.memory),
        ):
            if spec is not None and spec.scope != "task":
                raise HarnessError(f"run-scoped {name} components are not supported")

    @property
    def requires_serial(self) -> bool:
        components = (self.agent, self.environment, self.vln, self.memory)
        return any(
            spec and (spec.serial or spec.scope == "run") for spec in components
        ) or bool(self.memory and self.memory.params.get("writeback", True))

    def __call__(self, episode: EnvironmentEpisode) -> NavigationStack:
        if self._close_task is not None:
            raise HarnessError("run-scoped VLN close is still in progress or failed")
        navigator = self._run_vln
        if self.vln is not None and self.vln.scope == "run" and navigator is None:
            navigator = self.vln.create()
            enable = getattr(navigator, "enable_run_scope", None)
            close = getattr(navigator, "close_run", None)
            if not callable(enable) or not callable(close):
                raise HarnessError(
                    "run-scoped VLN must implement enable_run_scope() and close_run()"
                )
            enable()
            self._run_vln = navigator
        return NavigationStack(
            agent=self.agent.create(),
            environment=self.environment.create(episode=episode),
            vln=(
                navigator
                if navigator is not None
                else (self.vln.create() if self.vln else None)
            ),
            memory=self.memory.create() if self.memory else None,
        )

    async def close_run(self) -> None:
        navigator = self._run_vln
        if navigator is None:
            return
        if self._close_task is not None and self._close_task.done():
            try:
                close_error = self._close_task.exception()
            except asyncio.CancelledError:
                close_error = asyncio.CancelledError()
            if close_error is not None:
                self._close_task = None
        if self._close_task is None:

            async def close_navigator() -> None:
                result = navigator.close_run()
                if inspect.isawaitable(result):
                    await result

            self._close_task = asyncio.create_task(
                close_navigator(), name="run-scoped-vln-close"
            )
        close_task = self._close_task
        try:
            await asyncio.shield(close_task)
        except asyncio.CancelledError as cancellation:
            try:
                await asyncio.shield(close_task)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            if self._run_vln is navigator:
                self._run_vln = None
                self._close_task = None
            raise cancellation
        if self._run_vln is navigator:
            self._run_vln = None
            self._close_task = None


@dataclass(slots=True)
class InteractiveNavigationSession:
    environment: EnvironmentConfig
    agent: AgentConfig
    harness: NavigationHarness = field(default_factory=NavigationHarness)
    _factory: ConfiguredStackFactory = field(init=False, repr=False)
    _session_id: str = field(
        init=False, default_factory=lambda: uuid.uuid4().hex[:12], repr=False
    )
    _task_index: int = field(init=False, default=0, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._factory = _stack_factory(self.agent, self.environment)

    @classmethod
    def from_configs(
        cls, environment_path: str | Path, agent_path: str | Path
    ) -> "InteractiveNavigationSession":
        return cls(
            load_environment_config(environment_path),
            load_agent_config(agent_path),
        )

    async def navigate(self, instruction: str) -> NavigationResult:
        if self._closed:
            raise HarnessError("interactive navigation session is closed")
        instruction = instruction.strip()
        if not instruction:
            raise HarnessError("navigation instruction must not be empty")
        self._task_index += 1
        task_id = f"interactive:{self._session_id}:{self._task_index}"
        options = self.environment.interactive
        goal = NavGoal(
            f"{task_id}:goal:0",
            instruction,
            str(options.get("goal_modality", "language")),
            dict(options.get("goal_public", {})),
        )
        task = NavTask(
            task_id,
            goal,
            scene_id=options.get("scene_id"),
            public=dict(options.get("task_public", {})),
        )
        episode = EnvironmentEpisode(task, dict(options.get("setup", {})))
        return await self.harness.run_task(task, self._factory(episode))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._factory.close_run()


async def run_config(
    runner_path: str | Path, agent_path: str | Path
) -> tuple[SuiteSummary, Path]:
    runner_config = load_runner_config(runner_path)
    agent_config = load_agent_config(agent_path)
    settings = runner_config.settings
    harness = NavigationHarness(
        timeout_s=float(settings.get("timeout_s", 300)),
        shutdown_timeout_s=float(settings.get("shutdown_timeout_s", 10)),
    )
    bench_parallelism = int(settings.get("bench_parallelism", 1))
    semaphore = asyncio.Semaphore(bench_parallelism)

    async def run_bench(index: int) -> tuple[int, RunSummary]:
        config = runner_config.benches[index]
        async with semaphore:
            benchmark: Any | None = None
            try:
                benchmark = config.benchmark.create()
                if not _is_benchmark(benchmark):
                    raise HarnessError(
                        "configured benchmark does not implement the Benchmark contract"
                    )
                summary = await BenchRunner(harness).run(
                    benchmark,
                    _stack_factory(agent_config, config.environment),
                    parallelism=int(settings.get("task_parallelism", 1)),
                    max_cases=settings.get("max_cases"),
                )
            except Exception as error:
                summary = _failed_benchmark(config, benchmark, error)
            return index, summary

    indexed = await asyncio.gather(
        *(run_bench(index) for index in range(len(runner_config.benches)))
    )
    indexed.sort(key=lambda item: item[0])
    suite = SuiteSummary(tuple(summary for _, summary in indexed))
    output_root = Path(runner_config.output.get("root", "runs/latest"))
    manifest_path = write_manifest(output_root, runner_config, agent_config, suite)
    return suite, manifest_path


def write_manifest(
    output_root: Path,
    runner_config: RunnerConfig,
    agent_config: AgentConfig,
    summary: SuiteSummary,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[float]] = {}
    benchmarks = []
    for run in summary.runs:
        records = []
        run_metrics: dict[str, list[float]] = {}
        for record in run.records:
            for name, value in record.metrics.items():
                numeric = float(value)
                metrics.setdefault(name, []).append(numeric)
                run_metrics.setdefault(name, []).append(numeric)
            records.append(_record_document(record))
        benchmarks.append(
            {
                "name": run.benchmark,
                "split": run.split,
                "validation_status": run.validation_status,
                "error": run.error,
                "cleanup_errors": list(run.cleanup_errors),
                "aggregate_metrics": _aggregate_metrics(run_metrics),
                "records": records,
            }
        )
    resolved_data = {
        "runner": dict(runner_config.data),
        "agent": dict(agent_config.data),
    }
    sources = tuple(dict.fromkeys((*runner_config.sources, *agent_config.sources)))
    canonical = json.dumps(
        resolved_data, sort_keys=True, separators=(",", ":"), default=_json_default
    )
    document = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "config": resolved_data,
        "config_sources": list(sources),
        "config_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "provenance": overlay(
            runner_config.data.get("provenance", {}),
            agent_config.data.get("provenance", {}),
        ),
        "aggregate_metrics": _aggregate_metrics(metrics),
        "benchmarks": benchmarks,
    }
    target = output_root / "manifest.json"
    descriptor, temporary = tempfile.mkstemp(
        dir=output_root, prefix=".manifest-", suffix=".json.tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _stack_factory(
    agent: AgentConfig, environment: EnvironmentConfig
) -> ConfiguredStackFactory:
    return ConfiguredStackFactory(
        agent=agent.core,
        environment=environment.component,
        vln=agent.vln,
        memory=agent.memory,
    )


def _is_benchmark(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("name", "split", "validation_status", "cases", "score")
    )


def _failed_benchmark(
    config: BenchmarkConfig, benchmark: Any | None, error: Exception
) -> RunSummary:
    name = getattr(benchmark, "name", config.benchmark.factory)
    split = getattr(benchmark, "split", config.benchmark.params.get("split", ""))
    validation_status = getattr(benchmark, "validation_status", "unavailable")
    cleanup_errors = tuple(getattr(error, "_harness_cleanup_errors", ()))
    return RunSummary(
        benchmark=str(name),
        split=str(split),
        validation_status=str(validation_status),
        records=(),
        error=f"{type(error).__name__}: {error}",
        cleanup_errors=cleanup_errors,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return {"type": type(value).__name__}


def _record_document(record: CaseRecord) -> dict[str, Any]:
    return {
        "index": record.index,
        "case_id": record.case_id,
        "error": record.error,
        "metrics": dict(record.metrics),
        "execution_id": record.result.execution_id if record.result else None,
        "terminal": asdict(record.result.terminal) if record.result else None,
        "environment": record.result.environment if record.result else None,
        "cleanup_errors": (
            list(record.result.cleanup_errors) if record.result else []
        ),
        "audit": (
            [asdict(event) for event in record.result.audit]
            if record.result
            else []
        ),
    }


def _aggregate_metrics(values: Mapping[str, list[float]]) -> dict[str, float]:
    return {
        name: sum(items) / len(items)
        for name, items in values.items()
        if items
    }


def run_config_sync(
    runner_path: str | Path, agent_path: str | Path
) -> tuple[SuiteSummary, Path]:
    return asyncio.run(run_config(runner_path, agent_path))

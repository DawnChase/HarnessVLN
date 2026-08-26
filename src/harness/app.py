from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benches.base import Benchmark, BenchmarkCase
from harness.config import ComponentSpec, ResolvedConfig, load_config
from harness.contracts import NavigationStack
from harness.errors import HarnessError
from harness.runner import BenchRunner, RunSummary
from harness.runtime import NavigationHarness


@dataclass(slots=True)
class ConfiguredStackFactory:
    agent: ComponentSpec
    environment: ComponentSpec
    vln: ComponentSpec | None = None
    memory: ComponentSpec | None = None

    @property
    def requires_serial(self) -> bool:
        return bool(self.memory and self.memory.params.get("writeback", True))

    def __call__(self, case: BenchmarkCase) -> NavigationStack:
        return NavigationStack(
            agent=self.agent.create(),
            environment=self.environment.create(case=case),
            vln=self.vln.create() if self.vln else None,
            memory=self.memory.create() if self.memory else None,
        )


async def run_config(paths: Sequence[str | Path]) -> tuple[RunSummary, Path]:
    resolved = load_config(paths)
    data = resolved.data
    benchmark = ComponentSpec.from_config(data["benchmark"]).create()
    if not _is_benchmark(benchmark):
        raise HarnessError("configured benchmark does not implement the Benchmark contract")
    stack = _stack_factory(data["stack"])
    runner_config = data["runner"]
    harness = NavigationHarness(
        timeout_s=float(runner_config.get("timeout_s", 300)),
        shutdown_timeout_s=float(runner_config.get("shutdown_timeout_s", 10)),
    )
    summary = await BenchRunner(harness).run(
        benchmark,
        stack,
        parallelism=int(runner_config.get("parallelism", 1)),
    )
    output_root = Path(data.get("output", {}).get("root", "runs/latest"))
    manifest_path = write_manifest(output_root, resolved, summary)
    return summary, manifest_path


def write_manifest(
    output_root: Path, resolved: ResolvedConfig, summary: RunSummary
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, list[float]] = {}
    records = []
    for record in summary.records:
        for name, value in record.metrics.items():
            metrics.setdefault(name, []).append(float(value))
        records.append(
            {
                "index": record.index,
                "case_id": record.case_id,
                "error": record.error,
                "metrics": dict(record.metrics),
                "execution_id": record.result.execution_id if record.result else None,
                "terminal": asdict(record.result.terminal) if record.result else None,
                "environment": record.result.environment if record.result else None,
                "cleanup_errors": list(record.result.cleanup_errors) if record.result else [],
                "audit": [asdict(event) for event in record.result.audit]
                if record.result
                else [],
            }
        )
    document = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "config": resolved.data,
        "config_sources": list(resolved.sources),
        "config_digest": resolved.digest,
        "provenance": dict(resolved.data.get("provenance", {})),
        "benchmark": {
            "name": summary.benchmark,
            "split": summary.split,
            "validation_status": summary.validation_status,
        },
        "aggregate_metrics": {
            name: sum(values) / len(values) for name, values in metrics.items() if values
        },
        "records": records,
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


def _stack_factory(value: Mapping[str, Any]) -> ConfiguredStackFactory:
    return ConfiguredStackFactory(
        agent=ComponentSpec.from_config(value["agent"]),
        environment=ComponentSpec.from_config(value["environment"]),
        vln=ComponentSpec.from_config(value["vln"]) if value.get("vln") else None,
        memory=ComponentSpec.from_config(value["memory"]) if value.get("memory") else None,
    )


def _is_benchmark(value: Any) -> bool:
    return all(
        hasattr(value, name)
        for name in ("name", "split", "validation_status", "cases", "score")
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return {"type": type(value).__name__}


def run_config_sync(paths: Sequence[str | Path]) -> tuple[RunSummary, Path]:
    return asyncio.run(run_config(paths))

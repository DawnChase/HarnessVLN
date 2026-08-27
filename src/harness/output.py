from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from harness.errors import HarnessError
from harness.video import VideoSettings, VideoWriter


JsonObject = dict[str, Any]
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ModuleOutput:
    """A filesystem-agnostic output scope handed to one runtime module."""

    def __init__(
        self,
        *,
        module: str,
        record_path: Path | None,
        artifact_dir: Path | None,
        run_dir: Path | None,
        video: VideoSettings | Mapping[str, Any] | None = None,
    ) -> None:
        self.module = module
        self._record_path = record_path
        self._artifact_dir = artifact_dir
        self._run_dir = run_dir
        self._video = (
            video if isinstance(video, VideoSettings) else VideoSettings.from_mapping(video)
        )
        self._record: JsonObject = {}
        self._streams: dict[str, VideoWriter] = {}
        self._stream_records: dict[str, JsonObject] = {}
        self._errors: list[str] = []
        self._touched = False
        self._lock = threading.RLock()
        self._closed = False

    @property
    def enabled(self) -> bool:
        return self._record_path is not None

    def record(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("module output record must be a mapping")
        with self._lock:
            self._ensure_open()
            if self._record_path is None:
                return
            self._touched = True
            self._record.update({str(key): _normalize(value) for key, value in data.items()})
            self._write_record()

    def event(self, data: Mapping[str, Any]) -> None:
        if self._record_path is None:
            return
        path = self._record_path.with_suffix(".events.jsonl")
        _append_jsonl(path, data)

    def frame(
        self,
        stream: str,
        image: Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._ensure_open()
            if self._record_path is None:
                return
            name = _safe_component(stream, "stream")
            self._touched = True
            if not self._video.enabled:
                self._stream_records[name] = {
                    "status": "disabled",
                    "reason": "video output is disabled",
                }
                self._write_record()
                return
            if name in self._stream_records and self._stream_records[name].get(
                "status"
            ) in {"failed", "unavailable"}:
                return
            try:
                writer = self._streams.get(name)
                if writer is None:
                    assert self._artifact_dir is not None and self._run_dir is not None
                    writer = VideoWriter(
                        self._artifact_dir / f"{name}.mp4",
                        run_dir=self._run_dir,
                        settings=self._video,
                    )
                    self._streams[name] = writer
                writer.write(image, metadata or {})
                self._stream_records[name] = writer.record("recording")
                if writer.frame_count == 1:
                    self._write_record()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                self._errors.append(f"{self.module}.{name}: {message}")
                self._stream_records[name] = {
                    "status": "failed",
                    "error": message,
                }
                self._write_record()
                writer = self._streams.pop(name, None)
                if writer is not None:
                    writer.abort()
                if self._video.required:
                    raise HarnessError(
                        f"required video stream {self.module}.{name} failed: {message}"
                    ) from error

    def unavailable(self, stream: str, reason: str) -> None:
        with self._lock:
            self._ensure_open()
            if self._record_path is None:
                return
            name = _safe_component(stream, "stream")
            self._touched = True
            if name in self._streams:
                raise HarnessError(
                    f"cannot mark active video stream unavailable: {self.module}.{name}"
                )
            self._stream_records[name] = {
                "status": "unavailable",
                "reason": str(reason),
            }
            self._write_record()
            if self._video.enabled and self._video.required:
                raise HarnessError(
                    f"required video stream {self.module}.{name} is unavailable: {reason}"
                )

    def finish(self) -> tuple[str, ...]:
        with self._lock:
            if self._closed:
                return tuple(self._errors)
            for name, writer in tuple(self._streams.items()):
                try:
                    writer.close()
                    self._stream_records[name] = writer.record("saved")
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    self._errors.append(f"{self.module}.{name}: {message}")
                    self._stream_records[name] = {
                        **writer.record("failed"),
                        "error": message,
                    }
                    writer.abort()
            self._streams.clear()
            self._closed = True
            self._write_record()
            return tuple(self._errors)

    def _write_record(self) -> None:
        if self._record_path is None or not self._touched:
            return
        document = {
            "schema_version": 1,
            "module": self.module,
            **self._record,
        }
        if self._stream_records:
            document["streams"] = dict(self._stream_records)
        _atomic_json(self._record_path, document)

    def _ensure_open(self) -> None:
        if self._closed:
            raise HarnessError(f"module output is closed: {self.module}")


class EpisodeOutput:
    def __init__(
        self,
        path: Path,
        *,
        run_dir: Path,
        video: VideoSettings,
    ) -> None:
        self.path = path
        self._run_dir = run_dir
        self._video = video
        self._modules: dict[str, ModuleOutput] = {}
        self._closed = False
        self.path.mkdir(parents=True, exist_ok=False)

    def module(self, name: str) -> ModuleOutput:
        if self._closed:
            raise HarnessError("episode output is closed")
        name = _safe_component(name, "module")
        existing = self._modules.get(name)
        if existing is not None:
            return existing
        if name == "benchmark":
            record_path = self.path / "episode.json"
        elif name == "environment":
            record_path = self.path / "environment.json"
        else:
            record_path = self.path / "components" / f"{name}.json"
        output = ModuleOutput(
            module=name,
            record_path=record_path,
            artifact_dir=self.path / "artifacts" / name,
            run_dir=self._run_dir,
            video=self._video,
        )
        self._modules[name] = output
        return output

    def event(self, data: Mapping[str, Any]) -> None:
        if self._closed:
            raise HarnessError("episode output is closed")
        _append_jsonl(self.path / "events.jsonl", data)

    def finish(self, result: Mapping[str, Any]) -> tuple[str, ...]:
        if self._closed:
            return ()
        errors = [
            error
            for output in self._modules.values()
            for error in output.finish()
        ]
        document = dict(result)
        document["output_errors"] = errors
        document["finished_at_unix"] = time.time()
        _atomic_json(self.path / "result.json", document)
        self._closed = True
        return tuple(errors)


class BenchOutput:
    def __init__(
        self,
        path: Path,
        *,
        run_dir: Path,
        video: VideoSettings,
        benchmark_id: str,
    ) -> None:
        self.path = path
        self.benchmark_id = benchmark_id
        self._run_dir = run_dir
        self._video = video
        self._episodes: list[tuple[EpisodeOutput, JsonObject]] = []
        self._summary: JsonObject = {
            "schema_version": 1,
            "benchmark_id": benchmark_id,
            "status": "running",
            "started_at_unix": time.time(),
            "episodes": [],
        }
        self.path.mkdir(parents=True, exist_ok=False)
        _atomic_json(self.path / "summary.json", self._summary)

    def episode(
        self,
        index: int,
        case_id: str,
        record: Mapping[str, Any],
    ) -> EpisodeOutput:
        name = f"{index:06d}-{_slug(case_id)}"
        output = EpisodeOutput(
            self.path / "episodes" / name,
            run_dir=self._run_dir,
            video=self._video,
        )
        output.module("benchmark").record({"index": index, **dict(record)})
        self._episodes.append(
            (
                output,
                {
                    "index": index,
                    "case_id": case_id,
                    "path": str(
                        (output.path / "result.json").relative_to(self.path)
                    ),
                },
            )
        )
        return output

    def finish(self, summary: Mapping[str, Any]) -> None:
        summary_record = dict(summary)
        supplied_episodes = {
            int(item["index"]): dict(item)
            for item in summary_record.pop("episodes", ())
        }
        episodes = [
            {**reference, **supplied_episodes.get(int(reference["index"]), {})}
            for _, reference in sorted(
                self._episodes, key=lambda item: int(item[1]["index"])
            )
        ]
        case_counts = summary_record.get("case_counts", {})
        has_case_failures = isinstance(case_counts, Mapping) and any(
            int(case_counts.get(name, 0)) > 0
            for name in (
                "task_failures",
                "case_errors",
                "cleanup_errors",
                "output_errors",
            )
        )
        self._summary = {
            "schema_version": 1,
            "benchmark_id": self.benchmark_id,
            "status": (
                "failed"
                if summary_record.get("error")
                or summary_record.get("cleanup_errors")
                or has_case_failures
                else "completed"
            ),
            "started_at_unix": self._summary["started_at_unix"],
            "finished_at_unix": time.time(),
            **summary_record,
            "episodes": episodes,
        }
        _atomic_json(self.path / "summary.json", self._summary)

    @property
    def summary(self) -> Mapping[str, Any]:
        return self._summary


class RunOutput:
    def __init__(
        self,
        output: Mapping[str, Any],
        *,
        resolved_config: Mapping[str, Any],
        config_sources: Sequence[str],
        config_digest: str,
        provenance: Mapping[str, Any],
    ) -> None:
        root = Path(output.get("root", "runs"))
        root.mkdir(parents=True, exist_ok=True)
        requested_id = output.get("run_id")
        if requested_id is not None:
            run_id = _safe_component(str(requested_id), "run_id")
            run_dir = root / run_id
            run_dir.mkdir(parents=False, exist_ok=False)
        else:
            while True:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run_id = f"{stamp}-{config_digest[:8]}-{uuid.uuid4().hex[:6]}"
                run_dir = root / run_id
                try:
                    run_dir.mkdir(parents=False, exist_ok=False)
                except FileExistsError:
                    continue
                break
        self.run_id = run_id
        self.path = run_dir
        self._video = VideoSettings.from_mapping(output.get("video"))
        self._benches: dict[int, BenchOutput] = {}
        self._created_at = time.time()
        config_dir = self.path / "config"
        config_dir.mkdir()
        _atomic_yaml(config_dir / "resolved.yaml", resolved_config)
        _atomic_json(
            config_dir / "sources.json",
            {
                "config_digest": config_digest,
                "sources": list(config_sources),
            },
        )
        self._manifest: JsonObject = {
            "schema_version": 3,
            "run_id": run_id,
            "status": "running",
            "created_at_unix": self._created_at,
            "config": {
                "path": "config/resolved.yaml",
                "sources_path": "config/sources.json",
                "digest": config_digest,
            },
            "provenance": _normalize(provenance),
            "aggregate_metrics": {},
            "benchmarks": [],
        }
        self._write_manifest()

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def benchmark(
        self,
        index: int,
        name: str,
        split: str,
    ) -> BenchOutput:
        if index in self._benches:
            raise HarnessError(f"benchmark output index already exists: {index}")
        benchmark_id = f"{index:03d}-{_slug(name)}-{_slug(split or 'default')}"
        output = BenchOutput(
            self.path / "benches" / benchmark_id,
            run_dir=self.path,
            video=self._video,
            benchmark_id=benchmark_id,
        )
        self._benches[index] = output
        self._write_manifest()
        return output

    def finish(
        self,
        *,
        aggregate_metrics: Mapping[str, float],
        status: str,
    ) -> Path:
        self._manifest.update(
            {
                "status": status,
                "finished_at_unix": time.time(),
                "aggregate_metrics": dict(aggregate_metrics),
            }
        )
        self._write_manifest()
        return self.manifest_path

    def _write_manifest(self) -> None:
        self._manifest["benchmarks"] = [
            {
                "benchmark_id": bench.benchmark_id,
                "path": str((bench.path / "summary.json").relative_to(self.path)),
                "status": bench.summary.get("status", "running"),
                "name": bench.summary.get("name"),
                "split": bench.summary.get("split"),
                "validation_status": bench.summary.get("validation_status"),
                "aggregate_metrics": bench.summary.get("aggregate_metrics", {}),
                "case_counts": bench.summary.get("case_counts", {}),
                "error": bench.summary.get("error"),
            }
            for _, bench in sorted(self._benches.items())
        ]
        _atomic_json(self.manifest_path, self._manifest)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_normalize(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}-", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                _normalize(value),
                handle,
                allow_unicode=False,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(_normalize(value), handle, sort_keys=True)
        handle.write("\n")
        handle.flush()


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_normalize(item) for item in sorted(value, key=str)]
    if isinstance(value, (bytes, bytearray)):
        return {"type": type(value).__name__, "size": len(value)}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {
            "type": type(value).__name__,
            "shape": [int(item) for item in shape],
            "dtype": str(dtype),
        }
    return {"type": type(value).__name__}


def _safe_component(value: str, label: str) -> str:
    value = value.strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid {label}: {value!r}")
    cleaned = _SAFE_NAME.sub("_", value).strip("._-")
    if not cleaned:
        raise ValueError(f"invalid {label}: {value!r}")
    return cleaned


def _slug(value: str) -> str:
    cleaned = _SAFE_NAME.sub("_", value.strip()).strip("._-") or "unnamed"
    if len(cleaned) <= 80:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:69]}-{digest}"


NULL_MODULE_OUTPUT = ModuleOutput(
    module="disabled",
    record_path=None,
    artifact_dir=None,
    run_dir=None,
)

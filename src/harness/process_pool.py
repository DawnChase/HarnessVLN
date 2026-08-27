from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue
import signal
import sys
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import islice
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from benches.base import Benchmark, BenchmarkCase
from harness.config import ComponentSpec
from harness.errors import HarnessError
from harness.output import BenchOutput, EpisodeOutput
from harness.runner import (
    BenchmarkSummary,
    CaseCompleted,
    CaseRecord,
    execute_case,
)
from harness.runtime import NavigationHarness
from harness.video import VideoSettings


@dataclass(frozen=True, slots=True)
class DeviceSlot:
    slot_id: int
    physical_device: int
    local_device: int = 0


def build_device_slots(settings: Mapping[str, Any]) -> tuple[DeviceSlot, ...]:
    devices = settings.get("devices")
    if devices is None:
        if "workers_per_device" in settings:
            raise HarnessError("runner.workers_per_device requires runner.devices")
        return ()
    workers_per_device = int(settings.get("workers_per_device", 1))
    slots: list[DeviceSlot] = []
    for physical_device in devices:
        for _ in range(workers_per_device):
            slots.append(DeviceSlot(len(slots), int(physical_device)))
    return tuple(slots)


class DevicePool:
    """Lease GPU process slots atomically to concurrently running benches."""

    def __init__(self, slots: tuple[DeviceSlot, ...]) -> None:
        if not slots:
            raise ValueError("device pool requires at least one slot")
        self.slots = slots
        self._available = list(slots)
        self._condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, count: int) -> AsyncIterator[tuple[DeviceSlot, ...]]:
        if count < 1:
            raise HarnessError("task_parallelism must be at least 1")
        if count > len(self.slots):
            raise HarnessError(
                "task_parallelism exceeds configured GPU worker slots: "
                f"{count} > {len(self.slots)}"
            )
        async with self._condition:
            await self._condition.wait_for(lambda: len(self._available) >= count)
            leased = tuple(self._available[:count])
            del self._available[:count]
        try:
            yield leased
        finally:
            async with self._condition:
                self._available.extend(leased)
                self._available.sort(key=lambda slot: slot.slot_id)
                self._condition.notify_all()


@dataclass(frozen=True, slots=True)
class _WorkerConfig:
    slot: DeviceSlot
    agent: ComponentSpec
    environment: ComponentSpec
    vln: ComponentSpec | None
    memory: ComponentSpec | None
    benchmark: ComponentSpec
    timeout_s: float
    shutdown_timeout_s: float
    run_dir: Path
    video: VideoSettings
    worker_log: Path


@dataclass(frozen=True, slots=True)
class _EpisodeJob:
    index: int
    case: BenchmarkCase
    output_path: Path


@dataclass(frozen=True, slots=True)
class _WorkerReady:
    slot_id: int


@dataclass(frozen=True, slots=True)
class _WorkerRecord:
    slot_id: int
    record: CaseRecord


@dataclass(frozen=True, slots=True)
class _WorkerDone:
    slot_id: int
    cleanup_error: str | None = None


@dataclass(frozen=True, slots=True)
class _WorkerFatal:
    slot_id: int
    error: str


class ProcessBenchmarkExecutor:
    """Run complete episodes in persistent, GPU-pinned worker processes."""

    def __init__(
        self,
        *,
        agent: ComponentSpec,
        environment: ComponentSpec,
        vln: ComponentSpec | None,
        memory: ComponentSpec | None,
        benchmark: ComponentSpec,
        timeout_s: float,
        shutdown_timeout_s: float,
    ) -> None:
        self.agent = agent
        self.environment = environment
        self.vln = vln
        self.memory = memory
        self.benchmark = benchmark
        self.timeout_s = timeout_s
        self.shutdown_timeout_s = shutdown_timeout_s

    async def run(
        self,
        benchmark: Benchmark,
        *,
        slots: tuple[DeviceSlot, ...],
        max_cases: int | None,
        output: BenchOutput,
        on_case_complete: CaseCompleted | None = None,
    ) -> BenchmarkSummary:
        if not slots:
            raise HarnessError("process benchmark execution requires GPU slots")
        if max_cases is not None and max_cases < 1:
            raise HarnessError("max_cases must be at least 1")

        cases = benchmark.cases()
        selected = cases if max_cases is None else islice(cases, max_cases)
        indexed_cases = enumerate(selected)

        def next_job() -> _EpisodeJob | None:
            try:
                index, case = next(indexed_cases)
            except StopIteration:
                return None
            episode = output.episode(index, case.case_id, case.output_record())
            return _EpisodeJob(index, case, episode.path)

        first = next_job()
        if first is None:
            return BenchmarkSummary(
                benchmark.name,
                benchmark.split,
                benchmark.validation_status,
                (),
            )

        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        request_queues: dict[int, Any] = {}
        processes: dict[int, BaseProcess] = {}
        worker_log_dir = output.path / "logs"
        worker_log_dir.mkdir(parents=True, exist_ok=True)
        try:
            for slot in slots:
                requests = context.Queue(maxsize=1)
                config = _WorkerConfig(
                    slot=slot,
                    agent=self.agent,
                    environment=self.environment,
                    vln=self.vln,
                    memory=self.memory,
                    benchmark=self.benchmark,
                    timeout_s=self.timeout_s,
                    shutdown_timeout_s=self.shutdown_timeout_s,
                    run_dir=output.run_dir,
                    video=output.video,
                    worker_log=(
                        worker_log_dir
                        / f"worker-{slot.slot_id:03d}-gpu-{slot.physical_device}.log"
                    ),
                )
                worker_process = context.Process(
                    target=_worker_entry,
                    args=(config, requests, result_queue),
                    name=f"harness-gpu-{slot.physical_device}-slot-{slot.slot_id}",
                )
                worker_process.start()
                request_queues[slot.slot_id] = requests
                processes[slot.slot_id] = worker_process

            await _wait_until_ready(result_queue, processes)

            records: list[CaseRecord] = []
            cleanup_errors: list[str] = []
            pending_first: _EpisodeJob | None = first
            stopping: set[int] = set()
            for slot in slots:
                job: _EpisodeJob | None = (
                    pending_first if pending_first is not None else next_job()
                )
                pending_first = None
                if job is None:
                    request_queues[slot.slot_id].put(None)
                    stopping.add(slot.slot_id)
                else:
                    request_queues[slot.slot_id].put(job)

            done: set[int] = set()
            while len(done) < len(slots):
                message = await _receive(result_queue, processes, done)
                if isinstance(message, _WorkerRecord):
                    records.append(message.record)
                    if on_case_complete is not None:
                        on_case_complete(message.record)
                    following_job = next_job()
                    if following_job is None:
                        request_queues[message.slot_id].put(None)
                        stopping.add(message.slot_id)
                    else:
                        request_queues[message.slot_id].put(following_job)
                elif isinstance(message, _WorkerDone):
                    if message.slot_id not in stopping:
                        raise HarnessError(
                            f"GPU worker slot {message.slot_id} stopped before shutdown"
                        )
                    done.add(message.slot_id)
                    if message.cleanup_error is not None:
                        cleanup_errors.append(message.cleanup_error)
                elif isinstance(message, _WorkerFatal):
                    raise HarnessError(
                        f"GPU worker slot {message.slot_id} failed: {message.error}"
                    )
                else:
                    raise HarnessError(
                        f"unexpected GPU worker response: {type(message).__name__}"
                    )

            for process in processes.values():
                await asyncio.to_thread(process.join, 5)
                if process.exitcode != 0:
                    raise HarnessError(
                        f"GPU worker {process.name} exited with code {process.exitcode}"
                    )
            records.sort(key=lambda record: record.index)
            return BenchmarkSummary(
                benchmark.name,
                benchmark.split,
                benchmark.validation_status,
                tuple(records),
                cleanup_errors=tuple(cleanup_errors),
            )
        finally:
            await _close_processes(processes)
            for requests in request_queues.values():
                requests.close()
                requests.cancel_join_thread()
            result_queue.close()
            result_queue.cancel_join_thread()


async def _wait_until_ready(
    results: Any,
    processes: Mapping[int, BaseProcess],
) -> None:
    ready: set[int] = set()
    while len(ready) < len(processes):
        message = await _receive(results, processes, set())
        if isinstance(message, _WorkerFatal):
            raise HarnessError(
                f"GPU worker slot {message.slot_id} failed to start: {message.error}"
            )
        if not isinstance(message, _WorkerReady):
            raise HarnessError(
                f"unexpected GPU worker startup response: {type(message).__name__}"
            )
        ready.add(message.slot_id)


async def _receive(
    results: Any,
    processes: Mapping[int, BaseProcess],
    completed: set[int],
) -> Any:
    while True:
        try:
            return await asyncio.to_thread(results.get, True, 0.2)
        except queue.Empty:
            failed = [
                process
                for slot_id, process in processes.items()
                if slot_id not in completed
                and process.exitcode is not None
                and process.exitcode != 0
            ]
            if failed:
                details = ", ".join(
                    f"{process.name}={process.exitcode}" for process in failed
                )
                raise HarnessError(f"GPU worker process failed: {details}")


async def _close_processes(
    processes: Mapping[int, BaseProcess],
) -> None:
    for process in processes.values():
        if not process.is_alive():
            continue
        pid = process.pid
        if pid is None:
            continue
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError):
            pass
    for process in processes.values():
        if process.pid is None:
            continue
        await asyncio.to_thread(process.join, 5)
        if process.is_alive():
            pid = process.pid
            if pid is None:
                continue
            try:
                if os.getpgid(pid) == pid:
                    os.killpg(pid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError):
                pass
            await asyncio.to_thread(process.join, 5)


def _worker_entry(config: _WorkerConfig, requests: Any, results: Any) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    try:
        _redirect_worker_output(config.worker_log)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.slot.physical_device)
        os.environ["HARNESS_WORKER_SLOT"] = str(config.slot.slot_id)
        asyncio.run(_worker_loop(config, requests, results))
    except BaseException as error:
        results.put(
            _WorkerFatal(
                config.slot.slot_id,
                f"{type(error).__name__}: {error}",
            )
        )
        raise


async def _worker_loop(config: _WorkerConfig, requests: Any, results: Any) -> None:
    from harness.app import ConfiguredStackFactory

    benchmark = config.benchmark.create()
    factory = ConfiguredStackFactory(
        agent=config.agent,
        environment=config.environment,
        vln=config.vln,
        memory=config.memory,
    )
    harness = NavigationHarness(config.timeout_s, config.shutdown_timeout_s)
    results.put(_WorkerReady(config.slot.slot_id))
    cleanup_error: str | None = None
    fatal_error: str | None = None
    try:
        while True:
            job = requests.get()
            if job is None:
                break
            resources = {
                "worker_id": config.slot.slot_id,
                "pid": os.getpid(),
                "gpu": {
                    "physical_device": config.slot.physical_device,
                    "local_device": config.slot.local_device,
                },
                "worker_log": str(config.worker_log.relative_to(config.run_dir)),
            }
            output = EpisodeOutput.open_existing(
                job.output_path,
                run_dir=config.run_dir,
                video=config.video,
            )
            record = await execute_case(
                harness,
                job.index,
                job.case,
                benchmark,
                factory,
                output,
                resources=resources,
            )
            results.put(_WorkerRecord(config.slot.slot_id, record))
    except BaseException as error:
        fatal_error = f"{type(error).__name__}: {error}"
    finally:
        try:
            await factory.close_session()
        except BaseException as error:
            cleanup_error = f"worker {config.slot.slot_id}: {type(error).__name__}: {error}"
        if fatal_error is not None:
            results.put(_WorkerFatal(config.slot.slot_id, fatal_error))
        else:
            results.put(_WorkerDone(config.slot.slot_id, cleanup_error))


def _redirect_worker_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (AttributeError, OSError):
            pass
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        if descriptor > 2:
            os.close(descriptor)

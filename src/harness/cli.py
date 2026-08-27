from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from harness.app import InteractiveNavigationSession, execute_runner_sync
from harness.runner import BenchmarkSummary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "env":
        return asyncio.run(_run_interactive(arguments.environment, arguments.agent))
    return _run_benchmarks(arguments.runner, arguments.agent)


def _run_benchmarks(runner: str, agent: str) -> int:
    run_summary, manifest = execute_runner_sync(runner, agent)
    failed = False
    for benchmark in run_summary.benchmarks:
        task_failures, case_errors, bench_errors, cleanup_errors = _failure_counts(
            benchmark
        )
        failed = failed or bool(
            task_failures or case_errors or bench_errors or cleanup_errors
        )
        print(
            f"{benchmark.benchmark}/{benchmark.split}: "
            f"{len(benchmark.records)} cases, {task_failures} task failures, "
            f"{case_errors} case errors, {bench_errors} bench errors, "
            f"{cleanup_errors} cleanup errors"
        )
        if benchmark.error is not None:
            print(f"  bench error: {benchmark.error}")
    print(manifest.resolve())
    return int(failed)


async def _run_interactive(environment: str, agent: str) -> int:
    session = InteractiveNavigationSession.from_configs(environment, agent)
    failed = False
    try:
        while True:
            try:
                instruction = input("nav> ")
            except EOFError:
                break
            instruction = instruction.strip()
            if instruction.lower() in {"exit", "quit"}:
                break
            if not instruction:
                continue
            result = await session.navigate(instruction)
            terminal = result.terminal
            detail = f": {terminal.reason}" if terminal.reason else ""
            print(f"{result.task_id}: {terminal.status}{detail}")
            failed = failed or terminal.status != "completed" or bool(
                result.cleanup_errors
            )
    finally:
        await session.close()
    return int(failed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli",
        description="Run or interact with a HarnessVLN navigation agent",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the benches referenced by a runner YAML")
    run.add_argument(
        "--runner",
        required=True,
        metavar="PATH",
        help="runner YAML that references one or more bench YAML files",
    )
    run.add_argument(
        "--agent",
        required=True,
        metavar="PATH",
        help="agent YAML that references its VLN and memory plugins",
    )
    environment = commands.add_parser(
        "env", help="send navigation instructions to an agent in one environment"
    )
    environment.add_argument(
        "--environment",
        required=True,
        metavar="PATH",
        help="simulator, robot, or service environment YAML",
    )
    environment.add_argument(
        "--agent",
        required=True,
        metavar="PATH",
        help="agent YAML that references its VLN and memory plugins",
    )
    return parser


def _failure_counts(summary: BenchmarkSummary) -> tuple[int, int, int, int]:
    task_failures = sum(
        record.result is not None and record.result.terminal.status == "failed"
        for record in summary.records
    )
    case_errors = sum(record.error is not None for record in summary.records)
    bench_errors = int(summary.error is not None)
    cleanup_errors = len(summary.cleanup_errors) + sum(
        len(record.result.cleanup_errors)
        for record in summary.records
        if record.result is not None
    )
    return task_failures, case_errors, bench_errors, cleanup_errors


if __name__ == "__main__":
    raise SystemExit(main())

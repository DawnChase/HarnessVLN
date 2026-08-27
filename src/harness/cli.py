from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from harness.app import InteractiveNavigationSession, run_config_sync
from harness.runner import RunSummary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "env":
        return asyncio.run(_run_interactive(arguments.environment, arguments.agent))
    return _run_benchmarks(arguments.runner, arguments.agent)


def _run_benchmarks(runner: str, agent: str) -> int:
    summary, manifest = run_config_sync(runner, agent)
    failed = False
    for run in summary.runs:
        task_failures, runner_errors, cleanup_errors = _failure_counts(run)
        failed = failed or bool(task_failures or runner_errors or cleanup_errors)
        print(
            f"{run.benchmark}/{run.split}: "
            f"{len(run.records)} cases, {task_failures} task failures, "
            f"{runner_errors} runner errors, {cleanup_errors} cleanup errors"
        )
        if run.error is not None:
            print(f"  bench error: {run.error}")
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


def _failure_counts(summary: RunSummary) -> tuple[int, int, int]:
    task_failures = sum(
        record.result is not None and record.result.terminal.status == "failed"
        for record in summary.records
    )
    runner_errors = int(summary.error is not None) + sum(
        record.error is not None for record in summary.records
    )
    cleanup_errors = len(summary.cleanup_errors) + sum(
        len(record.result.cleanup_errors)
        for record in summary.records
        if record.result is not None
    )
    return task_failures, runner_errors, cleanup_errors


if __name__ == "__main__":
    raise SystemExit(main())

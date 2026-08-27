from __future__ import annotations

import argparse
from collections.abc import Sequence

from harness.app import run_config_sync
from harness.runner import RunSummary


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    summary, manifest = run_config_sync(_config_paths(arguments))
    task_failures, runner_errors, cleanup_errors = _failure_counts(summary)
    print(
        f"{summary.benchmark}/{summary.split}: "
        f"{len(summary.records)} cases, {task_failures} task failures, "
        f"{runner_errors} runner errors, {cleanup_errors} cleanup errors"
    )
    print(manifest.resolve())
    return int(bool(task_failures or runner_errors or cleanup_errors))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m harness.cli",
        description="Run a HarnessVLN benchmark",
    )
    configs = parser.add_argument_group("experiment components")
    configs.add_argument(
        "--benchmark",
        required=True,
        metavar="PATH",
        help="benchmark case loader and scorer YAML",
    )
    configs.add_argument(
        "--agent",
        required=True,
        metavar="PATH",
        help="agent core YAML",
    )
    configs.add_argument(
        "--environment",
        required=True,
        action="append",
        metavar="PATH",
        help="simulator or robot environment YAML; repeat for environment overlays",
    )
    configs.add_argument(
        "--vln",
        metavar="PATH",
        help="optional VLN navigator YAML",
    )
    configs.add_argument(
        "--memory",
        metavar="PATH",
        help="optional spatial memory YAML",
    )
    configs.add_argument(
        "--run",
        required=True,
        metavar="PATH",
        help="runner, output, and provenance YAML",
    )
    configs.add_argument(
        "--overlay",
        action="append",
        default=[],
        metavar="PATH",
        help="final YAML overlay; repeat to apply multiple overrides in order",
    )
    return parser


def _config_paths(arguments: argparse.Namespace) -> tuple[str, ...]:
    paths = [
        arguments.benchmark,
        arguments.agent,
        *arguments.environment,
    ]
    if arguments.vln is not None:
        paths.append(arguments.vln)
    if arguments.memory is not None:
        paths.append(arguments.memory)
    paths.append(arguments.run)
    paths.extend(arguments.overlay)
    return tuple(paths)


def _failure_counts(summary: RunSummary) -> tuple[int, int, int]:
    task_failures = sum(
        record.result is not None and record.result.terminal.status == "failed"
        for record in summary.records
    )
    runner_errors = sum(record.error is not None for record in summary.records)
    cleanup_errors = sum(
        len(record.result.cleanup_errors)
        for record in summary.records
        if record.result is not None
    )
    return task_failures, runner_errors, cleanup_errors


if __name__ == "__main__":
    raise SystemExit(main())

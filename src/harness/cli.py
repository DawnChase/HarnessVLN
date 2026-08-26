from __future__ import annotations

import argparse
from collections.abc import Sequence

from harness.app import run_config_sync
from harness.runner import RunSummary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a HarnessVLN benchmark")
    parser.add_argument("configs", nargs="+", help="YAML files in overlay order")
    arguments = parser.parse_args(argv)
    summary, manifest = run_config_sync(arguments.configs)
    task_failures, runner_errors, cleanup_errors = _failure_counts(summary)
    print(
        f"{summary.benchmark}/{summary.split}: "
        f"{len(summary.records)} cases, {task_failures} task failures, "
        f"{runner_errors} runner errors, {cleanup_errors} cleanup errors"
    )
    print(manifest.resolve())
    return int(bool(task_failures or runner_errors or cleanup_errors))


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

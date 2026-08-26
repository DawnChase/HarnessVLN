from __future__ import annotations

from pathlib import Path

from harness import cli
from harness.runner import CaseRecord, RunSummary
from harness.runtime import NavigationResult, Terminal


def _result(
    status: str = "completed", *, cleanup_errors: tuple[str, ...] = ()
) -> NavigationResult:
    return NavigationResult(
        execution_id="execution",
        task_id="task",
        terminal=Terminal(status, "reason", "harness"),
        environment={},
        audit=(),
        cleanup_errors=cleanup_errors,
    )


def test_cli_returns_zero_for_a_clean_run(monkeypatch, capsys, tmp_path: Path) -> None:
    summary = RunSummary(
        "bench",
        "split",
        "contract",
        (CaseRecord(0, "case", _result(), {"success": 1.0}),),
    )
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(cli, "run_config_sync", lambda paths: (summary, manifest))

    assert cli.main(["config.yaml"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "bench/split: 1 cases, 0 task failures, 0 runner errors, 0 cleanup errors",
        str(manifest.resolve()),
    ]


def test_cli_returns_nonzero_and_separates_failure_kinds(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    summary = RunSummary(
        "bench",
        "split",
        "contract",
        (
            CaseRecord(0, "task-failure", _result("failed"), {}),
            CaseRecord(1, "runner-error", None, {}, "RuntimeError: failed"),
            CaseRecord(
                2,
                "cleanup-error",
                _result(cleanup_errors=("environment: cleanup timed out",)),
                {},
            ),
        ),
    )
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(cli, "run_config_sync", lambda paths: (summary, manifest))

    assert cli.main(["config.yaml"]) == 1
    assert capsys.readouterr().out.splitlines()[0] == (
        "bench/split: 3 cases, 1 task failures, 1 runner errors, 1 cleanup errors"
    )

from __future__ import annotations

import builtins
from pathlib import Path

from harness import cli
from harness.runner import BenchmarkSummary, CaseRecord, RunSummary
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
        (
            BenchmarkSummary(
                "bench",
                "split",
                "contract",
                (CaseRecord(0, "case", _result(), {"success": 1.0}),),
            ),
        )
    )
    manifest = tmp_path / "manifest.json"
    received = []

    def run(runner, agent):
        received.extend((runner, agent))
        return summary, manifest

    monkeypatch.setattr(cli, "execute_runner_sync", run)

    assert (
        cli.main(
            [
                "run",
                "--runner",
                "runner.yaml",
                "--agent",
                "agent.yaml",
            ]
        )
        == 0
    )
    assert received == ["runner.yaml", "agent.yaml"]
    assert capsys.readouterr().out.splitlines() == [
        "bench/split: 1 cases, 0 task failures, 0 case errors, 0 bench errors, "
        "0 cleanup errors, 0 output errors",
        str(manifest.resolve()),
    ]


def test_cli_returns_nonzero_and_separates_failure_kinds(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    summary = RunSummary(
        (
            BenchmarkSummary(
                "bench",
                "split",
                "contract",
                (
                    CaseRecord(0, "task-failure", _result("timeout"), {}),
                    CaseRecord(
                        1,
                        "case-error",
                        None,
                        {},
                        error="RuntimeError: failed",
                        error_stage="execution",
                    ),
                    CaseRecord(
                        2,
                        "cleanup-error",
                        _result(cleanup_errors=("environment: cleanup timed out",)),
                        {},
                        output_errors=("environment.main_camera: encode failed",),
                    ),
                ),
            ),
        )
    )
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(
        cli, "execute_runner_sync", lambda runner, agent: (summary, manifest)
    )

    assert (
        cli.main(
            [
                "run",
                "--runner",
                "runner.yaml",
                "--agent",
                "agent.yaml",
            ]
        )
        == 1
    )
    assert capsys.readouterr().out.splitlines()[0] == (
        "bench/split: 3 cases, 1 task failures, 1 case errors, 0 bench errors, "
        "1 cleanup errors, 1 output errors"
    )


def test_cli_reports_bench_level_failure_and_cleanup(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    summary = RunSummary(
        (
            BenchmarkSummary(
                "broken-bench",
                "split",
                "contract",
                (),
                error="RuntimeError: case stream failed",
                cleanup_errors=("RuntimeError: close failed",),
            ),
        )
    )
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(
        cli, "execute_runner_sync", lambda runner, agent: (summary, manifest)
    )

    assert (
        cli.main(
            [
                "run",
                "--runner",
                "runner.yaml",
                "--agent",
                "agent.yaml",
            ]
        )
        == 1
    )
    assert capsys.readouterr().out.splitlines() == [
        "broken-bench/split: 0 cases, 0 task failures, 0 case errors, "
        "1 bench errors, 1 cleanup errors, 0 output errors",
        "  bench error: RuntimeError: case stream failed",
        str(manifest.resolve()),
    ]


def test_env_cli_sends_instructions_to_the_agent_until_quit(
    monkeypatch, capsys
) -> None:
    commands = iter(("Stay at the marker.", "quit"))
    monkeypatch.setattr(builtins, "input", lambda prompt: next(commands))

    assert (
        cli.main(
            [
                "env",
                "--environment",
                "config/envs/dummy.yaml",
                "--agent",
                "config/agents/passthrough.yaml",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out.strip()
    assert output.startswith("interactive:")
    assert output.endswith("completed: all navigation goals completed")

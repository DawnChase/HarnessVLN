from __future__ import annotations

import io

from harness.progress import RunProgress, benchmark_case_total
from harness.runner import CaseRecord


class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


class SizedBenchmark:
    name = "r2r_ce"
    split = "val_unseen"
    validation_status = "contract"
    case_count = 3


def test_progress_renders_completed_episode_count_on_the_last_line() -> None:
    stream = TerminalBuffer()
    progress = RunProgress(stream=stream)
    benchmark = SizedBenchmark()
    progress.register("bench", benchmark, None)  # type: ignore[arg-type]

    progress.advance("bench", CaseRecord(0, "first", None, {}))
    progress.advance(
        "bench",
        CaseRecord(
            1,
            "second",
            None,
            {},
            error="RuntimeError: failed",
            error_stage="execution",
        ),
    )
    progress.advance("bench", CaseRecord(2, "third", None, {}))
    progress.close()

    rendered = stream.getvalue()
    last_line = rendered.rstrip("\n").split("\r")[-1]
    assert "r2r_ce/val_unseen" in last_line
    assert "3/3" in last_line
    assert "failed=1" in last_line


def test_progress_is_silent_when_stream_is_not_a_terminal() -> None:
    stream = io.StringIO()
    progress = RunProgress(stream=stream)
    progress.register("bench", SizedBenchmark(), None)  # type: ignore[arg-type]
    progress.advance("bench", CaseRecord(0, "first", None, {}))
    progress.close()

    assert stream.getvalue() == ""


def test_benchmark_total_respects_case_limit() -> None:
    benchmark = SizedBenchmark()

    assert benchmark_case_total(benchmark, None) == 3  # type: ignore[arg-type]
    assert benchmark_case_total(benchmark, 2) == 2  # type: ignore[arg-type]

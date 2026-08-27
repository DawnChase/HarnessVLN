#!/usr/bin/env python3
"""Replay a completed HarnessVLN record and export static viewer assets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from harness.config import load_runner_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--trace-id", default=None)
    parser.add_argument("--output", default=Path("viewer/data"), type=Path)
    parser.add_argument("--image-quality", default=82, type=int)
    parser.add_argument("--map-resolution", default=900, type=int)
    return parser.parse_args()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def slug_case_id(case_id: str) -> str:
    return case_id.lower().replace(":", "-").replace("_", "-")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        values.append(value)
    return values


def resolve_output_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing {label} path")
    root = base.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes its output scope: {value}") from error
    return path


def scoped_records(
    manifest_path: Path, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for benchmark in manifest.get("benchmarks", []):
        summary_path = resolve_output_path(
            manifest_path.parent, benchmark.get("path"), "benchmark summary"
        )
        summary = load_json(summary_path)
        for episode in summary.get("episodes", []):
            result_path = resolve_output_path(
                summary_path.parent, episode.get("path"), "episode result"
            )
            record = load_json(result_path)
            events_path = record.get("events_path")
            record["audit"] = (
                load_jsonl(
                    resolve_output_path(
                        result_path.parent, events_path, "episode events"
                    )
                )
                if events_path
                else []
            )
            environment_path = record.get("environment_path")
            record["environment"] = (
                load_json(
                    resolve_output_path(
                        result_path.parent, environment_path, "environment record"
                    )
                )
                if environment_path
                else {}
            )
            records.append(record)
    return records


def select_record(
    manifest_path: Path,
    manifest: dict[str, Any],
    case_id: str | None,
) -> dict[str, Any]:
    if int(manifest.get("schema_version", 0)) >= 3:
        records = scoped_records(manifest_path, manifest)
    else:
        records = [
            record
            for benchmark in manifest.get("benchmarks", [])
            for record in benchmark.get("records", [])
        ]
    if case_id is not None:
        records = [record for record in records if record.get("case_id") == case_id]
    if len(records) != 1:
        raise ValueError(f"expected one manifest record, found {len(records)}")
    return records[0]


def find_case(runner_path: Path, case_id: str):
    runner = load_runner_config(runner_path)
    for benchmark_config in runner.benches:
        benchmark = benchmark_config.benchmark.create()
        for case in benchmark.cases():
            if case.case_id == case_id:
                return benchmark_config, case
    raise ValueError(f"case not found in runner configuration: {case_id}")


def instruction_from_audit(audit: list[dict[str, Any]]) -> str:
    for event in audit:
        if event.get("name") == "vln.navigate.start":
            return str(event.get("arguments", {}).get("instruction", ""))
    return ""


def action_events(audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in audit
        if event.get("name") == "nav.move.discrete"
        and event.get("outcome") == "ok"
    ]


def map_palette(topdown: np.ndarray) -> np.ndarray:
    colors = np.array(
        [
            [18, 20, 19],
            [68, 73, 70],
            [151, 158, 154],
        ],
        dtype=np.uint8,
    )
    return colors[np.clip(topdown, 0, 2)]


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    runner_path = args.runner.expanduser().resolve()
    manifest = load_json(manifest_path)
    record = select_record(manifest_path, manifest, args.case_id)
    case_id = str(record["case_id"])
    trace_id = args.trace_id or slug_case_id(case_id)
    benchmark_config, case = find_case(runner_path, case_id)

    audit = list(record.get("audit", []))
    events = action_events(audit)
    if not events:
        raise ValueError(f"record has no successful motion events: {case_id}")

    output_root = args.output.expanduser().resolve()
    trace_root = output_root / trace_id
    if trace_root.exists():
        shutil.rmtree(trace_root)
    frames_root = trace_root / "frames"
    frames_root.mkdir(parents=True)

    adapter = benchmark_config.environment.component.create(
        episode=case.environment_episode
    )
    session = adapter.native_factory(case.environment_episode)
    frames: list[dict[str, Any]] = []
    map_points: list[dict[str, float]] = []
    first_time = float(events[0]["monotonic_time"])

    try:
        observation = session.reset()

        from habitat.utils.visualizations import maps
        import quaternion

        topdown = maps.get_topdown_map_from_sim(
            session.sim,
            map_resolution=args.map_resolution,
            draw_border=True,
        )
        occupied = np.argwhere(topdown > 0)
        if occupied.size:
            padding = 12
            row_min = max(0, int(occupied[:, 0].min()) - padding)
            row_max = min(topdown.shape[0], int(occupied[:, 0].max()) + padding + 1)
            col_min = max(0, int(occupied[:, 1].min()) - padding)
            col_max = min(topdown.shape[1], int(occupied[:, 1].max()) + padding + 1)
        else:
            row_min, col_min = 0, 0
            row_max, col_max = topdown.shape

        cropped = topdown[row_min:row_max, col_min:col_max]
        lower_bound, upper_bound = session.sim.pathfinder.get_bounds()
        grid_size = (
            abs(float(upper_bound[2] - lower_bound[2])) / topdown.shape[0],
            abs(float(upper_bound[0] - lower_bound[0])) / topdown.shape[1],
        )
        meters_per_pixel = float(
            maps.calculate_meters_per_pixel(args.map_resolution, sim=session.sim)
        )
        Image.fromarray(map_palette(cropped), "RGB").save(
            trace_root / "navmesh.webp", "WEBP", quality=88, method=4
        )

        def map_position(position: Any) -> dict[str, float]:
            row = int((float(position[2]) - float(lower_bound[2])) / grid_size[0])
            col = int((float(position[0]) - float(lower_bound[0])) / grid_size[1])
            return {"x": float(col - col_min), "y": float(row - row_min)}

        def capture(index: int, action: str, elapsed_s: float) -> None:
            rgb = np.asarray(observation["rgb"])
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            frame_path = frames_root / f"frame-{index:03d}.webp"
            Image.fromarray(rgb.astype(np.uint8), "RGB").save(
                frame_path,
                "WEBP",
                quality=args.image_quality,
                method=4,
            )

            state = session.sim.get_agent_state()
            position = np.asarray(state.position, dtype=float)
            forward = quaternion.rotate_vectors(
                state.rotation, np.array([0.0, 0.0, -1.0])
            )
            point = map_position(position)
            metrics = session.get_metrics()
            compass = observation.get("compass", 0.0)
            compass_value = float(np.asarray(compass).reshape(-1)[0])
            gps = observation.get("gps")
            frame = {
                "index": index,
                "action": action,
                "elapsed_s": round(elapsed_s, 3),
                "image": f"frames/{frame_path.name}",
                "world": {
                    "x": round(float(position[0]), 5),
                    "y": round(float(position[1]), 5),
                    "z": round(float(position[2]), 5),
                },
                "map": {
                    **point,
                    "heading_x": round(float(forward[0]), 6),
                    "heading_y": round(float(forward[2]), 6),
                },
                "yaw": round(compass_value, 6),
                "gps": json_value(gps) if gps is not None else None,
                "distance_to_goal": round(
                    float(metrics.get("distance_to_goal", math.nan)), 5
                ),
            }
            frames.append(frame)
            map_points.append(point)

        capture(0, "start", 0.0)
        for index, event in enumerate(events, start=1):
            action = str(event["arguments"]["action"])
            observation = session.step(adapter.native_actions[action])
            capture(index, action, float(event["monotonic_time"]) - first_time)
    finally:
        session.close()

    goal_positions = [
        goal.get("position")
        for goal in case.env_setup.get("goals", [])
        if isinstance(goal, dict) and goal.get("position") is not None
    ]
    reference_path = list(case.truth.get("reference_path", []))
    reference_map = [map_position(point) for point in reference_path]
    goals = [map_position(point) for point in goal_positions]
    goal_radius_m = 3.0
    raw_goals = case.env_setup.get("goals", [])
    if raw_goals and isinstance(raw_goals[0], dict):
        goal_radius_m = float(raw_goals[0].get("radius", goal_radius_m))

    terminal = record.get("terminal") or {}
    final_metrics = dict(record.get("metrics", {}))
    trace = {
        "schema_version": 1,
        "id": trace_id,
        "label": f"JanusVLN · R2R val_unseen · episode {case_id.rsplit(':', 1)[-1]}",
        "case_id": case_id,
        "model": "JanusVLN Base",
        "benchmark": "R2R-CE val_unseen",
        "scene_id": case.task.scene_id,
        "instruction": instruction_from_audit(audit) or case.task.instruction,
        "metrics": final_metrics,
        "terminal": {
            "kind": terminal.get("kind", "completed"),
            "reason": terminal.get("reason", "model emitted STOP"),
        },
        "duration_s": round(
            float(audit[-1]["monotonic_time"] - audit[0]["monotonic_time"]), 3
        ),
        "action_count": len(events),
        "map": {
            "image": "navmesh.webp",
            "width": int(cropped.shape[1]),
            "height": int(cropped.shape[0]),
            "meters_per_pixel": round(meters_per_pixel, 6),
            "path": map_points,
            "reference_path": reference_map,
            "goals": goals,
            "goal_radius_m": goal_radius_m,
        },
        "frames": frames,
    }
    (trace_root / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=True) + "\n"
    )

    index_path = output_root / "index.json"
    index_data = {"traces": []}
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
    traces = [item for item in index_data.get("traces", []) if item.get("id") != trace_id]
    traces.append(
        {
            "id": trace_id,
            "label": trace["label"],
            "path": f"{trace_id}/trace.json",
            "model": trace["model"],
            "benchmark": trace["benchmark"],
            "success": float(final_metrics.get("sr", 0.0)) > 0,
        }
    )
    index_path.write_text(
        json.dumps({"traces": sorted(traces, key=lambda item: item["label"])}, indent=2)
        + "\n"
    )
    print(trace_root / "trace.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio_ffmpeg  # type: ignore[import-untyped]


JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class VideoSettings:
    enabled: bool = True
    required: bool = False
    fps: float = 10.0
    codec: str = "libx264"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "VideoSettings":
        data = dict(value or {})
        settings = cls(
            enabled=bool(data.get("enabled", True)),
            required=bool(data.get("required", False)),
            fps=float(data.get("fps", 10.0)),
            codec=str(data.get("codec", "libx264")).strip(),
        )
        if settings.fps <= 0:
            raise ValueError("output.video.fps must be positive")
        if not settings.codec:
            raise ValueError("output.video.codec must not be empty")
        return settings


class VideoWriter:
    def __init__(
        self,
        path: Path,
        *,
        run_dir: Path,
        settings: VideoSettings,
    ) -> None:
        self.path = path
        self.run_dir = run_dir
        self.settings = settings
        self.frames_path = path.with_suffix(".frames.jsonl")
        self.partial_path = path.with_name(f".{path.stem}.partial.mp4")
        self.width: int | None = None
        self.height: int | None = None
        self.frame_count = 0
        self._writer: Any = None

    def write(self, image: Any, metadata: Mapping[str, Any]) -> None:
        import numpy as np

        frame = np.asarray(image)
        if frame.dtype != np.uint8:
            raise ValueError(f"video frame dtype must be uint8, got {frame.dtype}")
        if frame.ndim != 3 or frame.shape[2] not in {3, 4}:
            raise ValueError(
                "video frame shape must be height x width x 3/4, "
                f"got {tuple(frame.shape)}"
            )
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if width % 2 or height % 2:
            raise ValueError("H.264 video frame width and height must be even")
        if self._writer is None:
            self._start(width, height)
        elif (width, height) != (self.width, self.height):
            raise ValueError(
                "video frame size changed from "
                f"{self.width}x{self.height} to {width}x{height}"
            )
        self._writer.send(np.ascontiguousarray(frame))
        self.frames_path.parent.mkdir(parents=True, exist_ok=True)
        with self.frames_path.open("a", encoding="utf-8") as handle:
            json.dump(
                {"frame_index": self.frame_count, **dict(metadata)},
                handle,
                sort_keys=True,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is None:
            return
        writer, self._writer = self._writer, None
        writer.close()
        if not self.partial_path.is_file() or self.partial_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg produced an empty video")
        os.replace(self.partial_path, self.path)

    def abort(self) -> None:
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()
        self.partial_path.unlink(missing_ok=True)

    def record(self, status: str) -> JsonObject:
        value: JsonObject = {
            "status": status,
            "type": "video",
            "media_type": "video/mp4",
            "codec": self.settings.codec,
            "fps": self.settings.fps,
            "frame_count": self.frame_count,
            "path": str(self.path.relative_to(self.run_dir)),
            "frames_path": str(self.frames_path.relative_to(self.run_dir)),
        }
        if self.width is not None and self.height is not None:
            value.update({"width": self.width, "height": self.height})
        if status == "saved" and self.path.exists():
            value.update(
                {
                    "size_bytes": self.path.stat().st_size,
                    "sha256": _file_digest(self.path),
                }
            )
        return value

    def _start(self, width: int, height: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        self._writer = imageio_ffmpeg.write_frames(
            self.partial_path,
            (width, height),
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            fps=self.settings.fps,
            codec=self.settings.codec,
            macro_block_size=1,
            ffmpeg_log_level="error",
            ffmpeg_timeout=30,
            output_params=["-movflags", "+faststart"],
        )
        self._writer.send(None)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {
            "type": type(value).__name__,
            "shape": [int(item) for item in shape],
            "dtype": str(dtype),
        }
    return {"type": type(value).__name__}

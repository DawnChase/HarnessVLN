from __future__ import annotations

import shutil
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any


MEDIA_REF_KEY = "$harness_array"


class FileArrayStore:
    """Process-local backing files for arrays that must cross the JSON RPC wire."""

    def __init__(self, root: str | Path | None = None) -> None:
        self._configured_root = Path(root) if root is not None else None
        self._root: Path | None = None
        self._next_id = 0
        self._lock = threading.Lock()

    @property
    def root(self) -> Path | None:
        return self._root

    def encode(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Mapping):
            return {str(key): self.encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.encode(item) for item in value]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        tobytes = getattr(value, "tobytes", None)
        if shape is not None and dtype is not None and callable(tobytes):
            return self._store_array(value, shape, dtype, tobytes)
        raise TypeError(f"RPC value is not JSON serializable: {type(value).__name__}")

    def close(self) -> None:
        root, self._root = self._root, None
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)

    def _store_array(self, value: Any, shape: Any, dtype: Any, tobytes: Any) -> dict[str, Any]:
        del value
        with self._lock:
            root = self._ensure_root()
            path = root / f"array-{self._next_id:08d}.bin"
            self._next_id += 1
            payload = tobytes(order="C")
            with path.open("wb") as stream:
                stream.write(payload)
        return {
            MEDIA_REF_KEY: {
                "version": 1,
                "path": str(path),
                "shape": [int(item) for item in shape],
                "dtype": str(dtype),
                "order": "C",
            }
        }

    def _ensure_root(self) -> Path:
        if self._root is None:
            parent = self._configured_root
            if parent is not None:
                parent.mkdir(parents=True, exist_ok=True)
            self._root = Path(
                tempfile.mkdtemp(
                    prefix="harness-vln-media-",
                    dir=str(parent) if parent is not None else None,
                )
            )
        return self._root


def decode_media_refs(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {MEDIA_REF_KEY}:
            descriptor = value[MEDIA_REF_KEY]
            if not isinstance(descriptor, Mapping) or descriptor.get("version") != 1:
                raise ValueError("invalid Harness array reference")
            if descriptor.get("order") != "C":
                raise ValueError("unsupported Harness array order")
            import numpy as np

            return np.memmap(
                str(descriptor["path"]),
                mode="r",
                dtype=str(descriptor["dtype"]),
                shape=tuple(int(item) for item in descriptor["shape"]),
                order="C",
            )
        return {str(key): decode_media_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_media_refs(item) for item in value]
    return value

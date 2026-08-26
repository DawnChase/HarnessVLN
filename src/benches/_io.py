from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from harness.errors import HarnessError


def load_json(path: Path) -> Any:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return json.load(handle)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"failed to read dataset file {path}: {error}") from error


def require_fields(value: dict[str, Any], fields: set[str], context: str) -> None:
    missing = sorted(fields - value.keys())
    if missing:
        raise HarnessError(f"{context} is missing fields: {missing}")

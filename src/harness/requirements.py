from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from harness.errors import HarnessError
from schemas import NavigationProfile


class RequirementMismatch(HarnessError):
    pass


def check_navigation_requirements(
    owner: str,
    requirements: Mapping[str, Any],
    profile: NavigationProfile,
) -> None:
    offered = profile.as_dict()
    required_channels = set(requirements.get("observation_channels", ()))
    available_channels = set(profile.observation_channels)
    missing = sorted(required_channels - available_channels)
    if missing:
        raise RequirementMismatch(
            f"{owner} requires unavailable observation channels: {missing}"
        )

    for section in ("motion", "camera"):
        required = requirements.get(section, {})
        if not isinstance(required, Mapping):
            raise RequirementMismatch(f"{owner} requirement {section} must be a mapping")
        _check_mapping(owner, section, required, offered[section])


def _check_mapping(
    owner: str,
    path: str,
    required: Mapping[str, Any],
    offered: Mapping[str, Any],
) -> None:
    for key, expected in required.items():
        item_path = f"{path}.{key}"
        if key not in offered:
            raise RequirementMismatch(f"{owner} requires missing {item_path}")
        actual = offered[key]
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                raise RequirementMismatch(
                    f"{owner} requires {item_path}={dict(expected)!r}, got {actual!r}"
                )
            _check_mapping(owner, item_path, expected, actual)
        elif key == "actions" and isinstance(expected, Sequence) and not isinstance(
            expected, str
        ):
            missing = sorted(set(expected) - set(actual))
            if missing:
                raise RequirementMismatch(
                    f"{owner} requires unavailable {item_path}: {missing}"
                )
        elif actual != expected:
            raise RequirementMismatch(
                f"{owner} requires {item_path}={expected!r}, got {actual!r}"
            )

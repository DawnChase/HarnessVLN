from __future__ import annotations

import importlib
import pickle
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from envs.habitat import HabitatEnvironment, create_native_session
from harness.config import load_symbol
from schemas import EnvironmentEpisode


class GOATHabitatEnvironment(HabitatEnvironment):
    def __init__(self, episode: EnvironmentEpisode, **kwargs: Any) -> None:
        channels = list(kwargs.pop("observation_channels", ("rgb", "gps", "compass")))
        if "goal_image" not in channels:
            channels.append("goal_image")
        super().__init__(episode, observation_channels=channels, **kwargs)
        self._goal_specs = tuple(episode.setup.get("goal_specs", ()))
        if len(self._goal_specs) != len(self._goal_stream):
            raise ValueError("GOAT goal_specs must align with goal_stream")
        self._goal_images: dict[int, Any] = {}
        self._goal_results: list[dict[str, Any]] = []

    async def _observe(self, actor: str, arguments: dict[str, Any]) -> dict[str, Any]:
        observation = await super()._observe(actor, arguments)
        observation["channels"]["goal_image"] = self._goal_image()
        return observation

    async def _finish_goal(
        self, actor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        goal_index = self._goal_index
        transition = await super()._finish_goal(actor, arguments)
        if transition.get("accepted") is True:
            self._record_goal_result(goal_index)
        return transition

    def result(self) -> dict[str, Any]:
        return {**super().result(), "goal_results": list(self._goal_results)}

    def _goal_image(self) -> Any:
        spec = self._goal_specs[self._goal_index]
        if spec["modality"] != "image":
            return None
        if self._goal_index not in self._goal_images:
            image_goal = spec["image_goal"]
            rendered = self._session.sim.get_observations_at(
                position=image_goal["position"],
                rotation=image_goal["rotation"],
                keep_agent_at_new_pose=False,
            )
            self._goal_images[self._goal_index] = rendered["rgb"]
        return self._goal_images[self._goal_index]

    def _record_goal_result(self, goal_index: int) -> None:
        if len(self._goal_results) > goal_index:
            return
        success_metric = self._metrics.get("success", {})
        spl_metric = self._metrics.get("spl", {})
        distance_metric = self._metrics.get("distance_to_goal", {})
        successes = (
            success_metric.get("subtask_success", ())
            if isinstance(success_metric, Mapping)
            else ()
        )
        spls = (
            spl_metric.get("spl_by_subtask", ())
            if isinstance(spl_metric, Mapping)
            else ()
        )
        if goal_index >= len(successes) or goal_index >= len(spls):
            raise ValueError("GOAT native metrics do not contain the finished goal")
        result = {
            "goal_id": self._goal_stream[goal_index].goal_id,
            "modality": self._goal_stream[goal_index].modality,
            "success": float(successes[goal_index]),
            "spl": float(spls[goal_index]),
        }
        if isinstance(distance_metric, Mapping):
            distance = distance_metric.get("prev_distance_to_target")
            if isinstance(distance, (int, float)):
                result["distance_to_goal"] = float(distance)
        self._goal_results.append(result)


def from_episode(
    episode: EnvironmentEpisode,
    *,
    native_factory: str = "envs.goat:create_goat_session",
    native_factory_params: Mapping[str, Any] | None = None,
    **adapter_params: Any,
) -> GOATHabitatEnvironment:
    factory = load_symbol(native_factory)

    def build(private_episode: EnvironmentEpisode) -> Any:
        return factory(private_episode, **dict(native_factory_params or {}))

    return GOATHabitatEnvironment(episode, native_factory=build, **adapter_params)


def create_goat_session(
    episode: EnvironmentEpisode,
    *,
    goat_root: str | Path,
    habitat_root: str | Path,
    config_path: str | Path,
    scene_dataset_config: str | Path,
    config_values: Mapping[str, Any] | None = None,
) -> Any:
    return create_native_session(
        episode,
        config_path=config_path,
        config_loader="envs.goat:load_goat_config",
        config_loader_params={"goat_root": str(goat_root)},
        config_values=config_values,
        source_root=habitat_root,
        scene_dataset_config=scene_dataset_config,
    )


def load_goat_config(
    config_path: str,
    config_options: Sequence[Any],
    *,
    goat_root: str | Path,
) -> Any:
    _load_goat_runtime(goat_root)
    import habitat

    config = habitat.get_config(
        config_path=config_path,
        overrides=[str(option) for option in config_options],
    )
    _configure_goat_task(config)
    return config


def _configure_goat_task(config: Any) -> None:
    from habitat.config import read_write
    from habitat.config.default_structured_configs import (
        ActionConfig,
        MeasurementConfig,
    )
    from omegaconf import open_dict

    @dataclass
    class GoatDistanceToGoalConfig(MeasurementConfig):
        type: str = "GoatDistanceToGoal"
        distance_to: str = "VIEW_POINTS"

    @dataclass
    class GoatSuccessConfig(MeasurementConfig):
        type: str = "GoatSuccess"
        success_distance: float = 0.25

    @dataclass
    class GoatSPLConfig(MeasurementConfig):
        type: str = "GoatSPL"

    @dataclass
    class GoatSoftSPLConfig(MeasurementConfig):
        type: str = "GoatSoftSPL"

    @dataclass
    class SubtaskStopActionConfig(ActionConfig):
        type: str = "SubtaskStopAction"

    actions = dict(config.habitat.task.actions)
    actions["subtask_stop"] = SubtaskStopActionConfig()
    sensors = {
        name: config.habitat.task.lab_sensors[name]
        for name in ("gps_sensor", "compass_sensor")
    }
    measurements = {
        "distance_to_goal": GoatDistanceToGoalConfig(),
        "success": GoatSuccessConfig(),
        "spl": GoatSPLConfig(),
        "soft_spl": GoatSoftSPLConfig(),
    }
    with read_write(config), open_dict(config):
        config.habitat.task.actions = actions
        config.habitat.task.lab_sensors = sensors
        config.habitat.task.measurements = measurements


def _load_goat_runtime(goat_root: str | Path) -> None:
    root = Path(goat_root).expanduser().resolve()
    package_root = root / "goat_bench"
    if not (package_root / "dataset" / "goat_dataset.py").is_file():
        raise FileNotFoundError(f"GOAT-Bench source package not found under: {root}")

    package = sys.modules.get("goat_bench")
    if package is None:
        package = types.ModuleType("goat_bench")
        package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
        package.__package__ = "goat_bench"
        sys.modules["goat_bench"] = package

    utility_name = "goat_bench.utils.utils"
    if utility_name not in sys.modules:
        utilities = types.ModuleType(utility_name)

        def load_pickle(path: str | Path) -> Any:
            with Path(path).open("rb") as handle:
                return pickle.load(handle)

        utilities.load_pickle = load_pickle  # type: ignore[attr-defined]
        sys.modules[utility_name] = utilities

    for module_name in (
        "goat_bench.task.goat_task",
        "goat_bench.dataset.languagenav_dataset",
        "goat_bench.dataset.goat_dataset",
        "goat_bench.measurements.nav",
    ):
        module = importlib.import_module(module_name)
        module_path = Path(str(module.__file__)).resolve()
        if not module_path.is_relative_to(package_root):
            raise RuntimeError(
                f"GOAT module resolved outside configured source: {module_path}"
            )

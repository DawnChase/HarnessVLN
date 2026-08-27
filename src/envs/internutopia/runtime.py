from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

from harness.errors import HarnessError
from schemas import EnvironmentEpisode


ProjectKind = Literal["internnav", "vlnverse"]
EpisodeReviser = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _ProjectSpec:
    namespace: str
    default_config_module: str
    episode_module: str
    episode_factory: str
    extension_module: str
    model_source: str | None = None


_PROJECTS: dict[ProjectKind, _ProjectSpec] = {
    "internnav": _ProjectSpec(
        namespace="internnav",
        default_config_module="internnav.configs.evaluator.vln_default_config",
        episode_module="internnav.env.utils.episode_loader",
        episode_factory="generate_vln_episode",
        extension_module="internnav.env.utils.internutopia_extension",
        model_source="third_party/diffusion-policy",
    ),
    "vlnverse": _ProjectSpec(
        namespace="vlnverse",
        default_config_module="vlnverse.configs.evaluator.default_config",
        episode_module="vlnverse.evaluator.utils.eval",
        episode_factory="generate_episode",
        extension_module="vlnverse.projects.internutopia_vln_extension",
        model_source="vlnverse/model/encoder",
    ),
}


def create_vln_pe_session(
    episode: EnvironmentEpisode,
    *,
    scene_data_dir: str | Path,
    robot_usd_path: str | Path,
    eval_config: str | Path = "scripts/eval/configs/h1_cma_cfg.py",
    internutopia_root: str | Path = "cache/upstream/internutopia",
    project_root: str | Path = "cache/upstream/internnav",
    headless: bool = True,
    accept_eula: bool = False,
) -> Any:
    return _create_session(
        episode,
        project="internnav",
        scene_data_dir=scene_data_dir,
        robot_usd_path=robot_usd_path,
        eval_config=eval_config,
        internutopia_root=internutopia_root,
        project_root=project_root,
        headless=headless,
        accept_eula=accept_eula,
    )


def create_vlnverse_session(
    episode: EnvironmentEpisode,
    *,
    scene_data_dir: str | Path,
    robot_usd_path: str | Path,
    eval_config: str | Path = (
        "scripts/eval/configs/h1_internvla_n1_cfg_vlnverse_coarse.py"
    ),
    internutopia_root: str | Path = "cache/upstream/internutopia",
    project_root: str | Path = "cache/upstream/vlnverse",
    headless: bool = True,
    accept_eula: bool = False,
) -> Any:
    return _create_session(
        episode,
        project="vlnverse",
        scene_data_dir=scene_data_dir,
        robot_usd_path=robot_usd_path,
        eval_config=eval_config,
        internutopia_root=internutopia_root,
        project_root=project_root,
        headless=headless,
        accept_eula=accept_eula,
    )


def build_native_config(
    episode: EnvironmentEpisode,
    *,
    project: ProjectKind,
    scene_data_dir: str | Path,
    robot_usd_path: str | Path,
    eval_config: str | Path,
    internutopia_root: str | Path,
    project_root: str | Path,
    headless: bool = True,
) -> Any:
    """Build one official InternUtopia episode without starting SimulationApp."""

    spec = _PROJECTS[project]
    source_root = _require_directory(project_root, f"{project} source root")
    utopia_root = _require_directory(internutopia_root, "InternUtopia source root")
    if not (source_root / spec.namespace).is_dir():
        raise HarnessError(
            f"{project} source root does not contain {spec.namespace}/: {source_root}"
        )
    if not (utopia_root / "internutopia").is_dir():
        raise HarnessError(
            f"InternUtopia source root does not contain internutopia/: {utopia_root}"
        )
    _activate_sources(utopia_root, source_root, spec)

    config_path = _source_path(eval_config, source_root)
    scene_root = _require_directory(scene_data_dir, f"{project} scene data")
    robot_path = _require_file(robot_usd_path, f"{project} robot USD")
    policy_path = robot_path.parent / "policy/move_by_speed/h1_loco_jit_policy.pt"
    _require_file(policy_path, f"{project} H1 locomotion policy")

    eval_cfg = _load_eval_config(config_path)
    eval_cfg = _override_eval_config(
        eval_cfg,
        episode,
        scene_root=scene_root,
        robot_path=robot_path,
        headless=headless,
    )
    try:
        config_factory = getattr(
            importlib.import_module(spec.default_config_module), "get_config"
        )
        final_cfg = config_factory(eval_cfg)
    except SystemExit as error:
        raise HarnessError(
            f"{project} rejected evaluator config {config_path}"
        ) from error

    final_cfg.task.task_settings["env_num"] = 1
    final_cfg.task.task_settings["use_distributed"] = False
    final_cfg.task.task_settings["proc_num"] = 1
    final_cfg.env.env_settings.pop("distribution_config", None)
    final_cfg.env.env_settings["headless"] = headless

    reviser, skip_list = _episode_rules(project)
    robot_offset = final_cfg.dataset.dataset_settings["robot_offset"]
    native_episode, scan = prepare_native_episode(
        episode,
        dataset_type=str(episode.setup["dataset_type"]),
        robot_offset=robot_offset,
        reviser=reviser,
        skip_trajectories=skip_list,
    )
    path_key = str(episode.setup["path_key"])
    loader = SimpleNamespace(
        resumed_path_key_list=[path_key],
        path_key_data={path_key: native_episode},
        path_key_scan={path_key: scan},
        task_name=final_cfg.task.task_name,
    )
    episode_factory = getattr(
        importlib.import_module(spec.episode_module), spec.episode_factory
    )
    if project == "internnav":
        native_episodes = episode_factory(loader, final_cfg.task)
    else:
        native_episodes = episode_factory(loader, final_cfg)
    if len(native_episodes) != 1:
        raise HarnessError(
            f"{project} generated {len(native_episodes)} native episodes for one case"
        )

    from internutopia.core.config import Config, SimConfig

    return Config(
        simulator=SimConfig(**final_cfg.env.env_settings),
        env_num=1,
        env_offset_size=final_cfg.task.task_settings["offset_size"],
        task_configs=native_episodes,
    )


def prepare_native_episode(
    environment_episode: EnvironmentEpisode,
    *,
    dataset_type: str,
    robot_offset: Sequence[float],
    reviser: EpisodeReviser | None = None,
    skip_trajectories: Sequence[Any] = (),
) -> tuple[dict[str, Any], str]:
    """Apply the coordinate conversion used by the pinned upstream loaders."""

    raw = environment_episode.setup.get("native_episode")
    if not isinstance(raw, Mapping):
        raise HarnessError("Isaac case has no native_episode mapping")
    episode = copy.deepcopy(dict(raw))
    trajectory_id = episode.get("trajectory_id")
    episode_id = episode.get("episode_id")
    path_key = f"{trajectory_id}_{episode_id}"
    if path_key != str(environment_episode.setup.get("path_key")):
        raise HarnessError(
            "Isaac episode path_key mismatch: expected "
            f"{environment_episode.setup.get('path_key')}, got {path_key}"
        )
    if trajectory_id in skip_trajectories:
        raise HarnessError(f"Isaac upstream skip list excludes trajectory {trajectory_id}")

    start = _numeric_vector(episode.get("start_position"), 3, "start_position")
    rotation = _numeric_vector(episode.get("start_rotation"), 4, "start_rotation")
    episode["original_start_position"] = list(start)
    episode["original_start_rotation"] = list(rotation)
    if dataset_type == "mp3d":
        scene_id = episode.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise HarnessError("MP3D episode has no scene_id")
        parts = Path(scene_id).parts
        if len(parts) < 2:
            raise HarnessError(f"invalid MP3D scene_id: {scene_id}")
        scan = parts[-2]
        x, z, y = start
        episode["start_position"] = [x, -y, z]
        r1, r2, r3, r4 = rotation
        episode["start_rotation"] = _rotate_quaternion_z_90(
            [-r4, r1, r3, -r2]
        )
        if "reference_path" in episode:
            episode["reference_path"] = [
                [point[0], -point[2], point[1]]
                for point in episode["reference_path"]
            ]
    elif dataset_type in {"kujiale", "grscene"}:
        scan_value = episode.get("scan")
        if not isinstance(scan_value, str) or not scan_value:
            raise HarnessError(f"{dataset_type} episode has no scan")
        scan = scan_value
        episode["start_position"] = list(start)
        episode["start_rotation"] = list(rotation)
    else:
        raise HarnessError(f"unsupported InternUtopia dataset type: {dataset_type}")

    if reviser is not None:
        episode = reviser(episode)
    offset = _numeric_vector(robot_offset, 3, "robot_offset")
    episode["start_position"] = _add_vectors(episode["start_position"], offset)
    if "reference_path" in episode:
        episode["reference_path"] = [
            _add_vectors(point, offset) for point in episode["reference_path"]
        ]
    instruction = episode.get("instruction")
    if isinstance(instruction, dict):
        instruction.setdefault("instruction_tokens", [])
    return episode, scan


def resource_status(
    *,
    scene_data_dir: str | Path,
    robot_usd_path: str | Path,
    eval_config: str | Path,
    internutopia_root: str | Path,
    project_root: str | Path,
) -> dict[str, bool]:
    source_root = Path(project_root).expanduser().resolve()
    robot_path = Path(robot_usd_path).expanduser().resolve()
    return {
        "internutopia_root": Path(internutopia_root).expanduser().resolve().is_dir(),
        "project_root": source_root.is_dir(),
        "eval_config": _source_path(eval_config, source_root, require=False).is_file(),
        "scene_data_dir": Path(scene_data_dir).expanduser().resolve().is_dir(),
        "robot_usd_path": robot_path.is_file(),
        "locomotion_policy": (
            robot_path.parent / "policy/move_by_speed/h1_loco_jit_policy.pt"
        ).is_file(),
    }


def _create_session(
    episode: EnvironmentEpisode,
    *,
    project: ProjectKind,
    scene_data_dir: str | Path,
    robot_usd_path: str | Path,
    eval_config: str | Path,
    internutopia_root: str | Path,
    project_root: str | Path,
    headless: bool,
    accept_eula: bool,
) -> Any:
    if accept_eula:
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    native_config = build_native_config(
        episode,
        project=project,
        scene_data_dir=scene_data_dir,
        robot_usd_path=robot_usd_path,
        eval_config=eval_config,
        internutopia_root=internutopia_root,
        project_root=project_root,
        headless=headless,
    )

    # Bootstrapping Kit exposes carb/omni before the extension registry is imported.
    import isaacsim  # type: ignore[import-not-found]  # noqa: F401

    extension_module = importlib.import_module(_PROJECTS[project].extension_module)
    extension_module.import_extensions()
    from internutopia.core.vec_env import Env

    return Env(native_config)


def _override_eval_config(
    eval_cfg: Any,
    episode: EnvironmentEpisode,
    *,
    scene_root: Path,
    robot_path: Path,
    headless: bool,
) -> Any:
    split = episode.task.public.get("split")
    if not isinstance(split, str) or not split:
        raise HarnessError("Isaac case has no public split")
    eval_cfg = eval_cfg.model_copy(deep=True)
    eval_cfg.task.task_name = _task_name(episode.task.task_id)
    eval_cfg.task.scene.scene_data_dir = str(scene_root)
    eval_cfg.task.robot_usd_path = str(robot_path)
    eval_cfg.task.task_settings.update(
        {"env_num": 1, "use_distributed": False, "proc_num": 1}
    )
    eval_cfg.env.env_settings["headless"] = headless
    eval_cfg.env.env_settings.pop("distribution_config", None)
    eval_cfg.dataset.dataset_type = str(episode.setup["dataset_type"])
    eval_cfg.dataset.dataset_settings.update(
        {
            "base_data_dir": str(episode.setup["dataset_root"]),
            "split_data_types": [split],
            "filter_same_trajectory": False,
            "filter_stairs": False,
            "run_type": "eval",
            "retry_list": [],
        }
    )
    return eval_cfg


def _load_eval_config(path: Path) -> Any:
    module_name = "_harness_eval_" + hashlib.sha256(
        str(path).encode("utf-8")
    ).hexdigest()[:16]
    module = _load_module(module_name, path)
    value = getattr(module, "eval_cfg", None)
    if value is None or not hasattr(value, "model_copy"):
        raise HarnessError(f"evaluator config has no Pydantic eval_cfg: {path}")
    return value.model_copy(deep=True)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HarnessError(f"cannot load Python config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _episode_rules(
    project: ProjectKind,
) -> tuple[EpisodeReviser, Sequence[Any]]:
    if project == "internnav":
        module = importlib.import_module(
            "internnav.env.utils.episode_loader.dataset_utils"
        )
    else:
        module = importlib.import_module(
            "vlnverse.projects.dataloader.data_reviser"
        )
    return module.revise_one_data, tuple(module.skip_list)


def _activate_sources(
    internutopia_root: Path, project_root: Path, spec: _ProjectSpec
) -> None:
    paths = [internutopia_root, project_root]
    if spec.model_source is not None:
        model_source = project_root / spec.model_source
        if model_source.is_dir():
            paths.append(model_source)
    for path in reversed(paths):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _source_path(
    value: str | Path, source_root: Path, *, require: bool = True
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source_root / path
    path = path.resolve()
    return _require_file(path, "evaluator config") if require else path


def _require_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise HarnessError(f"{label} not found: {path}")
    return path


def _require_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise HarnessError(f"{label} not found: {path}")
    return path


def _numeric_vector(value: Any, size: int, label: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise HarnessError(f"{label} must be a numeric vector")
    if len(value) != size:
        raise HarnessError(f"{label} must contain {size} values")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise HarnessError(f"{label} must be a numeric vector") from error


def _add_vectors(left: Sequence[Any], right: Sequence[float]) -> list[float]:
    values = _numeric_vector(left, len(right), "episode coordinate")
    return [value + offset for value, offset in zip(values, right)]


def _rotate_quaternion_z_90(rotation: Sequence[float]) -> list[float]:
    w1, x1, y1, z1 = rotation
    root = math.sqrt(0.5)
    w2, x2, y2, z2 = root, 0.0, 0.0, root
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _task_name(case_id: str) -> str:
    return "harness_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)

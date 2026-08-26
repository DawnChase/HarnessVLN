from __future__ import annotations

import asyncio

import pytest

from agents import PassthroughVLNAgent
from envs import DummyNavigationEnvironment
from harness import NavigationHarness, NavigationStack
from harness.config import ComponentSpec, load_config, overlay
from harness.errors import HarnessError
from harness.requirements import RequirementMismatch, check_navigation_requirements
from schemas import MotionProfile, NavGoal, NavigationProfile, NavTask
from vln import DummyVLNNavigator


def test_yaml_overlay_recurses_replaces_lists_and_null_disables_plugin(tmp_path) -> None:
    base = tmp_path / "base.yaml"
    experiment = tmp_path / "experiment.yaml"
    base.write_text(
        """
benchmark:
  factory: benches.fake:Benchmark
  params: {split: val}
stack:
  agent:
    factory: agents.passthrough:PassthroughVLNAgent
    params: {poll_period_s: 0.1}
  environment: {factory: envs.fake:Environment}
  vln: {factory: vln.fake:Navigator, params: {channels: [rgb, depth]}}
  memory: {factory: memory.fake:Memory}
runner: {parallelism: 4, timeout_s: 10, seed: 1}
output: {tags: [base]}
"""
    )
    experiment.write_text(
        """
stack:
  agent: {params: {poll_period_s: 0.01}}
  vln: {params: {channels: [rgb]}}
  memory: null
runner: {parallelism: 2}
output: {tags: [experiment]}
"""
    )

    resolved = load_config((base, experiment))
    assert resolved.data["stack"]["agent"]["factory"].endswith(
        "PassthroughVLNAgent"
    )
    assert resolved.data["stack"]["agent"]["params"]["poll_period_s"] == 0.01
    assert resolved.data["stack"]["vln"]["params"]["channels"] == ["rgb"]
    assert resolved.data["stack"]["memory"] is None
    assert resolved.data["runner"]["timeout_s"] == 10
    assert resolved.data["output"]["tags"] == ["experiment"]
    assert len(resolved.digest) == 64


def test_overlay_does_not_mutate_inputs() -> None:
    base = {"nested": {"a": 1}, "items": [1]}
    override_value = {"nested": {"b": 2}, "items": [2]}
    assert overlay(base, override_value) == {
        "nested": {"a": 1, "b": 2},
        "items": [2],
    }
    assert base == {"nested": {"a": 1}, "items": [1]}


def test_overlay_replaces_parameters_when_component_factory_changes() -> None:
    base = {
        "vln": {
            "factory": "vln.old:Navigator",
            "params": {"old_only": True, "shared": "old"},
        }
    }
    replacement = {
        "vln": {
            "factory": "vln.new:Navigator",
            "params": {"shared": "new"},
        }
    }

    assert overlay(base, replacement) == {
        "vln": {
            "factory": "vln.new:Navigator",
            "params": {"shared": "new"},
        }
    }

    same_factory = {"vln": {"params": {"shared": "patched"}}}
    assert overlay(base, same_factory)["vln"]["params"] == {
        "old_only": True,
        "shared": "patched",
    }


def test_import_path_factory_creates_plugin_without_registry_edit() -> None:
    spec = ComponentSpec(
        "agents.passthrough:PassthroughVLNAgent", {"poll_period_s": 0.25}
    )
    agent = spec.create()
    assert isinstance(agent, PassthroughVLNAgent)
    assert agent.poll_period_s == 0.25
    with pytest.raises(HarnessError, match="cannot load factory"):
        ComponentSpec("missing.module:Factory", {}).create()


def test_component_serial_metadata_is_not_forwarded_to_factory() -> None:
    spec = ComponentSpec.from_config(
        {
            "factory": "agents.passthrough:PassthroughVLNAgent",
            "params": {"poll_period_s": 0.25},
            "serial": True,
        }
    )

    assert spec.serial is True
    assert isinstance(spec.create(), PassthroughVLNAgent)


def test_navigation_requirements_compare_semantics_not_only_tool_name() -> None:
    profile = NavigationProfile(
        frozenset({"rgb", "depth", "pose"}),
        MotionProfile(
            "nav.move.discrete",
            frozenset({"forward", "turn_left", "turn_right"}),
            frame="habitat_world",
            units="meters_degrees",
            forward_m=0.25,
            turn_deg=15,
        ),
        camera={"height": 480, "width": 640},
    )
    check_navigation_requirements(
        "model",
        {
            "observation_channels": ["rgb", "depth"],
            "motion": {
                "tool": "nav.move.discrete",
                "actions": ["forward", "turn_left"],
                "forward_m": 0.25,
                "turn_deg": 15,
            },
            "camera": {"height": 480},
        },
        profile,
    )
    with pytest.raises(RequirementMismatch, match="turn_deg=30"):
        check_navigation_requirements(
            "model", {"motion": {"turn_deg": 30}}, profile
        )


def test_requirement_mismatch_prevents_vln_and_agent_start() -> None:
    class IncompatibleVLN(DummyVLNNavigator):
        requirements = {"motion": {"forward_m": 0.25}}

        def __init__(self):
            super().__init__()
            self.start_calls = 0

        async def start(self, task, tools):
            self.start_calls += 1
            return await super().start(task, tools)

    class TrackingAgent:
        required_tools: frozenset[str] = frozenset()

        def __init__(self):
            self.run_calls = 0

        async def run(self, context):
            self.run_calls += 1
            await context.nav.stop("completed")

    async def scenario():
        goal = NavGoal("goal", "go")
        navigator = IncompatibleVLN()
        agent = TrackingAgent()
        result = await NavigationHarness(timeout_s=1).run_task(
            NavTask("mismatch", goal),
            NavigationStack(
                agent,
                DummyNavigationEnvironment((goal,), targets=(0,)),
                vln=navigator,
            ),
        )
        assert result.terminal.status == "failed"
        assert "forward_m=0.25" in result.terminal.reason
        assert navigator.start_calls == 0
        assert agent.run_calls == 0

    asyncio.run(scenario())

from __future__ import annotations

from harness.app import ConfiguredStackFactory, _stack_factory
from harness.config import ComponentSpec


def spec(name: str, *, serial: bool = False) -> ComponentSpec:
    return ComponentSpec(name, {}, serial)


def test_stack_factory_propagates_component_serial_constraint() -> None:
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.isaac_vln_pe:from_case", serial=True),
    )

    assert factory.requires_serial is True


def test_stack_factory_allows_parallel_read_only_components() -> None:
    factory = ConfiguredStackFactory(
        agent=spec("agents.passthrough:PassthroughVLNAgent"),
        environment=spec("envs.dummy:from_case"),
    )

    assert factory.requires_serial is False


def test_agent_and_memory_fragments_create_independent_plugins() -> None:
    from harness.config import load_config

    resolved = load_config(
        (
            "config/benches/dummy.yaml",
            "config/runs/dummy_passthrough.yaml",
            "config/agents/subtask.yaml",
            "config/memory/dummy_landmark.yaml",
        )
    )
    factory = _stack_factory(resolved.data["stack"])
    stack = factory(next(iter(factory_case_source().cases())))

    assert type(stack.agent).__name__ == "SubtaskNavigationAgent"
    assert type(stack.memory).__name__ == "DummyLandmarkMemory"
    assert factory.requires_serial is True
    assert "spatial.search" in stack.agent.required_tools


def factory_case_source():
    from benches.dummy import DummyBenchmark

    return DummyBenchmark(
        split="fixture",
        cases=[{"task_id": "case", "instruction": "Go.", "target": 0}],
    )

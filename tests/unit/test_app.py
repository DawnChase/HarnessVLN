from __future__ import annotations

from harness.app import ConfiguredStackFactory
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

from __future__ import annotations

from pathlib import Path


PLUGIN_ROOTS = {
    "agents": {"__init__.py", "_jobs.py"},
    "benches": {"__init__.py", "_io.py", "base.py"},
    "envs": {"__init__.py"},
    "memory": {"__init__.py"},
    "vln": {"__init__.py", "rpc.py", "worker.py"},
}


def test_concrete_plugins_live_below_their_domain_package() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"

    for domain, allowed_base_files in PLUGIN_ROOTS.items():
        domain_root = source_root / domain
        direct_python_files = {path.name for path in domain_root.glob("*.py")}
        assert direct_python_files <= allowed_base_files, (
            f"concrete {domain} plugins require their own subdirectory: "
            f"{sorted(direct_python_files - allowed_base_files)}"
        )

        plugin_directories = sorted(
            path
            for path in domain_root.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        )
        for plugin_directory in plugin_directories:
            assert (plugin_directory / "__init__.py").is_file()
            assert any(
                path.suffix == ".py" and path.name != "__init__.py"
                for path in plugin_directory.iterdir()
            ), f"plugin directory has no implementation: {plugin_directory}"

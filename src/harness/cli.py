from __future__ import annotations

import argparse

from harness.app import run_config_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HarnessVLN benchmark")
    parser.add_argument("configs", nargs="+", help="YAML files in overlay order")
    arguments = parser.parse_args()
    summary, manifest = run_config_sync(arguments.configs)
    failures = sum(record.error is not None for record in summary.records)
    print(
        f"{summary.benchmark}/{summary.split}: "
        f"{len(summary.records)} cases, {failures} runner errors"
    )
    print(manifest.resolve())


if __name__ == "__main__":
    main()

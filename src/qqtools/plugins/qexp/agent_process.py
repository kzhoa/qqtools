"""Retired standalone-agent entrypoint retained for clear upgrade diagnostics."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Parse the former process arguments so old invocations receive a targeted error."""
    parser = argparse.ArgumentParser(description="retired qexp standalone agent process")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--machine-runtime-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise RuntimeError("standalone agent runtime was removed; run 'qexp agent migrate-project'.")


if __name__ == "__main__":
    raise SystemExit(main())

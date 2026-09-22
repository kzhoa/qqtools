"""Retired standalone-agent entrypoint retained for clear upgrade diagnostics."""

from __future__ import annotations

import argparse
import shlex


def build_parser() -> argparse.ArgumentParser:
    """Parse the former process arguments so old invocations receive a targeted error."""
    parser = argparse.ArgumentParser(description="retired qexp standalone agent process")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--machine-runtime-root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = shlex.join(
        [
            "qexp",
            "admin",
            "migrate",
            "agent",
            "--project",
            args.shared_root,
            "--machine",
            args.machine,
        ]
    )
    raise RuntimeError(f"standalone agent runtime was removed; run '{command}'.")


if __name__ == "__main__":
    raise SystemExit(main())

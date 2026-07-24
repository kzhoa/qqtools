#!/usr/bin/env python3
"""Small local qualification probe for qexp atomic JSON and exclusive create."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from qqtools.plugins.qexp.runtime.store import CASConflict, create_if_absent, read_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=None)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.directory) as directory:
        root = Path(directory)
        path = root / "exclusive.json"
        create_if_absent(path, {"winner": 1})
        try:
            create_if_absent(path, {"winner": 2})
        except CASConflict:
            pass
        else:
            raise SystemExit("exclusive create did not reject the second writer")
        if read_json(path) != {"winner": 1}:
            raise SystemExit("atomic JSON readback mismatch")
    print("qexp shared-filesystem local primitives: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

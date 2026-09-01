from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("point", ["file_fsync", "replace", "directory_fsync"])
def test_atomic_replace_crash_boundaries_never_expose_partial_json(
    tmp_path: Path,
    checkout_subprocess_env: dict[str, str],
    point: str,
) -> None:
    record = tmp_path / "record.json"
    atomic_replace(record, {"value": "old"})
    writer = """
import os
import sys
from pathlib import Path
from qqtools.plugins.qexp.runtime import store

path = Path(sys.argv[1])
point = sys.argv[2]
original_fsync = store.os.fsync
original_replace = store.os.replace
fsync_calls = 0

def crash_fsync(descriptor):
    global fsync_calls
    fsync_calls += 1
    if point == "file_fsync" and fsync_calls == 1:
        os._exit(17)
    if point == "directory_fsync" and fsync_calls == 2:
        os._exit(17)
    return original_fsync(descriptor)

def crash_replace(source, destination):
    if point == "replace":
        os._exit(17)
    return original_replace(source, destination)

store.os.fsync = crash_fsync
store.os.replace = crash_replace
store.atomic_replace(path, {"value": "new"})
"""
    process = subprocess.run(
        [sys.executable, "-c", writer, str(record), point],
        env=checkout_subprocess_env,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    assert process.returncode == 17, process.stderr
    assert read_json(record) in ({"value": "old"}, {"value": "new"})

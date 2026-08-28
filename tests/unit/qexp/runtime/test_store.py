import os
import stat
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.store import (
    CASConflict,
    atomic_replace,
    cas_update,
    create_if_absent,
    read_json,
)


def test_create_if_absent_and_cas(tmp_path: Path):
    path = tmp_path / "record.json"
    create_if_absent(path, {"meta": {"revision": 1}, "value": 1})
    with pytest.raises(CASConflict):
        create_if_absent(path, {"meta": {"revision": 1}, "value": 2})
    value = read_json(path)
    value["value"] = 2
    cas_update(path, 1, value)
    assert read_json(path)["meta"]["revision"] == 2


def test_atomic_replace_flushes_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flushed_types: list[str] = []

    def record_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        flushed_types.append("directory" if stat.S_ISDIR(mode) else "file")

    monkeypatch.setattr(
        "qqtools.plugins.qexp.runtime.store.os.fsync",
        record_fsync,
    )

    path = tmp_path / "record.json"
    atomic_replace(path, {"value": 1})

    assert read_json(path) == {"value": 1}
    assert flushed_types == ["file", "directory"]

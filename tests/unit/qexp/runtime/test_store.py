import os
import stat
from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.store import (
    CASConflict,
    atomic_replace,
    cas_update,
    create_if_absent,
    fenced_mutations,
    iter_json,
    read_json,
)


def test_create_if_absent_and_cas(tmp_path: Path):
    path = tmp_path / "record.json"
    create_if_absent(path, {"meta": {"revision": 1}, "value": 1})
    with pytest.raises(CASConflict):
        create_if_absent(path, {"meta": {"revision": 1}, "value": 2})
    value = read_json(path)
    assert value == {"meta": {"revision": 1}, "value": 1}
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


def test_iter_json_returns_sorted_regular_json_files(tmp_path: Path) -> None:
    (tmp_path / "b.json").touch()
    (tmp_path / "a.json").touch()
    (tmp_path / "ignored.txt").touch()
    (tmp_path / "directory.json").mkdir()
    (tmp_path / "link.json").symlink_to(tmp_path / "a.json")

    assert iter_json(tmp_path) == [tmp_path / "a.json", tmp_path / "b.json"]


def test_mutation_fence_is_scoped_and_checked_after_temporary_write(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    path = shared / "record.json"
    atomic_replace(path, {"value": "original"})
    revoked = False

    def fence():
        if revoked:
            raise RuntimeError("epoch revoked")

    def revoke_before_replace(_stat):
        nonlocal revoked
        revoked = True

    with fenced_mutations(shared, fence):
        with pytest.raises(RuntimeError, match="epoch revoked"):
            atomic_replace(path, {"value": "stale"}, before_replace=revoke_before_replace)
        assert read_json(path) == {"value": "original"}
        with pytest.raises(RuntimeError, match="epoch revoked"):
            create_if_absent(shared / "new.json", {"value": "stale"})
        assert not (shared / "new.json").exists()
        atomic_replace(tmp_path / "local.json", {"value": "local"})
    atomic_replace(path, {"value": "current"})
    assert read_json(path) == {"value": "current"}
    assert sorted(item.name for item in shared.iterdir()) == ["record.json"]

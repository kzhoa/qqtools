from threading import Event, Thread

from qqtools.plugins.qexp.runtime.locks import (
    group_writer_lock,
    is_schema_narrow_protocol_active,
    schema_lock,
    schema_reader_lock,
    schema_writer_lock,
    task_writer_lock,
    task_locks,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace


def test_schema_readers_share_while_exclusive_writer_waits(tmp_path) -> None:
    root = tmp_path / ".qexp"
    reader_entered = Event()
    release_reader = Event()
    writer_entered = Event()

    def hold_reader() -> None:
        with schema_reader_lock(root):
            reader_entered.set()
            release_reader.wait(timeout=2)

    def hold_writer() -> None:
        with schema_lock(root):
            writer_entered.set()

    reader = Thread(target=hold_reader)
    reader.start()
    assert reader_entered.wait(timeout=1)
    with schema_reader_lock(root, blocking=False) as acquired:
        assert acquired
    writer = Thread(target=hold_writer)
    writer.start()
    assert not writer_entered.wait(timeout=0.05)
    release_reader.set()
    reader.join(timeout=2)
    writer.join(timeout=2)
    assert writer_entered.is_set()


def test_task_locks_reject_duplicate_identity(tmp_path) -> None:
    root = tmp_path / ".qexp"
    try:
        with task_locks(root, ["task-a", "task-a"]):
            pass
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate Task identities must be rejected")


def test_schema_writer_lock_is_reentrant_for_one_writer(tmp_path) -> None:
    class Config:
        shared_root = tmp_path / ".qexp"

    with schema_writer_lock(Config()):
        with schema_writer_lock(Config(), blocking=False) as acquired:
            assert acquired


def test_failed_nonblocking_schema_writer_lock_is_not_reentrant(tmp_path) -> None:
    class Config:
        shared_root = tmp_path / ".qexp"

    with schema_lock(Config.shared_root):
        with schema_writer_lock(Config(), blocking=False) as acquired:
            assert not acquired
            with schema_writer_lock(Config(), blocking=False) as nested:
                assert not nested


def test_malformed_narrow_activation_state_falls_back_to_wide_fence(tmp_path) -> None:
    class Config:
        shared_root = tmp_path / ".qexp"

    atomic_replace(
        Config.shared_root / "schema" / "version.json",
        {"schema": {"required_capabilities": ["group-ready-members-v1"]}},
    )
    atomic_replace(
        Config.shared_root / "indexes" / "ready" / "group-members" / "state.json",
        {"group_ready_members": 1},
    )

    assert is_schema_narrow_protocol_active(Config()) is False
    with schema_writer_lock(Config(), blocking=False) as acquired:
        assert acquired


def test_incomplete_active_member_state_does_not_select_narrow_fence(tmp_path) -> None:
    class Config:
        shared_root = tmp_path / ".qexp"

    atomic_replace(
        Config.shared_root / "schema" / "version.json",
        {"schema": {"required_capabilities": ["group-ready-members-v1"]}},
    )
    atomic_replace(
        Config.shared_root / "indexes" / "ready" / "group-members" / "state.json",
        {"group_ready_members": {"state": "active"}},
    )

    assert is_schema_narrow_protocol_active(Config()) is False


def test_group_writer_waits_for_schema_mutation(tmp_path, monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import locks

    class Config:
        shared_root = tmp_path / ".qexp"

    monkeypatch.setattr(locks, "is_schema_narrow_protocol_active", lambda _cfg: True)
    entered = Event()
    release = Event()

    def hold_schema_mutation() -> None:
        with schema_lock(Config.shared_root):
            entered.set()
            release.wait(timeout=2)

    thread = Thread(target=hold_schema_mutation)
    thread.start()
    assert entered.wait(timeout=1)
    with group_writer_lock(Config(), "exp", blocking=False) as acquired:
        assert not acquired
    release.set()
    thread.join(timeout=2)


def test_task_writer_waits_for_schema_mutation(tmp_path, monkeypatch) -> None:
    from qqtools.plugins.qexp.runtime import locks

    class Config:
        shared_root = tmp_path / ".qexp"

    monkeypatch.setattr(locks, "is_schema_narrow_protocol_active", lambda _cfg: True)
    entered = Event()
    release = Event()

    def hold_schema_mutation() -> None:
        with schema_lock(Config.shared_root):
            entered.set()
            release.wait(timeout=2)

    thread = Thread(target=hold_schema_mutation)
    thread.start()
    assert entered.wait(timeout=1)
    with task_writer_lock(Config(), "task-a", None, blocking=False) as acquired:
        assert not acquired
    release.set()
    thread.join(timeout=2)

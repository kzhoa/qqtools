from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.runtime.claims import archive_claim
from qqtools.plugins.qexp.runtime.paths import attempt_path, shared_paths, task_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.running_publication import publish_running_registration
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _starting_attempt(tmp_path: Path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1")
    task = submit(cfg, ["echo", "ok"], working_dir=tmp_path)
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    attempt.phase = "starting"
    atomic_replace(path, attempt.to_dict())
    task = load_task(cfg, task.task_id)
    task.claim_control["active_claim"]["launch_state"] = "starting"
    save_task(cfg, task)
    registration = {
        "task_id": task.task_id,
        "attempt_id": attempt.attempt_id,
        "fencing_token": attempt.current_fencing_token,
        "reservation_id": attempt.reservation_id,
        "wrapper_pid": 101,
        "wrapper_start_time_ticks": 202,
        "process_group_id": 303,
        "process_group_start_time_ticks": 404,
        "process_created_at": "2026-09-29T00:00:00+00:00",
    }
    return cfg, path, registration


def test_running_publication_is_shared_only_and_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, path, registration = _starting_attempt(tmp_path)
    manifest = cfg.runtime_root / "processes" / f"{registration['attempt_id']}.json"
    real_open, real_stat, real_resolve = Path.open, Path.stat, Path.resolve

    def forbid_local(path: Path) -> None:
        assert not path.is_relative_to(cfg.runtime_root), f"shared publication accessed local path {path}"

    def checked_open(path: Path, *args, **kwargs):
        forbid_local(path)
        return real_open(path, *args, **kwargs)

    def checked_stat(path: Path, *args, **kwargs):
        forbid_local(path)
        return real_stat(path, *args, **kwargs)

    def checked_resolve(path: Path, *args, **kwargs):
        forbid_local(path)
        return real_resolve(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", checked_open)
        patch.setattr(Path, "stat", checked_stat)
        patch.setattr(Path, "resolve", checked_resolve)
        assert publish_running_registration(cfg, registration, manifest) is True
        attempt = AttemptRecord.from_dict(read_json(path))
        task = load_task(cfg, registration["task_id"])
        assert attempt.phase == "running"
        assert task.claim_control["active_claim"]["launch_state"] == "running"
        assert attempt.process["local_process_manifest"] == str(manifest)
        assert attempt.process["process_group_id"] == registration["process_group_id"]
        assert attempt.timestamps["process_created_at"] == registration["process_created_at"]
        before = path.read_bytes(), task_path(cfg.shared_root, task.task_id).read_bytes()

        def reject_unnecessary_write() -> None:
            raise AssertionError("already published registration attempted another mutation")

        assert (
            publish_running_registration(cfg, registration, manifest, mutation_fence=reject_unnecessary_write) is False
        )
        assert (path.read_bytes(), task_path(cfg.shared_root, task.task_id).read_bytes()) == before


@pytest.mark.parametrize("boundary", ["before_attempt", "after_attempt"])
def test_running_publication_replays_after_fenced_partial_write(tmp_path: Path, boundary: str) -> None:
    cfg, path, registration = _starting_attempt(tmp_path)
    manifest = cfg.runtime_root / "processes" / f"{registration['attempt_id']}.json"
    task_before = load_task(cfg, registration["task_id"])

    class Fenced(RuntimeError):
        pass

    def fence() -> None:
        if boundary == "before_attempt" or AttemptRecord.from_dict(read_json(path)).phase == "running":
            raise Fenced("controller epoch replaced")

    with pytest.raises(Fenced, match="epoch replaced"):
        publish_running_registration(cfg, registration, manifest, mutation_fence=fence)
    partial = AttemptRecord.from_dict(read_json(path))
    assert partial.phase == ("starting" if boundary == "before_attempt" else "running")
    assert load_task(cfg, registration["task_id"]).claim_control["active_claim"]["launch_state"] == "starting"
    assert publish_running_registration(cfg, registration, manifest) is (boundary == "before_attempt")
    task = load_task(cfg, registration["task_id"])
    assert task.claim_control["active_claim"]["launch_state"] == "running"
    assert task.meta["revision"] == task_before.meta["revision"] + 1
    assert AttemptRecord.from_dict(read_json(path)).phase == "running"


@pytest.mark.parametrize("conflict", ["token", "reservation", "process", "timestamp", "phase"])
def test_running_publication_rejects_conflicting_registration(tmp_path: Path, conflict: str) -> None:
    cfg, path, registration = _starting_attempt(tmp_path)
    attempt = AttemptRecord.from_dict(read_json(path))
    if conflict == "token":
        registration["fencing_token"] += 1
    elif conflict == "reservation":
        registration["reservation_id"] = "other-reservation"
    elif conflict == "process":
        attempt.process["process_group_id"] = 999
    elif conflict == "timestamp":
        attempt.timestamps["process_created_at"] = "2026-09-28T00:00:00+00:00"
    else:
        attempt.phase = "succeeded"
    atomic_replace(path, attempt.to_dict())
    task_file = task_path(cfg.shared_root, registration["task_id"])
    before = path.read_bytes(), task_file.read_bytes()

    def reject_write() -> None:
        raise AssertionError("stale registration attempted a mutation")

    assert (
        publish_running_registration(cfg, registration, cfg.runtime_root / "manifest.json", mutation_fence=reject_write)
        is False
    )
    assert (path.read_bytes(), task_file.read_bytes()) == before


@pytest.mark.parametrize(
    "conflict",
    [
        None,
        "launch",
        "token",
        "reservation",
        "machine",
        "claimed",
        "new_attempt",
        "process",
        "timestamp",
        "archive_missing",
        "blocked_reason",
    ],
)
@pytest.mark.parametrize("archive_directory", ["claim_archive", "claim_pending"])
def test_expired_launch_registration_preserves_evidence_without_restoring_authority(
    tmp_path: Path, conflict: str | None, archive_directory: str
) -> None:
    cfg, path, registration = _starting_attempt(tmp_path)
    task = load_task(cfg, registration["task_id"])
    claim = task.claim_control["active_claim"]
    claim["launch_id"] = "a" * 32
    attempt = AttemptRecord.from_dict(read_json(path))
    attempt.authorization["launch_id"] = claim["launch_id"]
    attempt.phase = "orphaned"
    if conflict == "launch":
        claim["launch_id"] = "b" * 32
    elif conflict == "token":
        claim["fencing_token"] += 1
    elif conflict == "reservation":
        claim["reservation_id"] = "other"
    elif conflict == "machine":
        claim["machine_name"] = "gpu-other"
    elif conflict == "claimed":
        claim["launch_state"] = "claimed"
    elif conflict == "process":
        attempt.process["wrapper_pid"] = 999
    elif conflict == "timestamp":
        attempt.timestamps["process_created_at"] = "2026-09-28T00:00:00+00:00"
    archive_claim(cfg, task.task_id, claim, "lease_expired")
    paths = shared_paths(cfg.shared_root)
    archived = paths["claim_archive"] / task.task_id / f"{claim['fencing_token']}.json"
    if conflict == "archive_missing":
        archived.unlink()
    elif archive_directory == "claim_pending":
        pending = paths["claim_pending"] / task.task_id / archived.name
        pending.parent.mkdir(parents=True, exist_ok=True)
        archived.rename(pending)
    task.claim_control["active_claim"] = None
    task.attempt_control["current_attempt_id"] = None
    task.state.update({"projection": "blocked", "reason": "orphaned_attempt_requires_recovery"})
    if conflict == "blocked_reason":
        task.state["reason"] = "other_blocked_state"
    if conflict == "new_attempt":
        task.attempt_control["next_attempt_number"] += 1
    atomic_replace(path, attempt.to_dict())
    save_task(cfg, task)
    task_file = task_path(cfg.shared_root, task.task_id)
    before = path.read_bytes(), task_file.read_bytes()
    manifest = cfg.runtime_root / "manifest.json"

    if conflict is None:

        class Fenced(RuntimeError):
            pass

        def fence() -> None:
            raise Fenced("epoch changed")

        with pytest.raises(Fenced, match="epoch changed"):
            publish_running_registration(cfg, registration, manifest, mutation_fence=fence)
        assert (path.read_bytes(), task_file.read_bytes()) == before

    assert publish_running_registration(cfg, registration, manifest) is False
    stored = AttemptRecord.from_dict(read_json(path))
    assert task_file.read_bytes() == before[1]
    assert stored.phase == "orphaned"
    assert stored.timestamps["running_at"] is None
    if conflict is not None:
        assert path.read_bytes() == before[0]
    else:
        assert stored.process["wrapper_pid"] == registration["wrapper_pid"]
        assert stored.process["process_group_id"] == registration["process_group_id"]
        assert stored.timestamps["process_created_at"] == registration["process_created_at"]

        def reject_replay_write() -> None:
            raise AssertionError("matching orphan evidence was rewritten")

        assert publish_running_registration(cfg, registration, manifest, mutation_fence=reject_replay_write) is False

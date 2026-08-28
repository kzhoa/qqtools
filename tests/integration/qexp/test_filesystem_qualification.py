from pathlib import Path

import pytest

from qqtools.plugins.qexp import (
    FilesystemProbeEvidence,
    init_shared_root,
    load_filesystem_qualification,
    record_filesystem_qualification,
    submit,
)
from qqtools.plugins.qexp.commands.group import change_worker
from qqtools.plugins.qexp.commands.task import offer
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.machine_runtime import MachineRuntime
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.reservations import reserved_gpu_ids
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.scheduler import (
    authorize_launch,
    claim_task,
    resume_starting_attempt,
)

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _passing_evidence() -> FilesystemProbeEvidence:
    return FilesystemProbeEvidence(
        "host-a",
        "host-b",
        exclusive_lock=True,
        atomic_replace=True,
        fsync_visibility=True,
        failure_cleanup=True,
    )


def _failing_evidence() -> FilesystemProbeEvidence:
    return FilesystemProbeEvidence(
        "host-a",
        "host-b",
        exclusive_lock=True,
        atomic_replace=False,
        fsync_visibility=True,
        failure_cleanup=True,
    )


def _offered_task_and_remote_cfg(tmp_path: Path):
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "gpu-1"
    )
    task = submit(cfg, ["echo", "ok"], group="exp", sharing_mode="spillover")
    change_worker(cfg, "exp", "gpu-2", "add")
    offer(cfg, task.task_id)
    remote_cfg = RootConfig(
        cfg.shared_root,
        cfg.project_root,
        "gpu-2",
        tmp_path / "gpu-2",
    )
    return cfg, remote_cfg, task


def test_cross_host_claim_fails_before_reservation_without_qualification(
    tmp_path: Path,
) -> None:
    _cfg, remote_cfg, task = _offered_task_and_remote_cfg(tmp_path)

    with pytest.raises(RuntimeError, match="qualification evidence is missing"):
        claim_task(remote_cfg, task.task_id, [0])

    assert reserved_gpu_ids(remote_cfg.runtime_root) == set()


def test_cross_host_claim_accepts_valid_deployment_qualification(tmp_path: Path) -> None:
    cfg, remote_cfg, task = _offered_task_and_remote_cfg(tmp_path)
    record_filesystem_qualification(cfg, _passing_evidence())

    attempt = claim_task(remote_cfg, task.task_id, [0])

    assert attempt is not None
    assert attempt.machine_name == "gpu-2"
    assert load_filesystem_qualification(remote_cfg).is_qualified is True


def test_local_claim_does_not_require_cross_host_qualification(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    task = submit(cfg, ["echo", "ok"])

    assert claim_task(cfg, task.task_id, [0]) is not None
    assert load_filesystem_qualification(cfg).is_qualified is False


def test_failed_probe_evidence_is_persisted_as_unqualified(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    evidence = FilesystemProbeEvidence(
        "host-a",
        "host-b",
        exclusive_lock=False,
        atomic_replace=True,
        fsync_visibility=True,
        failure_cleanup=True,
    )

    with pytest.raises(ValueError, match="exclusive lock failed"):
        record_filesystem_qualification(cfg, evidence)

    assert load_filesystem_qualification(cfg).is_qualified is False


def test_invalid_probe_field_types_fail_closed_without_writer_crash(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    evidence = FilesystemProbeEvidence(
        1,  # type: ignore[arg-type]
        "host-b",
        exclusive_lock=1,  # type: ignore[arg-type]
        atomic_replace=True,
        fsync_visibility=True,
        failure_cleanup=True,
    )

    with pytest.raises(ValueError, match="both host identities are required"):
        record_filesystem_qualification(cfg, evidence)

    assert load_filesystem_qualification(cfg).is_qualified is False


def test_failed_requalification_revokes_previous_success(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime"
    )
    record_filesystem_qualification(cfg, _passing_evidence())
    with pytest.raises(ValueError, match="atomic replace visibility failed"):
        record_filesystem_qualification(cfg, _failing_evidence())

    qualification = load_filesystem_qualification(cfg)
    assert qualification.is_qualified is False
    assert qualification.reasons == ("atomic replace visibility failed",)

    local_cfg = init_shared_root(
        cfg.shared_root,
        "gpu-local",
        runtime_root=tmp_path / "local-runtime",
    )
    local_task = submit(local_cfg, ["echo", "local"])
    assert claim_task(local_cfg, local_task.task_id, [0]) is not None


def test_revoked_qualification_blocks_launch_authorization(tmp_path: Path) -> None:
    cfg, remote_cfg, task = _offered_task_and_remote_cfg(tmp_path)
    record_filesystem_qualification(cfg, _passing_evidence())
    attempt = claim_task(remote_cfg, task.task_id, [0])
    assert attempt is not None
    with pytest.raises(ValueError, match="atomic replace visibility failed"):
        record_filesystem_qualification(cfg, _failing_evidence())

    is_authorized = authorize_launch(
        remote_cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
    )

    assert is_authorized is False
    assert reserved_gpu_ids(remote_cfg.runtime_root) == set()
    assert load_task(remote_cfg, task.task_id).state == {
        "projection": "cancelled",
        "reason": "filesystem_unqualified",
    }


def test_revoked_qualification_blocks_starting_recovery(tmp_path: Path) -> None:
    cfg, remote_cfg, task = _offered_task_and_remote_cfg(tmp_path)
    record_filesystem_qualification(cfg, _passing_evidence())
    attempt = claim_task(remote_cfg, task.task_id, [0])
    assert attempt is not None
    assert authorize_launch(
        remote_cfg,
        task.task_id,
        attempt.attempt_id,
        attempt.current_fencing_token,
    )
    with pytest.raises(ValueError, match="atomic replace visibility failed"):
        record_filesystem_qualification(cfg, _failing_evidence())

    assert resume_starting_attempt(remote_cfg, task.task_id) is None
    assert reserved_gpu_ids(remote_cfg.runtime_root) == set()
    assert load_task(remote_cfg, task.task_id).state == {
        "projection": "cancelled",
        "reason": "filesystem_unqualified",
    }


def test_setup_rejects_a_present_malformed_qualification_record(tmp_path: Path) -> None:
    cfg = init_shared_root(
        tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "gpu-1"
    )
    atomic_replace(
        shared_paths(cfg.shared_root)["project"] / "filesystem-qualification.json",
        {"filesystem_qualification": {"version": 1}},
    )

    with pytest.raises(RuntimeError, match="qualification record is malformed"):
        init_shared_root(
            cfg.shared_root,
            "gpu-2",
            runtime_root=tmp_path / "gpu-2",
        )
    with pytest.raises(RuntimeError, match="qualification record is malformed"):
        MachineRuntime(tmp_path / "machine-runtime").ensure_binding(
            cfg.shared_root,
            cfg.machine_name,
        )

"""Persist supported historical orphan evidence without invoking target expiry."""

from qqtools.plugins.qexp.runtime.claims import archive_claim
from qqtools.plugins.qexp.runtime.group_discovery.changes import record_task_change
from qqtools.plugins.qexp.runtime.paths import attempt_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord, utc_now
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task


def persist_legacy_orphan(cfg, task_id: str) -> AttemptRecord:
    """Build the bounded-lease orphan image produced by a supported old writer."""
    task = load_task(cfg, task_id)
    claim = dict(task.claim_control["active_claim"])
    path = attempt_path(cfg.shared_root, task_id, task.attempt_control["current_attempt_number"])
    attempt = AttemptRecord.from_dict(read_json(path))
    history = claim.pop("ownership_transition", {}).get("source_active_lease_evidence", {})
    if isinstance(history.get("claim"), dict):
        claim.update(history["claim"])
    if isinstance(history.get("attempt"), dict):
        attempt.lease = history["attempt"]
    claim["authority_mode"] = attempt.authority_mode = "bounded_lease"
    attempt.phase = "orphaned"
    attempt.result.update(exit_code=None, signal=None, category=None, reason=None)
    attempt.timestamps["orphaned_at"] = utc_now()
    with record_task_change(
        cfg,
        task,
        "claim_loss",
        details={
            "expected_attempt_id": attempt.attempt_id,
            "expected_fencing_token": attempt.current_fencing_token,
            "reserved_generation": None,
        },
    ):
        atomic_replace(path, attempt.to_dict())
        archive_claim(cfg, task_id, claim, "lease_expired")
        task.claim_control["active_claim"] = None
        task.attempt_control["current_attempt_id"] = None
        task.state.update(projection="blocked", reason="orphaned_attempt_requires_recovery")
        task.meta["revision"] += 1
        task.meta["updated_at"] = utc_now()
        save_task(cfg, task)
    return attempt

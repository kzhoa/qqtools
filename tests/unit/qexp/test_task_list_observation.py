"""Selected list evidence is bounded, command-local and distinct from launch policy."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp import observer
from qqtools.plugins.qexp import task_list_observation as observation
from qqtools.plugins.qexp.runtime.paths import submission_path
from qqtools.plugins.qexp.runtime.records import AttemptRecord, TaskRecord, TaskSpec
from qqtools.plugins.qexp.task_live_progress import build_live_progress_selection


def task(task_id="task-1", operation_id="operation-1"):
    return TaskRecord.new(task_id=task_id, machine="g1", spec=TaskSpec(["true"], "/tmp", 1), operation_id=operation_id)


def operation(tasks, enabled=None):
    enabled = enabled or [True] * len(tasks)
    entries = [
        {"task_id": t.task_id, "requested": flag, "enabled": flag, "source": "explicit", "group_policy": None}
        for t, flag in zip(tasks, enabled)
    ]
    return {
        "submission": {
            "operation_id": tasks[0].submission_operation_id,
            "resolved_context": {"task_ids": [t.task_id for t in tasks]},
            "target_group": None,
        },
        "live_progress_selection": build_live_progress_selection(entries),
    }


def save_operation(cfg, tasks, value=None):
    path = submission_path(cfg.shared_root, tasks[0].submission_operation_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(operation(tasks) if value is None else value))
    return path


@pytest.fixture
def cfg(tmp_path):
    return SimpleNamespace(shared_root=tmp_path / "shared", runtime_root=tmp_path / "runtime", machine_name="g1")


def test_shared_operation_is_read_once_and_cache_discarded_after_repair(cfg, monkeypatch):
    tasks = [task(f"task-{i}") for i in range(3)]
    path = save_operation(cfg, tasks, operation(tasks, [True, False, True]))
    reads = []
    original = observation._read_bounded_json

    def read(path, **kwargs):
        reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(observation, "_read_bounded_json", read)
    result = observation.observe_reporting_policies(cfg, tasks)
    assert [result[t.task_id]["state"] for t in tasks] == ["enabled", "disabled", "enabled"]
    assert reads == [path]
    path.write_text("broken")
    result = observation.observe_reporting_policies(cfg, tasks)
    assert {item["reason"] for item in result.values()} == {"policy_invalid"}
    assert len(reads) == 2
    save_operation(cfg, tasks)
    assert observation.observe_reporting_policies(cfg, tasks)[tasks[0].task_id]["state"] == "enabled"
    assert len(reads) == 3


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda v: v.pop("live_progress_selection"), "policy_selection_missing"),
        (lambda v: v["submission"].update(operation_id="other"), "policy_identity_mismatch"),
        (lambda v: v["live_progress_selection"].update(version=2), "policy_invalid"),
        (lambda v: v["live_progress_selection"].update(selection_digest="0" * 64), "policy_invalid"),
        (lambda v: v["submission"].update(resolved_context={}), "policy_invalid"),
    ],
)
def test_unknown_policy_reasons_are_not_disabled(cfg, mutation, reason):
    tasks = [task()]
    value = operation(tasks)
    mutation(value)
    save_operation(cfg, tasks, value)
    assert observation.observe_reporting_policies(cfg, tasks)["task-1"] == {"state": "unknown", "reason": reason}


def test_missing_reference_missing_operation_and_invalid_reference(cfg):
    tasks = [task("missing-ref", None), task("missing-op"), task("bad-ref", "../escape")]
    result = observation.observe_reporting_policies(cfg, tasks)
    assert [result[t.task_id]["reason"] for t in tasks] == [
        "policy_reference_missing",
        "policy_operation_missing",
        "policy_identity_mismatch",
    ]


def test_missing_membership_does_not_poison_valid_sibling(cfg):
    valid, unrelated = task(), task("other")
    save_operation(cfg, [valid])
    result = observation.observe_reporting_policies(cfg, [valid, unrelated])
    assert result[valid.task_id]["state"] == "enabled"
    assert result[unrelated.task_id]["reason"] == "policy_identity_mismatch"


@pytest.mark.parametrize("extra,expected", [(0, "enabled"), (1, "unknown")])
def test_operation_byte_cap_boundary(cfg, extra, expected):
    tasks = [task()]
    path = save_operation(cfg, tasks)
    data = path.read_bytes()
    path.write_bytes(data + b" " * (observation.MAX_POLICY_BYTES + extra - len(data)))
    result = observation.observe_reporting_policies(cfg, tasks)["task-1"]
    assert result["state"] == expected
    if extra:
        assert result["reason"] == "policy_oversized"


def test_failed_operation_is_attempted_once_for_siblings(cfg, monkeypatch):
    tasks = [task(f"task-{i}") for i in range(4)]
    attempts = []

    def denied(path, **kwargs):
        attempts.append(path)
        raise PermissionError("sensitive internal path")

    monkeypatch.setattr(observation, "_read_bounded_json", denied)
    result = observation.observe_reporting_policies(cfg, tasks)
    assert len(attempts) == 1
    assert {x["reason"] for x in result.values()} == {"policy_read_failed"}
    assert "sensitive" not in repr(result)


def test_basic_fields_skip_dependencies_and_all_enrichment(cfg, monkeypatch):
    tasks = [task(f"task-{i}") for i in range(3)]
    monkeypatch.setattr(observer, "iter_json", lambda _: [t.task_id for t in tasks])
    monkeypatch.setattr(observer, "read_json", lambda key: next(t.to_dict() for t in tasks if t.task_id == key))

    def forbidden(*args, **kwargs):
        raise AssertionError("unselected optional source read")

    monkeypatch.setattr(observer, "dependency_gate", forbidden)
    monkeypatch.setattr(observation, "load_task", forbidden)
    monkeypatch.setattr(observation, "observe_reporting_policies", forbidden)
    assert len(observer.list_tasks(cfg, fields=("task", "state"), limit=2)) == 2


@pytest.mark.parametrize("limit", [0, -1, 1001])
def test_enriched_legacy_limits_reject_before_task_read(cfg, monkeypatch, limit):
    monkeypatch.setattr(observer, "iter_json", lambda _: pytest.fail("read before limit validation"))
    with pytest.raises(ValueError):
        observer.list_tasks(cfg, fields=("task", "location"), limit=limit)


def test_queued_retry_does_not_read_preserved_attempt(cfg, monkeypatch):
    current = task()
    current.attempt_control.update(current_attempt_number=1, current_attempt_id=None, next_attempt_number=2)
    monkeypatch.setattr(observation, "load_task", lambda *_: current)
    monkeypatch.setattr(observation, "read_json", lambda *_: pytest.fail("queued retry read Attempt history"))
    result = observation.enrich_task_rows(cfg, [current], [observer._task_view(current)], ("task", "location"))
    assert result[0]["location"]["status"] == "absent"


def test_final_task_reread_failure_is_command_error(cfg, monkeypatch):
    current = task()

    def failed(*args):
        raise OSError("Task truth unavailable")

    monkeypatch.setattr(observation, "load_task", failed)
    with pytest.raises(OSError):
        observation.enrich_task_rows(cfg, [current], [observer._task_view(current)], ("task", "location"))


def test_changed_operation_invalidates_overall_evidence(cfg, monkeypatch):
    current = task()
    save_operation(cfg, [current], operation([current], [False]))
    later = deepcopy(current)
    later.submission_operation_id = "replacement"
    monkeypatch.setattr(observation, "load_task", lambda *_: later)
    result = observation.enrich_task_rows(cfg, [current], [observer._task_view(current)], ("task", "overall-progress"))[
        0
    ]
    assert result["progress"]["reason"] == "changed_during_read"


def running_task(task_id="task-1", operation_id="operation-1"):
    current = task(task_id, operation_id)
    attempt = AttemptRecord.claimed(
        current,
        "g1",
        [2, 0],
        "reservation-1",
        1,
        authority_mode="holder_bound",
        clock_evidence=None,
        attempt_id=f"attempt-{task_id}",
    )
    attempt.phase = "running"
    attempt.authorization["launch_id"] = "launch-1"
    attempt.process.update(wrapper_pid=10, wrapper_start_time_ticks=20)
    current.state["projection"] = "running"
    current.attempt_control.update(
        current_attempt_id=attempt.attempt_id, current_attempt_number=1, next_attempt_number=2
    )
    current.claim_control["active_claim"] = {
        "attempt_id": attempt.attempt_id,
        "attempt_number": 1,
        "fencing_token": 1,
        "machine_name": "g1",
        "authority_mode": "holder_bound",
    }
    return current, attempt


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("disabled", [False, True])
def test_combined_enrichment_read_budget_counts_all_sources(cfg, monkeypatch, shared, disabled):
    pairs = [running_task(f"task-{i}", "shared" if shared else f"op-{i}") for i in range(3)]
    tasks = [t for t, _ in pairs]
    counts = {"policy": 0, "attempt": 0, "snapshot": 0, "registration": 0, "task": 0}
    operations = {
        t.submission_operation_id: operation(
            [x for x in tasks if x.submission_operation_id == t.submission_operation_id],
            [not disabled] * sum(x.submission_operation_id == t.submission_operation_id for x in tasks),
        )
        for t in tasks
    }

    def policy(path, **kwargs):
        counts["policy"] += 1
        return operations[path.stem]

    def attempt_read(path):
        counts["attempt"] += 1
        return next(a.to_dict() for t, a in pairs if t.task_id == path.parent.name)

    def snapshot_read(path, **kwargs):
        counts["snapshot"] += 1
        raise FileNotFoundError(path)

    def registration(*args):
        counts["registration"] += 1
        return "generation-1"

    def reread(_cfg, task_id):
        counts["task"] += 1
        return next(t for t in tasks if t.task_id == task_id)

    monkeypatch.setattr(observation, "_read_bounded_json", policy)
    monkeypatch.setattr(observation, "read_json", attempt_read)
    monkeypatch.setattr(observation, "read_advisory_snapshot", snapshot_read)
    monkeypatch.setattr(observation, "_running_registration_generation", registration)
    monkeypatch.setattr(observation, "load_task", reread)
    rows = observation.enrich_task_rows(
        cfg, tasks, [observer._task_view(t) for t in tasks], ("task", "location", "activity", "overall-progress")
    )
    unique = 1 if shared else 3
    assert counts == {"policy": unique, "attempt": 3, "snapshot": 3 if disabled else 9, "registration": 6, "task": 3}
    assert sum(counts.values()) <= 7 * len(tasks) + unique
    assert all(r["location"]["assigned_gpus"] == [0, 2] for r in rows)
    assert len({r["observation_time"] for r in rows}) == 1


def test_location_only_never_reads_policy_registration_or_snapshots(cfg, monkeypatch):
    current, attempt = running_task()
    monkeypatch.setattr(observation, "read_json", lambda _: attempt.to_dict())
    monkeypatch.setattr(observation, "load_task", lambda *_: current)

    def forbidden(*args, **kwargs):
        raise AssertionError("unselected progress source")

    for name in ("observe_reporting_policies", "_running_registration_generation", "read_advisory_snapshot"):
        monkeypatch.setattr(observation, name, forbidden)
    result = observation.enrich_task_rows(cfg, [current], [observer._task_view(current)], ("task", "location"))[0]
    assert result["location"]["machine_name"] == "g1"


def test_registration_change_invalidates_only_progress(cfg, monkeypatch):
    current, attempt = running_task()
    monkeypatch.setattr(observation, "read_json", lambda _: attempt.to_dict())
    monkeypatch.setattr(observation, "load_task", lambda *_: current)
    generations = iter(["generation-1", "generation-2"])
    monkeypatch.setattr(observation, "_running_registration_generation", lambda *_: next(generations))
    monkeypatch.setattr(
        observation, "read_advisory_snapshot", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError())
    )
    result = observation.enrich_task_rows(
        cfg, [current], [observer._task_view(current)], ("task", "location", "activity")
    )[0]
    assert result["location"]["status"] == "available"
    assert result["progress"]["reason"] == "identity_mismatch"

"""Cross-home elapsed offers use Group authority and the normal transaction."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, project_maintenance, submit
from qqtools.plugins.qexp.commands import task as task_commands
from qqtools.plugins.qexp.commands.group import change_worker, create_group, group_control
from qqtools.plugins.qexp.lease import ClockCapability
from qqtools.plugins.qexp.runtime.availability import transitions
from qqtools.plugins.qexp.runtime.availability.transitions import clock_evidence
from qqtools.plugins.qexp.runtime.paths import group_path, shared_paths, submission_path
from qqtools.plugins.qexp.runtime.ready import iter_ready_marker_refs
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task
from tests.helpers.qexp.clock import set_offer_evaluation_time

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture
def due_task(tmp_path, monkeypatch):
    home = init_shared_root(tmp_path / ".qexp", "g2", runtime_root=tmp_path / "home-runtime")
    create_group(home, "exp")
    change_worker(home, "exp", "g8", "add")
    change_worker(home, "exp", "g9", "add")
    task = submit(home, ["echo", "ok"], group="exp", sharing_mode="spillover", offer_after_seconds=0)
    helper = replace(home, machine_name="g8", runtime_root=tmp_path / "helper-runtime")
    set_offer_evaluation_time(monkeypatch, home, task.task_id)
    return home, helper, task


def test_only_helper_maintenance_offers_publishes_ready_and_retires_deadline(due_task):
    home, helper, task = due_task
    index = shared_paths(home.shared_root)["offer_deadlines"] / f"{task.task_id}.json"
    active = index.resolve()

    progress = project_maintenance.advance_due_offer(helper)

    stored = load_task(home, task.task_id)
    assert (progress.outcome, progress.task_id) == ("offered", task.task_id)
    assert stored.placement_runtime["queue_scope"] == "shared"
    assert stored.placement_runtime["offered_by"] == "g8"
    assert stored.claim_control["active_claim"] is None
    assert stored.meta["revision"] == task.meta["revision"] + 1
    assert any(ref.task_id == task.task_id for ref in iter_ready_marker_refs(helper, "shared"))
    assert not index.exists() and not index.is_symlink()
    assert not active.exists()
    assert claim_task(helper, task.task_id, [0]) is not None


@pytest.mark.parametrize("evaluator", ["home", "helper"])
@pytest.mark.parametrize("drain_home", [False, True])
def test_direct_elapsed_offer_allows_active_helper_and_draining_home(due_task, evaluator, drain_home):
    home, helper, task = due_task
    if drain_home:
        change_worker(home, "exp", "g2", "drain")
    cfg = home if evaluator == "home" else helper
    result = task_commands.offer(cfg, task.task_id, reason="elapsed")
    assert result.resulting_state == "shared"
    assert load_task(home, task.task_id).placement_runtime["offered_by"] == cfg.machine_name
    revision = load_task(home, task.task_id).meta["revision"]
    replay = task_commands.offer(cfg, task.task_id, reason="elapsed")
    assert replay.idempotent
    assert load_task(home, task.task_id).meta["revision"] == revision


@pytest.mark.parametrize("state", ["draining", "removing", "removed", "nonmember", "other-group"])
def test_direct_elapsed_offer_rejects_helper_without_active_task_group_membership(due_task, state):
    home, helper, task = due_task
    group_file = group_path(home.shared_root, "exp")
    group = read_json(group_file)
    if state in {"removed", "nonmember", "other-group"}:
        del group["group"]["worker_set"]["g8"]
    else:
        group["group"]["worker_set"]["g8"]["state"] = state
    atomic_replace(group_file, group)
    if state == "other-group":
        create_group(helper, "unrelated")
    with pytest.raises(ValueError, match="active Group worker"):
        task_commands.offer(helper, task.task_id, reason="elapsed")
    assert load_task(home, task.task_id).placement_runtime["queue_scope"] == "home"
    assert load_task(home, task.task_id).meta["revision"] == task.meta["revision"]


def test_membership_change_between_discovery_and_locked_transition_rejects_offer(due_task, monkeypatch):
    home, helper, task = due_task
    real_offer = project_maintenance.offer

    def drain_then_offer(*args, **kwargs):
        change_worker(home, "exp", "g8", "drain")
        return real_offer(*args, **kwargs)

    monkeypatch.setattr(project_maintenance, "offer", drain_then_offer)
    progress = project_maintenance.advance_due_offer(helper)
    assert progress.reason == "offer_rejected"
    assert load_task(home, task.task_id).placement_runtime["queue_scope"] == "home"


@pytest.mark.parametrize("control", ["claim", "cancel", "cleanup", "unshare", "future", "terminal"])
def test_control_change_before_locked_helper_offer_cannot_mutate_stale_task(due_task, monkeypatch, control):
    home, helper, task = due_task
    real_offer = project_maintenance.offer
    observed = []

    def change_then_offer(*args, **kwargs):
        if control == "claim":
            assert claim_task(home, task.task_id, [0]) is not None
        elif control == "cancel":
            task_commands.cancel(home, task.task_id)
        elif control == "unshare":
            task_commands.keep_local(home, task.task_id)
        elif control == "future":
            task_commands.share(home, task.task_id, after_seconds=3600)
        else:
            stored = load_task(home, task.task_id)
            if control == "cleanup":
                stored.control["cleanup_state"] = "prepared"
            else:
                stored.state["projection"] = "failed"
            save_task(home, stored)
        observed.append(load_task(home, task.task_id).to_dict())
        return real_offer(*args, **kwargs)

    monkeypatch.setattr(project_maintenance, "offer", change_then_offer)
    result = project_maintenance.advance_due_offer(helper)
    assert result.outcome == "noop"
    assert load_task(home, task.task_id).to_dict() == observed[0]


def test_elapsed_evaluator_authorization_runs_under_group_and_task_locks(due_task, monkeypatch):
    home, helper, task = due_task
    held = set()
    for name in ("group_lock", "task_lock"):
        real = getattr(transitions, name)

        @contextmanager
        def track(*args, _real=real, _name=name, **kwargs):
            with _real(*args, **kwargs) as result:
                held.add(_name)
                try:
                    yield result
                finally:
                    held.remove(_name)

        monkeypatch.setattr(transitions, name, track)
    real_group = transitions._group_data

    def read_locked_group(*args):
        assert held == {"group_lock", "task_lock"}
        return real_group(*args)

    monkeypatch.setattr(transitions, "_group_data", read_locked_group)
    assert task_commands.offer(helper, task.task_id, reason="elapsed").resulting_state == "shared"


def test_offering_helper_does_not_broaden_fallback_claim_permissions(due_task, monkeypatch):
    home, helper, task = due_task
    task_commands.share(home, task.task_id, after_seconds=0, helper_machines=["g9"])
    set_offer_evaluation_time(monkeypatch, home, task.task_id)
    assert task_commands.offer(helper, task.task_id, reason="elapsed").resulting_state == "shared"
    stored = load_task(home, task.task_id)
    assert stored.placement_policy["fallback_constraint"] == ["g9"]
    assert claim_task(helper, task.task_id, [0]) is None
    allowed = replace(helper, machine_name="g9", runtime_root=helper.runtime_root.parent / "allowed")
    assert claim_task(allowed, task.task_id, [0]) is not None


@pytest.mark.parametrize("evidence", ["future", "missing", "unhealthy"])
def test_helper_cannot_offer_without_elapsed_clock_proof(due_task, monkeypatch, evidence):
    home, helper, task = due_task
    if evidence == "future":
        set_offer_evaluation_time(monkeypatch, home, task.task_id, seconds_after_deadline=-1)
    elif evidence == "missing":
        stored = load_task(home, task.task_id)
        stored.placement_runtime["offer_clock_evidence"] = None
        save_task(home, stored)
    else:
        monkeypatch.setattr(transitions, "clock_evidence", clock_evidence)
        monkeypatch.setattr(transitions, "clock_capability", lambda _cfg: ClockCapability("unhealthy", "test"))
    result = task_commands.offer(helper, task.task_id, reason="elapsed")
    assert result.resulting_state == "home"
    assert load_task(home, task.task_id).meta["revision"] == task.meta["revision"]


def test_local_only_home_is_supported_but_helper_is_rejected(tmp_path, monkeypatch):
    home = init_shared_root(tmp_path / ".qexp", "g2", runtime_root=tmp_path / "home")
    task = submit(home, ["echo", "local"])
    helper = replace(home, machine_name="g8", runtime_root=tmp_path / "helper")
    with pytest.raises(ValueError, match="Group"):
        task_commands.offer(helper, task.task_id, reason="elapsed")
    assert task_commands.offer(home, task.task_id, reason="elapsed").resulting_state == "home"
    assert claim_task(helper, task.task_id, [0]) is None
    assert claim_task(home, task.task_id, [0]) is not None


def test_group_cancellation_barrier_rejects_helper_offer(due_task):
    home, helper, task = due_task
    group_control(home, "exp", "cancel")
    with pytest.raises(ValueError):
        task_commands.offer(helper, task.task_id, reason="elapsed")
    assert load_task(home, task.task_id).placement_runtime["queue_scope"] == "home"


@pytest.mark.parametrize("evaluator", ["home", "helper"])
def test_invalid_home_membership_remains_a_blocker(due_task, evaluator):
    home, helper, task = due_task
    group_file = group_path(home.shared_root, "exp")
    group = read_json(group_file)
    group["group"]["worker_set"]["g2"]["state"] = "removing"
    atomic_replace(group_file, group)
    with pytest.raises(ValueError, match="active or draining"):
        task_commands.offer(home if evaluator == "home" else helper, task.task_id, reason="elapsed")
    assert load_task(home, task.task_id).placement_runtime["queue_scope"] == "home"


@pytest.mark.parametrize("blocker", ["uncommitted", "private"])
def test_helper_offer_preserves_submission_and_spillover_prerequisites(due_task, blocker):
    home, helper, task = due_task
    if blocker == "uncommitted":
        path = submission_path(home.shared_root, task.submission_operation_id)
        operation = read_json(path)
        operation["submission"]["state"] = "committing"
        atomic_replace(path, operation)
    else:
        stored = load_task(home, task.task_id)
        stored.placement_policy["sharing_mode"] = "private"
        # Retain stale proof to verify elapsed evaluation never broadens policy.
        save_task(home, stored)
    before = load_task(home, task.task_id).to_dict()
    with pytest.raises(ValueError):
        task_commands.offer(helper, task.task_id, reason="elapsed")
    assert load_task(home, task.task_id).to_dict() == before


@pytest.mark.parametrize("membership", ["nonmember", "draining"])
def test_repeated_unauthorized_maintenance_creates_no_operations_or_outbox_work(due_task, membership):
    home, helper, task = due_task
    group_file = group_path(home.shared_root, "exp")
    group = read_json(group_file)
    if membership == "nonmember":
        del group["group"]["worker_set"]["g8"]
    else:
        group["group"]["worker_set"]["g8"]["state"] = "draining"
    atomic_replace(group_file, group)
    operations = home.shared_root / "operations"

    def snapshot():
        return {path.relative_to(operations): path.read_bytes() for path in operations.rglob("*.json")}

    before = snapshot()
    for _ in range(4):
        assert project_maintenance.advance_due_offer(helper).reason == "offer_rejected"
    assert snapshot() == before
    assert load_task(home, task.task_id).placement_runtime["queue_scope"] == "home"
    assert (shared_paths(home.shared_root)["offer_deadlines"] / f"{task.task_id}.json").exists()


def test_completed_elapsed_operation_replays_after_evaluator_drains(due_task):
    home, helper, task = due_task
    request = transitions.AvailabilityTransitionRequest(
        action="elapsed_offer", task_id=task.task_id, operation_id="helper-elapsed-replay", reason="elapsed"
    )
    first = transitions.apply_availability_transition(helper, request)
    change_worker(home, "exp", "g8", "drain")
    replay = transitions.apply_availability_transition(helper, request)
    assert first.resulting_state == replay.resulting_state == "shared"
    assert replay.idempotent
    assert load_task(home, task.task_id).meta["revision"] == first.task.meta["revision"]

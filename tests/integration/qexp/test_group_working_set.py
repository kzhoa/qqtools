"""Published history, daily visibility and bounded due-work contracts."""

import shlex
import time

import pytest

from qqtools.plugins.qexp import init_shared_root, observer, submit
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.group import create_group, group_control, show_group, wake_group
from qqtools.plugins.qexp.runtime import group_working_set as index
from qqtools.plugins.qexp.runtime.group_working_set_expiry import expire_due_groups
from tests.integration.qexp.test_output_format import _base_args

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]

WEEK = 7 * 24 * 3600


def names(cfg, **options):
    items = []
    cursor = None
    while True:
        page = observer.list_groups_page(cfg, page_size=2, cursor=cursor, **options)
        items.extend(item["group"]["name"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            assert page["stop_reason"] == "exhausted"
            return items


@pytest.fixture
def cfg(tmp_path):
    return init_shared_root(tmp_path / "experiment with spaces" / ".qexp", "g1", runtime_root=tmp_path / "local")


def test_expiry_wake_and_cross_host_append(cfg, tmp_path):
    other = init_shared_root(cfg.shared_root, "g2", runtime_root=tmp_path / "other")
    create_group(cfg, "experiment", workers=["g1", "g2"])
    group_control(cfg, "experiment", "pause")
    later = time.time() + WEEK + 1
    before = index.due_candidates(cfg.shared_root, now=later, limit=1)
    assert names(cfg) == ["experiment"]
    assert show_group(other, "experiment")["group"]["dispatch_state"] == "paused"
    assert index.due_candidates(cfg.shared_root, now=later, limit=1) == before
    expire_due_groups(other, now=later, limit=1)
    assert names(cfg) == []
    assert names(cfg, all_groups=True) == ["experiment"]
    wake_group(other, "experiment")
    assert names(cfg) == ["experiment"]
    assert show_group(cfg, "experiment")["group"]["next_membership_sequence"] == 1
    expire_due_groups(cfg, now=later + 1, limit=1)
    assert names(other) == []
    task = submit(other, ["true"], group="experiment", idempotency_key="cross-host")
    assert task.state["projection"] == "queued"
    assert names(cfg) == names(cfg, all_groups=True) == ["experiment"]
    after = index.due_candidates(cfg.shared_root, now=later + 10, limit=1)
    assert after and after != before
    replay = submit(other, ["true"], group="experiment", idempotency_key="cross-host")
    assert replay.task_id == task.task_id
    assert index.due_candidates(cfg.shared_root, now=later + 10, limit=1) == after


def test_retained_due_groups_do_not_starve_settled_group(cfg):
    for name in ("a", "b", "c", "d"):
        submit(cfg, ["true"], group=name)
    create_group(cfg, "z")
    later = time.time() + WEEK + 1
    for _ in range(5):
        expire_due_groups(cfg, now=later, limit=1)
    assert names(cfg) == ["a", "b", "c", "d"]
    assert names(cfg, all_groups=True) == ["a", "b", "c", "d", "z"]
    assert index.due_candidates(cfg.shared_root, now=later, limit=5) == []
    assert expire_due_groups(cfg, now=later, limit=1) == 0


@pytest.mark.parametrize("retire_first", [False, True])
def test_stale_expiry_and_refresh_in_both_orders(cfg, retire_first):
    root = cfg.shared_root
    index.refresh_group(root, "g", "initial", now=1)
    name, revision, check = index.due_candidates(root, now=WEEK + 1, limit=1)[0]
    if retire_first:
        assert index.finish_check(root, name, revision, check, now=WEEK + 1, retire=True)
    assert index.refresh_group(root, name, "new-activity", now=WEEK + 2)
    assert not index.finish_check(root, name, revision, check, now=WEEK + 3, retire=True)
    for all_groups in (False, True):
        assert tuple(index.group_candidates(root, None, all_groups=all_groups, limit=2).keys) == ("g",)


def test_empty_filtered_slice_prints_executable_continuation(cfg, capsys):
    import json

    for i in range(51):
        create_group(cfg, f"g-{i:02}")
    group_control(cfg, "g-50", "pause")
    args = _base_args(cfg)
    request = ["group", "list", "--all", "--attention", "--page-size", "1"]
    assert main([*args, *request, "--format=json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["items"] == []
    assert first["stop_reason"] == "budget_exhausted"
    assert first["next_cursor"]
    with pytest.raises(ValueError, match="cursor"):
        observer.list_groups_page(cfg, attention=True, cursor=first["next_cursor"])
    assert main([*args, *request]) == 0
    human = capsys.readouterr().out
    assert "match" in human.lower()
    command = shlex.split("qexp " + human.split("qexp ", 1)[1].strip())
    assert command[1:3] == ["--project", str(cfg.shared_root)]
    assert main([*args, *command[3:], "--format=json"]) == 0
    last = json.loads(capsys.readouterr().out)
    assert [item["group"]["name"] for item in last["items"]] == ["g-50"]
    assert last["next_cursor"] is None and last["stop_reason"] == "exhausted"


@pytest.mark.parametrize("view", [[], ["--all"]])
def test_interrupted_index_returns_operational_cli_error(cfg, capsys, view):
    import json

    (cfg.shared_root / "indexes/group-working-set/redo.json").write_text("{}")
    assert main([*_base_args(cfg), "group", "list", *view, "--format=json"]) != 0
    captured = capsys.readouterr()
    assert json.loads(captured.out or captured.err)["error"]["code"] == "index_unavailable"


def test_outbox_advances_while_index_is_locked(cfg):
    import fcntl
    from concurrent.futures import ThreadPoolExecutor

    from qqtools.plugins.qexp.runtime.maintenance_service import advance_maintenance_work

    with ThreadPoolExecutor(1) as pool:
        with (cfg.shared_root / "indexes/group-working-set/index.lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            progress = pool.submit(advance_maintenance_work, cfg, reservation_runtime_root=None).result(timeout=3)
    assert progress["maintenance_state"] in {"waiting", "completed", "running", "pending"}


def test_wake_unknown_name_preserves_both_views(cfg):
    with pytest.raises(FileNotFoundError):
        wake_group(cfg, "missing")
    assert names(cfg) == names(cfg, all_groups=True) == []


def test_unhealthy_clock_does_not_defer_work_on_other_hosts(cfg, monkeypatch):
    from qqtools.plugins.qexp.runtime import group_working_set_expiry as expiry

    create_group(cfg, "g")
    first = index.next_check_time(cfg.shared_root)
    with monkeypatch.context() as unhealthy:
        unhealthy.setattr(expiry.time, "time", lambda: first + 365 * 86400)
        unhealthy.setattr(expiry, "_qualified_time", lambda _cfg: None)
        assert expire_due_groups(cfg) == 0
    assert index.next_check_time(cfg.shared_root) == first
    assert expire_due_groups(cfg, now=first + 1) == 1
    assert names(cfg) == []


def _drain_group_locators(cfg, name):
    from qqtools.plugins.qexp.runtime.group_discovery import locator
    from qqtools.plugins.qexp.runtime.group_discovery.control import GroupControlCursor, advance_group_control_rechecks
    from qqtools.plugins.qexp.runtime.group_discovery.maintenance import GroupMaintenance
    from qqtools.plugins.qexp.runtime.group_discovery.service import GroupDiscoveryService

    root = cfg.shared_root
    control = GroupControlCursor()
    maintenance = GroupMaintenance(root, name)
    service = None
    try:
        for _ in range(3000):
            work = {lane: locator.read_group_locator(root, name, lane) for lane in locator.LANES}
            if not any(work.values()):
                return
            if work["membership"]:
                generation = work["membership"]["generation"]
                if service is None:
                    service = GroupDiscoveryService(root, name, mode="locator", locator_generation=generation)
                service.update_locator_generation(generation)
                result = service.advance()
                assert result["state"] not in {"error", "ambiguous"}, result
                if result["state"] == "complete":
                    service.acknowledge_if_quiescent()
            if work["control"]:
                advance_group_control_rechecks(cfg, name, work["control"]["generation"], control)
            if work["maintenance"]:
                generation = work["maintenance"]["generation"]
                maintenance.advance(locator_generation=generation)
                maintenance.acknowledge_if_quiescent(generation)
        pytest.fail(f"Locators did not settle: {work}")
    finally:
        for owner in (service, maintenance):
            if owner is not None:
                owner.request_close()
                for _ in range(1000):
                    if owner.is_closed:
                        break
                    owner.advance()


def test_terminal_members_settle_across_slices_and_retry_restores_group(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.commands.task import retry
    from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
    from qqtools.plugins.qexp.scheduler import cancel_task, claim_task, fail_attempt
    from tests.helpers.qexp_discovery import isolated_group
    from tests.integration.qexp.test_group_service_quiescence import activate

    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    name = "experiment"
    from qqtools.plugins.qexp.runtime.group_discovery import maintenance
    from qqtools.plugins.qexp.runtime.submission import submit_specs

    monkeypatch.setattr(maintenance, "_LANE_IDLE_SECONDS", 0)
    specs = [{"task_id": f"t{i:02}", "command": ["true"]} for i in range(17)]
    submitted = submit_specs(cfg, specs, group_name=name, kind="bulk", idempotency_key="terminal-batch")
    failed = submitted[0]
    attempt = claim_task(cfg, failed.task_id, [0])
    assert fail_attempt(cfg, failed.task_id, attempt.attempt_id, attempt.current_fencing_token, "qualification")
    from qqtools.plugins.qexp.lifecycle import TerminalTransition, commit_terminal_transition_locked
    from qqtools.plugins.qexp.runtime.paths import attempt_path
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from qqtools.plugins.qexp.scheduler import authority_locks

    success = submitted[1]
    attempt = claim_task(cfg, success.task_id, [0])
    with authority_locks(cfg, success):
        transition = TerminalTransition(
            success.task_id,
            attempt.attempt_id,
            attempt.attempt_number,
            attempt.current_fencing_token,
            "succeeded",
            "process_exited",
            0,
            frozenset({"running"}),
            frozenset({"claimed"}),
            "active",
        )
        assert (
            commit_terminal_transition_locked(cfg, load_task(cfg, success.task_id), transition).outcome == "committed"
        )
    missed = submitted[2]
    attempt = claim_task(cfg, missed.task_id, [1])
    attempt_path(cfg.shared_root, missed.task_id, attempt.attempt_number).unlink()
    cancel_task(cfg, missed.task_id)
    for task in list(submitted)[3:]:
        cancel_task(cfg, task.task_id)
    _drain_group_locators(cfg, name)
    replay = submit_specs(cfg, specs, group_name=name, kind="bulk", idempotency_key="terminal-batch")
    assert replay[0].task_id == failed.task_id
    assert GroupCoverage(cfg.shared_root, name).status().is_complete
    later = time.time() + WEEK + 2
    assert expire_due_groups(cfg, now=later) == 1
    assert names(cfg) == [name]
    assert expire_due_groups(cfg, now=later + 61) == 1
    assert names(cfg) == []
    retry(cfg, failed.task_id)
    assert names(cfg) == names(cfg, all_groups=True) == [name]


@pytest.mark.parametrize("append_first", [True, False])
@pytest.mark.parametrize("terminate", [True, False])
def test_cross_machine_cancel_watermark_orders_contending_appends(cfg, tmp_path, monkeypatch, append_first, terminate):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from qqtools.plugins.qexp import scheduler
    from qqtools.plugins.qexp.commands import group as controls
    from qqtools.plugins.qexp.runtime import group_activity
    from qqtools.plugins.qexp.runtime.store import atomic_replace
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from tests.integration.qexp.test_scheduling_recovery_regressions import ProcessEvidence

    other = init_shared_root(cfg.shared_root, "g2", runtime_root=tmp_path / "second")
    create_group(cfg, "race", workers=["g1", "g2"])
    running = submit(cfg, ["true"], group="race")
    attempt = scheduler.claim_task(cfg, running.task_id, [0])
    assert scheduler.authorize_launch(cfg, running.task_id, attempt.attempt_id, attempt.current_fencing_token)
    atomic_replace(
        cfg.runtime_root / "processes" / f"{attempt.attempt_id}.json",
        {
            "process": {
                "task_id": running.task_id,
                "attempt_id": attempt.attempt_id,
                "fencing_token": attempt.current_fencing_token,
                "process_group_id": 9876,
                "observed_state": "running",
                "supervisor": "agent",
            }
        },
    )
    held = Event()
    release = Event()
    competing = Event()

    owner, method = (group_activity, "_prepare_locked") if append_first else (controls, "write_active_operation")
    original = getattr(owner, method)

    def pause_once(*args, **kwargs):
        if not held.is_set():
            held.set()
            assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, pause_once)

    def append():
        return submit(other, ["true"], group="race", task_id="contender")

    def cancel():
        return group_control(cfg, "race", "cancel", terminate_running=terminate)["cancellation_operation"]

    def compete(call):
        competing.set()
        return call()

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(append if append_first else cancel)
        assert held.wait(10)
        second = pool.submit(compete, cancel if append_first else append)
        assert competing.wait(5)
        release.set()
        operation = second.result() if append_first else first.result()
        member = first.result() if append_first else second.result()
    assert (member.group_membership_sequence <= operation["membership_high_watermark"]) == append_first
    if append_first:
        assert load_task(other, member.task_id).state["projection"] == "cancelled"
    else:
        assert scheduler.claim_task(other, member.task_id, [0])
    if terminate:
        assert running.task_id in operation["pending_machine_acknowledgements"]["g1"]
        monkeypatch.setattr(scheduler, "inspect_group_identity", lambda *_: ProcessEvidence(state="absent"))
        scheduler.reconcile_running_tasks(cfg)
        completed = show_group(cfg, "race")["cancellation_operation"]
        assert completed["state"] == "completed" and not completed["pending_machine_acknowledgements"]
    else:
        assert load_task(cfg, running.task_id).state["projection"] == "running"
    assert submit(other, ["true"], group="race").group_membership_sequence > operation["membership_high_watermark"]


@pytest.mark.parametrize("activity", ["append", "wake"])
@pytest.mark.parametrize("activity_first", [True, False])
def test_concurrent_expiry_and_activity_preserve_both_views(cfg, tmp_path, monkeypatch, activity_first, activity):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from qqtools.plugins.qexp.runtime import group_working_set_expiry as expiry

    other = init_shared_root(cfg.shared_root, "g2", runtime_root=tmp_path / "other")
    create_group(cfg, "exp", workers=["g1", "g2"])
    now = time.time() + WEEK + 1
    got_candidate, resume = Event(), Event()
    original = expiry._finish_candidate if activity_first else index.finish_check

    def hold(*args, **kwargs):
        got_candidate.set()
        assert resume.wait(10)
        return original(*args, **kwargs)

    if activity_first:
        monkeypatch.setattr(expiry, "_finish_candidate", hold)
    else:
        monkeypatch.setattr(index, "finish_check", hold)

    def refresh():
        return submit(other, ["true"], group="exp") if activity == "append" else wake_group(other, "exp")

    with ThreadPoolExecutor(2) as pool:
        retire = pool.submit(expire_due_groups, cfg, now=now)
        assert got_candidate.wait(10)
        update = pool.submit(refresh)
        if activity_first:
            update.result(timeout=5)
        resume.set()
        assert retire.result(timeout=5) == 1
        update.result(timeout=5)
    assert names(other) == names(cfg, all_groups=True) == ["exp"]

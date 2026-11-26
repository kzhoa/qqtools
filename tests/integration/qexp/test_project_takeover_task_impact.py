"""Observe takeover consequences from published qexp Task and Group records."""

import json
import shlex
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_takeover import (
    TakeoverUsageError,
    confirm_project_takeover,
    preview_project_takeover,
)
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.commands.task import cancel, offer, submit
from qqtools.plugins.qexp.runtime.paths import attempt_path, group_path, submission_path, task_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = pytest.mark.integration


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "shared project"
    runtime = MachineRuntime(tmp_path / "new machine")
    previous = MachineRuntime(tmp_path / "old machine")
    initialize_machine(runtime, "g3")
    initialize_machine(previous, "g2")
    initialize_project(root)
    for owner in (runtime, previous):
        assert register_projects(owner, [root])["projects"][0]["status"] == "registered"
    return root, runtime, previous


def config(runtime, root):
    return runtime.resolve_project_binding(root / ".qexp").root_config()


def expire_registration(cfg):
    path = cfg.shared_root / "machines" / cfg.machine_name / "registration.json"
    registration = read_json(path)
    registration["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, registration)


def command(project):
    root, runtime, _ = project
    return [
        "--machine-runtime-root",
        str(runtime.root),
        "project",
        "takeover",
        str(root),
        "--machine=g2",
        "--plan",
    ]


def snapshot(root):
    return {
        path: (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for path in (root, *root.rglob("*"))
    }


def test_task_preview_preserves_queued_interrupted_and_cancelled_work(project, capsys):
    root, runtime, previous = project
    cfg = config(previous, root)
    current = config(runtime, root)
    create_group(cfg, "training", ["g2", "g3"])
    submit(cfg, ["echo", "queued"], task_id="queued", group="training", working_dir=root)
    submit(cfg, ["echo", "interrupted"], task_id="interrupted", working_dir=root)
    attempt = claim_task(cfg, "interrupted", [0], reservation_runtime_root=previous.root)
    assert attempt is not None
    path = attempt_path(cfg.shared_root, "interrupted", attempt.attempt_number)
    document = read_json(path)
    document["attempt"]["termination"]["requested_at"] = "2026-01-01T00:00:00Z"
    atomic_replace(path, document)
    submit(cfg, ["echo", "cancelled"], task_id="cancelled", working_dir=root)
    assert cancel(cfg, "cancelled", reservation_runtime_root=previous.root).state["projection"] == "cancelled"
    submit(current, ["echo", "unrelated"], task_id="unrelated", working_dir=root)
    uncommitted = submit(cfg, ["echo", "unpublished"], task_id="unpublished", working_dir=root)
    operation_path = submission_path(cfg.shared_root, uncommitted.submission_operation_id)
    operation = read_json(operation_path)
    operation["submission"]["state"] = "preparing"
    atomic_replace(operation_path, operation)
    worker_path = group_path(cfg.shared_root, "training")
    group = read_json(worker_path)
    group["group"]["worker_set"].pop("g2")
    atomic_replace(worker_path, group)
    expire_registration(cfg)
    before = snapshot(root)

    assert main([*command(project), "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)
    impact = result["task_impact"]
    assert impact["status"] == "observed" and impact["count"] == 3
    assert impact["next_task"] is None
    tasks = {task["task_id"]: task for task in impact["tasks"]}
    assert set(tasks) == {"queued", "interrupted", "cancelled"}
    assert tasks["queued"]["disposition"] == "queued"
    assert tasks["queued"]["placement"] == {
        "state": "blocked",
        "reason": "worker_missing",
        "group_change": {
            "action": "add_primary_worker",
            "group_name": "training",
            "machine_name": "g2",
            "group_revision": group["meta"]["revision"],
            "worker_set_epoch": group["group"]["worker_set_epoch"],
        },
    }
    assert tasks["interrupted"]["disposition"] == "review_interrupted_attempt"
    evidence = tasks["interrupted"]["attempt"]
    assert evidence["attempt_id"] == attempt.attempt_id
    assert evidence["reservation_id"] == attempt.reservation_id
    assert evidence["process_state"] == "unknown"
    assert evidence["stop_pending"] is True
    assert tasks["cancelled"]["phase"] == "cancelled"
    assert tasks["cancelled"]["disposition"] == "recorded_result"
    assert snapshot(root) == before


def test_task_preview_paginates_and_supplies_shell_safe_continuation(project, capsys):
    root, _, previous = project
    cfg = config(previous, root)
    for index in range(27):
        submit(cfg, ["echo", "queued"], task_id=f"task-{index:02}", working_dir=root)
    expire_registration(cfg)
    before = snapshot(root)

    assert main(command(project)) == 0

    output = capsys.readouterr().out
    continuation = next(
        line.removeprefix("More Tasks: ") for line in output.splitlines() if line.startswith("More Tasks: ")
    )
    args = shlex.split(continuation)
    assert args[0] == "qexp"
    assert Path(args[args.index("takeover") + 1]) == root / ".qexp"
    assert "--tasks-after=task-24" in args

    assert main([*args[1:], "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)["task_impact"]
    assert result["count"] == 27
    assert [task["task_id"] for task in result["tasks"]] == ["task-25", "task-26"]
    assert result["next_task"] is None
    assert snapshot(root) == before


def test_unreadable_exact_attempt_does_not_authorize_retry(project, capsys):
    root, _, previous = project
    cfg = config(previous, root)
    task = submit(cfg, ["echo", "interrupted"], task_id="interrupted", working_dir=root)
    attempt = claim_task(cfg, task.task_id, [0], reservation_runtime_root=previous.root)
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    path.unlink()
    expire_registration(cfg)

    assert main([*command(project), "--format=json"]) == 0

    row = json.loads(capsys.readouterr().out)["task_impact"]["tasks"][0]
    assert row["disposition"] == "attempt_evidence_unavailable"
    assert row["attempt"] is None


def test_task_cancellation_is_visible_before_attempt_stop_publication(project, capsys):
    root, _, previous = project
    cfg = config(previous, root)
    pending = submit(cfg, ["echo", "interrupted"], task_id="interrupted", working_dir=root)
    attempt = claim_task(cfg, pending.task_id, [0], reservation_runtime_root=previous.root)
    assert attempt is not None
    path = task_path(cfg.shared_root, pending.task_id)
    document = read_json(path)
    document["task"]["control"].update(
        cancellation_requested_at="2026-01-01T00:00:00Z",
        terminate_running=True,
    )
    atomic_replace(path, document)
    expire_registration(cfg)

    assert main([*command(project), "--format=json"]) == 0

    row = json.loads(capsys.readouterr().out)["task_impact"]["tasks"][0]
    assert row["disposition"] == "review_interrupted_attempt"
    assert row["attempt"]["stop_pending"] is False
    assert row["cancellation"] == {
        "requested_at": "2026-01-01T00:00:00Z",
        "terminate_running": True,
        "acknowledged_at": None,
    }


def test_preview_distinguishes_the_remote_claim_from_a_running_process(project, capsys):
    root, successor, previous = project
    cfg = config(previous, root)
    other = config(successor, root)
    create_group(cfg, "training", ["g2", "g3"])
    task = submit(
        cfg, ["echo", "shared"], task_id="shared", group="training", sharing_mode="spillover", working_dir=root
    )
    offer(cfg, task.task_id)
    attempt = claim_task(other, task.task_id, [0], reservation_runtime_root=successor.root)
    assert attempt is not None and attempt.phase == "claimed"
    expire_registration(cfg)

    assert main([*command(project), "--format=json"]) == 0

    row = json.loads(capsys.readouterr().out)["task_impact"]["tasks"][0]
    assert row["disposition"] == "claimed_elsewhere"
    assert row["claim_machine"] == "g3"
    assert row["attempt"]["attempt_id"] == attempt.attempt_id
    assert row["attempt"]["phase"] == "claimed"
    assert row["attempt"]["process_state"] == "unknown"


def test_task_preview_fails_closed_on_malformed_control(project, capsys):
    root, _, previous = project
    cfg = config(previous, root)
    task = submit(cfg, ["echo", "queued"], task_id="queued", working_dir=root)
    path = task_path(cfg.shared_root, task.task_id)
    value = read_json(path)
    value["task"]["control"] = None
    atomic_replace(path, value)
    expire_registration(cfg)

    assert main([*command(project), "--format=json"]) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["task_impact"]["status"] == "unavailable"
    assert "target_task_impact_unavailable" in {item["code"] for item in result["blockers"]}


def test_confirmed_group_selection_covers_tasks_beyond_current_page(project, capsys):
    root, runtime, previous = project
    cfg = config(previous, root)
    for group, task in (("blue", "zz-blue"), ("green", "zz-green")):
        create_group(cfg, group, ["g2", "g3"])
        submit(cfg, ["echo", task], task_id=task, group=group, working_dir=root)
        path = group_path(cfg.shared_root, group)
        document = read_json(path)
        document["group"]["worker_set"].pop("g2")
        atomic_replace(path, document)
    for index in range(25):
        submit(cfg, ["echo", "queued"], task_id=f"task-{index:02}", working_dir=root)
    expire_registration(cfg)
    before = snapshot(root)

    assert main([*command(project), "--add-primary-worker=green", "--add-primary-worker=blue"]) == 0
    human = capsys.readouterr().out
    assert "Plan ID:" in human
    assert "Selected primary worker additions" in human
    continuation = next(line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("More Tasks: "))
    assert "--add-primary-worker=green" in shlex.split(continuation)
    assert "--add-primary-worker=blue" in shlex.split(continuation)
    assert main([*shlex.split(continuation)[1:], "--format=json"]) == 0
    continued = json.loads(capsys.readouterr().out)
    assert [task["task_id"] for task in continued["task_impact"]["tasks"]] == ["zz-blue", "zz-green"]
    assert [change["group_name"] for change in continued["selected_group_changes"]] == ["blue", "green"]

    preview = preview_project_takeover(runtime, root, "g2", selected_groups=("green", "blue"))
    plan_id = preview["plan_id"]
    assert preview["task_impact"]["next_task"] == "task-24"
    assert [item["group_name"] for item in preview["selected_group_changes"]] == ["blue", "green"]
    assert [item["affected_task_count"] for item in preview["selected_group_changes"]] == [1, 1]
    confirmed = confirm_project_takeover(runtime, root, "g2", plan_id, selected_groups=("blue", "green"))
    assert [item.group_name for item in confirmed.groups] == ["blue", "green"]
    with pytest.raises(TakeoverUsageError, match="--plan"):
        confirm_project_takeover(runtime, root, "g2", plan_id, selected_groups=("blue",))
    assert snapshot(root) == before

    group_pathname = group_path(cfg.shared_root, "green")
    updated = read_json(group_pathname)
    updated["meta"]["revision"] += 1
    atomic_replace(group_pathname, updated)
    with pytest.raises(TakeoverUsageError, match="--plan"):
        confirm_project_takeover(runtime, root, "g2", plan_id, selected_groups=("green", "blue"))

    off_page = task_path(cfg.shared_root, "zz-green")
    malformed = read_json(off_page)
    malformed["task"]["control"] = None
    atomic_replace(off_page, malformed)
    with pytest.raises(RuntimeError, match="zz-green.*malformed control"):
        preview_project_takeover(runtime, root, "g2", selected_groups=("green",))


def test_group_confirmation_includes_offered_tasks_from_another_home(project):
    root, runtime, previous = project
    target = config(previous, root)
    other = config(runtime, root)
    create_group(target, "training", ["g2", "g3"])
    submit(target, ["true"], task_id="private", group="training", working_dir=root)
    submit(other, ["true"], task_id="away", group="training", sharing_mode="spillover", working_dir=root)
    path = group_path(target.shared_root, "training")
    document = read_json(path)
    document["group"]["worker_set"].pop("g2")
    atomic_replace(path, document)
    expire_registration(target)

    def preview():
        return preview_project_takeover(runtime, root, "g2", selected_groups=("training",))

    initial = preview()
    assert initial["selected_group_changes"][0]["affected_task_count"] == 1
    offer(other, "away")
    current = preview()
    assert current["selected_group_changes"][0]["affected_task_count"] == 2
    assert [task["task_id"] for task in current["task_impact"]["tasks"]] == ["away", "private"]
    with pytest.raises(TakeoverUsageError, match="fresh --plan"):
        confirm_project_takeover(runtime, root, "g2", initial["plan_id"], selected_groups=("training",))

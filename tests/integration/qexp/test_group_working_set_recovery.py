"""Recover the index update across each durable publication boundary."""

import pytest

from qqtools.plugins.qexp.runtime import group_working_set as ws
from qqtools.plugins.qexp.runtime.observation import tree

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


class PowerLoss(BaseException):
    pass


@pytest.mark.parametrize("operation", ["create", "wake", "refresh", "retire", "defer"])
def test_redo_replays_every_durable_record_boundary(tmp_path, monkeypatch, operation):
    def setup(root):
        root.mkdir()
        ws.initialize_index(root)
        if operation != "create":
            ws.refresh_group(root, "name-1", "created", now=1)
        if operation == "wake":
            name, rev, due = ws.due_candidates(root, now=1_000_000, limit=1)[0]
            assert ws.finish_check(root, name, rev, due, now=1_000_000, retire=True)

    def update(root):
        if operation in {"refresh", "create", "wake"}:
            return ws.refresh_group(root, "name-1", "append", now=2)
        name, revision, check = ws.due_candidates(root, now=1_000_000, limit=1)[0]
        return ws.finish_check(root, name, revision, check, now=1_000_000, retire=operation == "retire")

    def projection(root):
        return (
            ws.group_candidates(root, None, all_groups=True, limit=2).keys,
            ws.group_candidates(root, None, all_groups=False, limit=2).keys,
            ws.due_candidates(root, now=2_000_000, limit=2),
        )

    def exercise(root, crash_at=None):
        setup(root)
        writes = 0

        def wrap(original):
            def write(*args, **kwargs):
                nonlocal writes
                result = original(*args, **kwargs)
                writes += 1
                if writes == crash_at:
                    raise PowerLoss()
                return result

            return write

        with monkeypatch.context() as patch:
            patch.setattr(ws, "atomic_replace", wrap(ws.atomic_replace))
            patch.setattr(tree, "atomic_replace", wrap(tree.atomic_replace))
            if crash_at is None:
                assert update(root)
            else:
                with pytest.raises(PowerLoss):
                    update(root)
        ws.recover_index(root)
        recovered = projection(root)
        ws.recover_index(root)
        assert projection(root) == recovered
        return writes, recovered

    writes, expected = exercise(tmp_path / "reference")
    assert expected[0] == ("name-1",)
    assert expected[1] == (() if operation == "retire" else ("name-1",))
    for boundary in range(1, writes + 1):
        _, recovered = exercise(tmp_path / f"crash-{boundary}", boundary)
        assert recovered == expected


@pytest.mark.parametrize("action", ["create", "wake", "append"])
def test_business_mutation_repairs_crash_before_index_publication(tmp_path, monkeypatch, action):
    from qqtools.plugins.qexp import init_shared_root, observer, submit
    from qqtools.plugins.qexp.commands.group import create_group, wake_group
    from qqtools.plugins.qexp.runtime import group_activity

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "local")
    root = cfg.shared_root
    if action != "create":
        create_group(cfg, "experiment")
        later = ws.next_check_time(root) + 1
        name, revision, due = ws.due_candidates(root, now=later, limit=1)[0]
        assert ws.finish_check(root, name, revision, due, now=later, retire=True)

    def crash(*args, **kwargs):
        raise PowerLoss()

    with monkeypatch.context() as patch:
        patch.setattr(group_activity, "refresh_group", crash)
        with pytest.raises(PowerLoss):
            if action == "create":
                create_group(cfg, "experiment")
            elif action == "wake":
                wake_group(cfg, "experiment")
            else:
                submit(cfg, ["true"], group="experiment", idempotency_key="append")
    assert group_activity.has_pending_activity(root, "experiment")
    assert group_activity.repair_activity_slice(cfg, 1) == 1
    assert group_activity.repair_activity_slice(cfg, 1) == 0
    for history in (False, True):
        page = observer.list_groups_page(cfg, all_groups=history)
        assert [item["group"]["name"] for item in page["items"]] == ["experiment"]
    if action == "append":
        deadline = ws.next_check_time(root)
        replay = submit(cfg, ["true"], group="experiment", idempotency_key="append")
        assert replay.group_membership_sequence == 1
        assert ws.next_check_time(root) == deadline


@pytest.mark.parametrize("cut", ["detach", "unlink"])
def test_interrupted_radix_pruning_preserves_siblings(tmp_path, monkeypatch, cut):
    monkeypatch.setattr(tree, "_MAX_LEAF_KEYS", 2)
    ws.initialize_index(tmp_path)
    for name in ("aa", "ab", "z"):
        ws.refresh_group(tmp_path, name, "creation", now=1)
    _, revision, due = ws.due_candidates(tmp_path, now=1_000_000, limit=3)[-1]
    method = "_persist_node" if cut == "detach" else "_unlink_node"
    original = getattr(tree.IndexTree, method)

    def interrupted(self, *args):
        original(self, *args)
        raise PowerLoss()

    with monkeypatch.context() as patch:
        patch.setattr(tree.IndexTree, method, interrupted)
        with pytest.raises(PowerLoss):
            ws.finish_check(tmp_path, "z", revision, due, now=1_000_000, retire=True)
    for _ in range(2):
        ws.recover_index(tmp_path)
        assert ws.group_candidates(tmp_path, None, all_groups=False, limit=4).keys == ("aa", "ab")
        assert ws.group_candidates(tmp_path, None, all_groups=True, limit=4).keys == ("aa", "ab", "z")
        assert [name for name, _, _ in ws.due_candidates(tmp_path, now=1_000_000, limit=4)] == ["aa", "ab"]


def test_preparing_cleanup_wakes_group_after_crash_before_task_flag(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import init_shared_root, submit
    from qqtools.plugins.qexp.commands import cleanup
    from qqtools.plugins.qexp.runtime.group_activity import repair_activity_slice
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from qqtools.plugins.qexp.scheduler import authority_locks, cancel_task

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "local")
    root = cfg.shared_root
    job = submit(cfg, ["true"], group="exp")
    cancel_task(cfg, job.task_id)
    later = ws.next_check_time(root) + 1
    name, revision, due = ws.due_candidates(root, now=later, limit=1)[0]
    assert ws.finish_check(root, name, revision, due, now=later, retire=True)

    def crash(*args):
        raise PowerLoss()

    with monkeypatch.context() as patch:
        patch.setattr(cleanup, "save_task", crash)
        with authority_locks(cfg, job), pytest.raises(PowerLoss):
            cleanup._start_cleanup_operation(cfg, load_task(cfg, job.task_id))
    assert load_task(cfg, job.task_id).control.get("cleanup_operation_id") is None
    assert repair_activity_slice(cfg, 1) == 1
    assert ws.group_candidates(root, None, all_groups=False, limit=2).keys == ("exp",)


def test_interrupted_empty_group_control_retains_visibility(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import init_shared_root
    from qqtools.plugins.qexp.commands import group as controls
    from qqtools.plugins.qexp.runtime.group_working_set_expiry import expire_due_groups

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "local")
    root = cfg.shared_root
    controls.create_group(cfg, "exp")
    write = controls.write_active_operation

    def crash_after_operation(*args):
        write(*args)
        raise PowerLoss()

    monkeypatch.setattr(controls, "write_active_operation", crash_after_operation)
    with pytest.raises(PowerLoss):
        controls.group_control(cfg, "exp", "cancel")
    assert not controls.show_group(cfg, "exp")["group"]["cancellation_barriers"]
    assert expire_due_groups(cfg, now=ws.next_check_time(root) + 1) == 1
    assert ws.group_candidates(root, None, all_groups=False, limit=1).keys == ("exp",)

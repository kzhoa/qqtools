"""Live indexed Task pages, durable publication, and bounded recovery."""

import base64
import json
import os
from dataclasses import replace

import pytest

from qqtools.plugins.qexp import observer, submit
from qqtools.plugins.qexp.runtime.locks import task_writer_lock
from qqtools.plugins.qexp.runtime.observation import projection
from qqtools.plugins.qexp.runtime.observation.api import ObservationError
from qqtools.plugins.qexp.runtime.observation.maintenance import ObservationMaintenance, request_rebuild
from qqtools.plugins.qexp.runtime.observation.tree import IndexTree
from qqtools.plugins.qexp.runtime.paths import task_path
from qqtools.plugins.qexp.runtime.records import TaskRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import delete_task, load_task, save_task
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def seed(cfg, ids):
    template = submit(cfg, ["true"], task_id=ids[0], group="experiment")
    for key in ids[1:]:
        task = TaskRecord.from_dict(template.to_dict())
        task.task_id = key
        task.submission_operation_id = None
        task.group_membership_sequence = None
        with task_writer_lock(cfg, key, task.group_name):
            save_task(cfg, task)


def change(cfg, key, phase):
    with task_writer_lock(cfg, key, "experiment"):
        task = load_task(cfg, key)
        task.state["projection"] = phase
        task.meta["revision"] += 1
        save_task(cfg, task)


def pages(cfg, **filters):
    cursor = None
    result = []
    for _ in range(1000):
        page = observer.list_tasks_page(cfg, page_size=2, cursor=cursor, **filters)
        result.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return result
    raise AssertionError("cursor did not terminate")


def finish_build(cfg):
    worker = ObservationMaintenance(cfg)
    try:
        for _ in range(20000):
            worker.advance()
            state = projection.read_state(cfg)
            if state and state["state"] == "active" and not state["dirty"]:
                return state
        raise AssertionError(f"index did not recover: {state}")
    finally:
        worker.close()


def rewrite_cursor(token, **changes):
    data = json.loads(base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)))
    data.update(changes)
    return base64.urlsafe_b64encode(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).decode()


def test_pages_equal_legacy_views_for_all_filter_shapes(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["task-1", "task-10", "task-2", "task-3", "task-4"])
    change(cfg, "task-2", "running")
    change(cfg, "task-4", "failed")
    filters = [
        {},
        {"phase": "running"},
        {"group": "experiment"},
        {"phase": "queued", "group": "experiment"},
        {"phase": "absent"},
    ]
    expected = [observer.list_tasks(cfg, limit=1000, **f) for f in filters]

    def no_inventory(*args, **kwargs):
        raise AssertionError("page inventoried history")

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", no_inventory)
        guard.setattr(os, "listdir", no_inventory)
        for f, oracle in zip(filters, expected):
            assert pages(cfg, **f) == oracle


def test_legacy_zero_negative_limits_and_empty_filters_preserved(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["task-1", "task-10", "task-2"])
    assert observer.list_tasks(cfg, limit=0) == []
    assert [x["task_id"] for x in observer.list_tasks(cfg, limit=-1)] == ["task-1", "task-10"]
    assert [x["task_id"] for x in observer.list_tasks(cfg, limit=-2)] == ["task-1"]
    assert observer.list_tasks(cfg, phase="", group="") == observer.list_tasks(cfg)


def test_live_chain_can_change_size_and_observe_changes_ahead(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a", "b", "c", "d"])
    first = observer.list_tasks_page(cfg, page_size=1, phase="queued")
    assert first["items"][0]["task_id"] == "a"
    change(cfg, "b", "failed")
    change(cfg, "a", "failed")
    change(cfg, "a", "queued")
    next_page = observer.list_tasks_page(cfg, page_size=10, phase="queued", cursor=first["next_cursor"])
    assert [item["task_id"] for item in next_page["items"]] == ["c", "d"]
    assert first["index_generation"] == next_page["index_generation"]
    assert next_page["next_cursor"] is None


def test_cursor_binds_identity_filters_order_and_generation(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a", "b"])
    token = observer.list_tasks_page(cfg, page_size=1)["next_cursor"]
    for mutated in [
        "!",
        "x" * 4097,
        rewrite_cursor(token, project="other"),
        rewrite_cursor(token, order="descending"),
        rewrite_cursor(token, last="../bad"),
    ]:
        with pytest.raises(ObservationError) as error:
            observer.list_tasks_page(cfg, cursor=mutated)
        assert error.value.code == "invalid_cursor"
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, cursor=token, phase="queued")
    assert error.value.code == "invalid_cursor"
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, cursor=rewrite_cursor(token, v=3))
    assert error.value.code == "cursor_expired"
    request_rebuild(cfg)
    finish_build(cfg)
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, cursor=token)
    assert error.value.code == "cursor_expired"


@pytest.mark.parametrize("size", [0, -1, 1001, True, "50", 2.5])
def test_page_size_validation(tmp_path, size):
    cfg = isolated_group(tmp_path, tail=0)
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, page_size=size)
    assert error.value.code == "invalid_argument"


def test_empty_budget_page_advances_past_stale_candidates(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["z-live"])
    state = projection.read_state(cfg)
    tree = IndexTree(projection.observation_path(cfg) / "generations" / state["generation"])
    for n in range(260):
        tree.add(f"a-stale-{n:04d}")
    first = observer.list_tasks_page(cfg, page_size=1)
    assert first["items"] == []
    assert first["stop_reason"] == "budget_exhausted"
    assert first["next_cursor"] is not None
    second = observer.list_tasks_page(cfg, page_size=1, cursor=first["next_cursor"])
    assert [item["task_id"] for item in second["items"]] == ["z-live"]


def test_exact_name_filter_uses_sparse_bounded_pages_and_binds_cursor(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, [*[f"a-{index:04d}" for index in range(260)], "z-match"])
    with task_writer_lock(cfg, "z-match", "experiment"):
        task = load_task(cfg, "z-match")
        task.name = "target"
        task.meta["revision"] += 1
        save_task(cfg, task)

    first = observer.list_tasks_page(cfg, name="target", page_size=1)
    assert first["items"] == []
    assert first["stop_reason"] == "budget_exhausted"
    assert first["next_cursor"] is not None
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, name="different", page_size=1, cursor=first["next_cursor"])
    assert error.value.code == "invalid_cursor"
    second = observer.list_tasks_page(cfg, name="target", page_size=1, cursor=first["next_cursor"])
    assert [item["task_id"] for item in second["items"]] == ["z-match"]
    assert second["next_cursor"] is not None
    final = observer.list_tasks_page(cfg, name="target", page_size=1, cursor=second["next_cursor"])
    assert final["items"] == []
    assert final["stop_reason"] == "exhausted"
    assert final["next_cursor"] is None


def test_publication_failure_preserves_truth_and_rebuilds_missing_entry(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])

    def fail(*args, **kwargs):
        raise OSError("injected index failure")

    with monkeypatch.context() as patch:
        patch.setattr(IndexTree, "add", fail)
        change(cfg, "a", "succeeded")
    assert load_task(cfg, "a").state["projection"] == "succeeded"
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, phase="succeeded")
    assert error.value.code == "index_unavailable"
    finish_build(cfg)
    assert [item["task_id"] for item in pages(cfg, phase="succeeded")] == ["a"]
    assert pages(cfg, phase="queued") == []


def test_query_detects_overlapping_publication(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a", "b"])
    original = projection.check_read

    def race(*args, **kwargs):
        change(cfg, "b", "failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(projection, "check_read", race)
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg)
    assert error.value.code == "index_unavailable"


def test_delete_retires_membership_and_recreate_keeps_generation(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a", "b"])
    initial = projection.read_state(cfg)["generation"]
    old = load_task(cfg, "a")
    with task_writer_lock(cfg, "a", "experiment"):
        delete_task(cfg, "a")
    assert [item["task_id"] for item in pages(cfg)] == ["b"]
    with task_writer_lock(cfg, "a", "experiment"):
        save_task(cfg, old)
    assert [item["task_id"] for item in pages(cfg)] == ["a", "b"]
    assert projection.read_state(cfg)["generation"] == initial


def test_corrupt_truth_is_explicit_failure(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    task_path(cfg.shared_root, "a").write_text("broken")
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg)
    assert error.value.code == "index_unavailable"


def test_legacy_root_build_resumes_after_every_slice(tmp_path):
    import shutil

    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, [f"task-{n:02d}" for n in range(12)])
    schema_path = cfg.shared_root / "schema/version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-observation-v1")
    atomic_replace(schema_path, schema)
    shutil.rmtree(projection.observation_path(cfg))
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg)
    assert error.value.code == "index_not_ready"
    for step in range(500):
        worker = ObservationMaintenance(cfg)
        try:
            worker.advance()
        finally:
            worker.close()
        if step == 5:
            change(cfg, "task-10", "failed")
        state = projection.read_state(cfg)
        if state and state["state"] == "active":
            break
    else:
        raise AssertionError(f"restartable build stalled: {state}")
    assert pages(cfg) == observer.list_tasks(cfg, limit=1000)
    assert [item["task_id"] for item in pages(cfg, phase="failed")] == ["task-10"]


@pytest.mark.parametrize("cut", ["intent", "truth", "index"])
def test_real_writer_death_never_serves_stale_negative(tmp_path, cut):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    pid = os.fork()
    if pid == 0:
        original_write = projection.write_state
        original_add = IndexTree.add

        def killed_write(*args, **kwargs):
            result = original_write(*args, **kwargs)
            state = args[1]
            if cut == "intent" and state["dirty"]:
                os._exit(79)
            return result

        def killed_add(*args, **kwargs):
            if cut == "truth":
                os._exit(79)
            result = original_add(*args, **kwargs)
            if cut == "index":
                os._exit(79)
            return result

        projection.write_state = killed_write
        IndexTree.add = killed_add
        change(cfg, "a", "failed")
        os._exit(80)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 79
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg)
    assert error.value.code == "index_unavailable"
    finish_build(cfg)
    assert pages(cfg) == observer.list_tasks(cfg, limit=1000)


def test_dependency_views_keep_complete_live_fields(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    submit(cfg, ["true"], task_id="prerequisite", group="experiment")
    submit(cfg, ["true"], task_id="dependent", group="experiment", depends_on_task_ids=["prerequisite"])
    assert pages(cfg) == observer.list_tasks(cfg, limit=1000)
    change(cfg, "prerequisite", "succeeded")
    assert pages(cfg) == observer.list_tasks(cfg, limit=1000)


def test_repeated_repairs_reclaim_obsolete_generations(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, [f"task-{n:02d}" for n in range(10)])
    generations = projection.observation_path(cfg) / "generations"
    for _ in range(3):
        request_rebuild(cfg)
        finish_build(cfg)
        owner = ObservationMaintenance(cfg)
        try:
            for _ in range(1000):
                owner.advance()
                current = projection.read_state(cfg)["generation"]
                if {path.name for path in generations.iterdir()} == {current}:
                    break
            else:
                raise AssertionError("obsolete generations did not reclaim")
        finally:
            owner.close()
    assert pages(cfg) == observer.list_tasks(cfg, limit=1000)


def test_missing_mandatory_catalog_is_unavailable(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    state = projection.read_state(cfg)
    catalog = IndexTree(projection.observation_path(cfg) / "generations" / state["generation"] / "catalog")
    catalog.root_path.unlink()
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, group="never-existed")
    assert error.value.code == "index_unavailable"


def test_missing_known_partition_is_unavailable(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    state = projection.read_state(cfg)
    partition = IndexTree(projection.observation_path(cfg) / "generations" / state["generation"], phase="queued")
    partition.root_path.unlink()
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg, phase="queued")
    assert error.value.code == "index_unavailable"


def test_degraded_source_does_not_churn_generations_until_explicit_repair(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    original = read_json(task_path(cfg.shared_root, "a"))
    task_path(cfg.shared_root, "a").write_text("broken")
    request_rebuild(cfg)
    owner = ObservationMaintenance(cfg)
    try:
        for _ in range(200):
            owner.advance()
        state = projection.read_state(cfg)
        assert state["state"] == "degraded"
        generation = state["generation"]
        for _ in range(100):
            owner.advance()
        assert projection.read_state(cfg)["generation"] == generation
    finally:
        owner.close()
    atomic_replace(task_path(cfg.shared_root, "a"), original)
    request_rebuild(cfg)
    finish_build(cfg)
    assert [item["task_id"] for item in pages(cfg)] == ["a"]


def test_real_lifecycle_retry_cancel_and_cleanup_publish_pages(tmp_path):
    from qqtools.plugins.qexp.commands.cleanup import clean, reconcile_cleanup_operations
    from qqtools.plugins.qexp.commands.task import cancel, retry
    from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt

    cfg = isolated_group(tmp_path, tail=0)
    task = submit(cfg, ["true"], task_id="lifecycle", group="experiment")
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test_failure")
    assert [item["task_id"] for item in pages(cfg, phase="failed")] == [task.task_id]
    retry(cfg, task.task_id)
    assert pages(cfg, phase="failed") == []
    assert [item["task_id"] for item in pages(cfg, phase="queued")] == [task.task_id]
    cancel(cfg, task.task_id)
    assert [item["task_id"] for item in pages(cfg, phase="cancelled")] == [task.task_id]
    clean(cfg, task_id=task.task_id)
    for _ in range(10):
        if not task_path(cfg.shared_root, task.task_id).exists():
            break
        reconcile_cleanup_operations(cfg)
    assert not task_path(cfg.shared_root, task.task_id).exists()
    assert pages(cfg) == []


def test_obsolete_generation_symlink_never_deletes_external_data(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "keep.json"
    sentinel.write_text("preserve")
    generations = projection.observation_path(cfg) / "generations"
    (generations / ("f" * 32)).symlink_to(external, target_is_directory=True)
    owner = ObservationMaintenance(cfg)
    try:
        for _ in range(40):
            try:
                owner.advance()
            except (OSError, ValueError, RuntimeError):
                pass
    finally:
        owner.close()
    assert sentinel.read_text() == "preserve"
    assert [item["task_id"] for item in pages(cfg)] == ["a"]


def test_corrupt_state_is_diagnosed_and_explicitly_rebuildable(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["a"])
    (projection.observation_path(cfg) / "state.json").write_text("invalid")
    assert projection.inspect_observation(cfg)["state"] == "degraded"
    with pytest.raises(ObservationError) as error:
        observer.list_tasks_page(cfg)
    assert error.value.code == "index_unavailable"
    request_rebuild(cfg)
    finish_build(cfg)
    assert [item["task_id"] for item in pages(cfg)] == ["a"]


def test_capture_io_failure_is_not_end_of_directory(tmp_path, monkeypatch):
    import ctypes
    import errno

    from qqtools.plugins.qexp.runtime import directory_capture as capture

    def failed_readdir(_stream):
        ctypes.set_errno(errno.EIO)
        return None

    monkeypatch.setattr(capture._LIBC, "readdir", failed_readdir)
    with pytest.raises(OSError) as error:
        capture.read_directory_entry(tmp_path, 0)
    assert error.value.errno == errno.EIO
    with pytest.raises(ValueError):
        capture.read_directory_entry(tmp_path, 1 << 64)


def test_machine_worker_rotates_and_isolates_one_failed_project(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from qqtools.plugins.qexp.agent.context import MachineRuntime, ProjectBinding
    from qqtools.plugins.qexp.runtime.observation import maintenance

    runtime = MachineRuntime(tmp_path / "machine-runtime")
    from qqtools.plugins.qexp import init_shared_root

    configs = [
        init_shared_root(tmp_path / f"p-{n}" / ".qexp", "g1", runtime_root=tmp_path / f"rt-{n}") for n in range(2)
    ]
    bindings = [ProjectBinding(f"p-{n}", cfg.shared_root, "g1") for n, cfg in enumerate(configs)]
    calls = []
    closed = []
    worker = maintenance.MachineObservationWorker(runtime)

    class ControlledMaintenance:
        def __init__(self, cfg):
            self.name = cfg.shared_root.parent.name

        def advance(self):
            calls.append(self.name)
            if len(calls) >= 8:
                worker._stop.set()
            if self.name == "p-0":
                raise OSError("isolated project error")
            return {"state": "building"}

        def close(self):
            closed.append(self.name)

    @contextmanager
    def eligible(_binding):
        yield True

    monkeypatch.setattr(maintenance, "ObservationMaintenance", ControlledMaintenance)
    monkeypatch.setattr(runtime, "load_registry", lambda: (1, bindings))
    monkeypatch.setattr(runtime, "binding_write_guard", eligible)
    monkeypatch.setattr(runtime, "binding_state", lambda _binding: "enabled")
    from threading import Timer

    deadline = Timer(3, worker._stop.set)
    deadline.start()
    try:
        worker._run()
    finally:
        deadline.cancel()
        deadline.join()
    assert calls == ["p-0", "p-1"] * 4
    assert sorted(closed) == ["p-0", "p-1"]


@pytest.mark.parametrize("phase", ["prepared", "moved"])
def test_observation_waits_for_completed_group_isolation(tmp_path, monkeypatch, phase):
    import shutil

    from qqtools.plugins.qexp import init_shared_root
    from qqtools.plugins.qexp.runtime import group_namespace
    from qqtools.plugins.qexp.runtime.locks import schema_lock

    cfg = init_shared_root(tmp_path / ".qexp", "g1", runtime_root=tmp_path / "runtime")
    schema_path = cfg.shared_root / "schema/version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-observation-v1")
    schema["schema"]["required_capabilities"].append("local-recovery-v1")
    atomic_replace(schema_path, schema)
    shutil.rmtree(projection.observation_path(cfg))
    original = group_namespace.atomic_replace

    def interrupted(path, value):
        original(path, value)
        if value.get("group_authority", {}).get("phase") == phase:
            raise OSError("interrupted Group isolation")

    with monkeypatch.context() as fault:
        fault.setattr(group_namespace, "atomic_replace", interrupted)
        with schema_lock(cfg.shared_root), pytest.raises(OSError):
            group_namespace.activate_group_authority_locked(cfg)
    assert group_namespace.has_group_authority_cutover(cfg.shared_root)
    assert not group_namespace.is_group_authority_isolated(cfg.shared_root)
    owner = ObservationMaintenance(cfg)
    try:
        for _ in range(3):
            owner.advance()
        assert projection.read_state(cfg) is None
        assert "task-observation-v1" not in read_json(schema_path)["schema"]["required_capabilities"]
        with schema_lock(cfg.shared_root):
            assert group_namespace.activate_group_authority_locked(cfg)
        for _ in range(20):
            owner.advance()
            state = projection.read_state(cfg)
            if state and state["state"] == "active":
                break
        assert state["state"] == "active"
    finally:
        owner.close()


def test_publisher_finishing_before_maintenance_lock_does_not_restart_history(tmp_path, monkeypatch):
    cfg = isolated_group(tmp_path, tail=0)
    seed(cfg, ["settled"])
    before = finish_build(cfg)
    read_state = projection.read_state
    reads = 0

    def observed_during_publication(candidate):
        nonlocal reads
        reads += 1
        state = read_state(candidate)
        # The optimistic read sees an in-flight intent. By the time maintenance
        # acquires ownership, the real publisher has completed the same generation.
        return {**state, "dirty": True} if reads == 2 else state

    monkeypatch.setattr(projection, "read_state", observed_during_publication)
    worker = ObservationMaintenance(cfg)
    try:
        monkeypatch.setattr(worker, "_capture_one", lambda state: pytest.fail("clean history was scanned"))
        result = worker.advance()
        assert result["state"] == "active"
        assert read_state(cfg) == before
    finally:
        worker.close()

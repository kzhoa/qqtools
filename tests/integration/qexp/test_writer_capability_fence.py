"""Writers validate capabilities inside the fence, before any authority mutation."""

from contextlib import contextmanager
from threading import Event, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.cleanup import clean, reconcile_cleanup_operations
from qqtools.plugins.qexp.commands.group import create_group, group_control
from qqtools.plugins.qexp.runtime import locks
from qqtools.plugins.qexp.runtime.paths import attempt_path, group_path, ready_state_path, task_path
from qqtools.plugins.qexp.runtime.ready import advance_ready_index_build
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt

pytestmark = pytest.mark.integration


def install_gate(cfg, gate):
    with locks.schema_lock(cfg.shared_root):
        if gate == "required":
            path = cfg.shared_root / "schema" / "version.json"
        elif gate == "group_state":
            path = cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json"
        else:
            path = ready_state_path(cfg.shared_root)
        value = read_json(path)
        if gate == "required":
            value["schema"]["required_capabilities"].append("future-writer-v1")
        elif gate == "group_state":
            value["group_ready_members"]["state"] = "degraded"
        else:
            assert value["ready_index"]["state"] in {"building", "active"}
            value["ready_index"]["writer_capability"] = "future-ready-v1"
        atomic_replace(path, value)


@pytest.fixture
def cfg(tmp_path):
    value = init_shared_root(tmp_path / ".qexp", "writer", runtime_root=tmp_path / "runtime")
    advance_ready_index_build(value)
    return value


@pytest.mark.parametrize("is_narrow", [False, True])
@pytest.mark.parametrize("gate", ["required", "ready"])
def test_cached_task_writer_rejects_before_entering_authority_section(cfg, is_narrow, gate):
    task = submit(cfg, ["true"])
    cached = load_task(cfg, task.task_id)
    path = task_path(cfg.shared_root, task.task_id)
    original = path.read_bytes()
    if not is_narrow:
        schema_path = cfg.shared_root / "schema" / "version.json"
        schema = read_json(schema_path)
        schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
        atomic_replace(schema_path, schema)
    install_gate(cfg, gate)
    has_entered = False
    with pytest.raises(RuntimeError, match="capabilit"):
        with locks.schema_writer_lock(cfg):
            has_entered = True
            save_task(cfg, cached)
    assert not has_entered
    assert path.read_bytes() == original


@pytest.mark.parametrize("gate", ["required", "group_state"])
def test_ungrouped_submission_rechecks_after_waiting_for_schema_fence(cfg, monkeypatch, gate):
    entered, resume = Event(), Event()
    original = locks.schema_reader_lock
    outcomes = []

    @contextmanager
    def delayed(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        with original(*args, **kwargs) as acquired:
            yield acquired

    def writer():
        try:
            outcomes.append(submit(cfg, ["true"]))
        except BaseException as exc:
            outcomes.append(exc)

    monkeypatch.setattr(locks, "schema_reader_lock", delayed)
    thread = Thread(target=writer)
    thread.start()
    try:
        assert entered.wait(5)
        install_gate(cfg, gate)
    finally:
        resume.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], RuntimeError), outcomes
    assert ("unsupported capabilities" if gate == "required" else "ordinary mutation is disabled") in str(outcomes[0])
    assert not list((cfg.shared_root / "tasks").glob("*.json"))


@pytest.mark.parametrize("gate", ["required", "ready"])
def test_group_control_rejects_before_group_truth_mutation(cfg, gate):
    create_group(cfg, "experiment")
    path = group_path(cfg.shared_root, "experiment")
    before = path.read_bytes()
    install_gate(cfg, gate)
    with pytest.raises(RuntimeError, match="capabilit"):
        group_control(cfg, "experiment", "seal")
    assert path.read_bytes() == before


@pytest.mark.parametrize("gate", ["required", "ready"])
def test_interrupted_cleanup_with_missing_task_cannot_delete_attempts_after_gate(cfg, monkeypatch, gate):
    from qqtools.plugins.qexp.commands import cleanup

    task = submit(cfg, ["true"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "fixture")
    path = attempt_path(cfg.shared_root, task.task_id, 1)
    before = path.read_bytes()
    original = cleanup.shutil.rmtree

    def interrupt(directory, *args, **kwargs):
        if directory == path.parent:
            raise OSError("interrupted after Task deletion")
        return original(directory, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cleanup.shutil, "rmtree", interrupt)
        with pytest.raises(OSError, match="interrupted after Task deletion"):
            clean(cfg, task_id=task.task_id)
    assert not task_path(cfg.shared_root, task.task_id).exists()
    assert path.read_bytes() == before
    install_gate(cfg, gate)
    with pytest.raises(RuntimeError, match="capabilit"):
        reconcile_cleanup_operations(cfg)
    assert path.read_bytes() == before


def test_ready_writer_gate_precedes_terminal_attempt_mutation(cfg):
    task = submit(cfg, ["true"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, 1)
    before = path.read_bytes()
    install_gate(cfg, "ready")
    with pytest.raises(RuntimeError, match="capabilit"):
        fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "rejected")
    assert path.read_bytes() == before

from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path

import pytest

from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.group import create_group, group_control
from qqtools.plugins.qexp.commands.task import cancel, edit_dependencies, submit
from qqtools.plugins.qexp.machine_config import init_shared_root
from qqtools.plugins.qexp.runtime import submission as submission_runtime
from qqtools.plugins.qexp.runtime.dependencies import dependency_gate
from qqtools.plugins.qexp.runtime.dependency_projection import (
    advance_rebuild,
    mark_degraded,
    projected_dependents,
    read_projection_state,
    start_rebuild,
)
from qqtools.plugins.qexp.runtime.locks import dependency_lock
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from qqtools.plugins.qexp.runtime.upgrade import UpgradeCoordinator
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _config(tmp_path: Path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "train", ["gpu-1"])
    create_group(cfg, "eval", ["gpu-1"])
    return cfg


def _set_projection(cfg, task_id: str, projection: str) -> None:
    task = load_task(cfg, task_id)
    task.state["projection"] = projection
    task.meta["revision"] += 1
    atomic_replace(cfg.shared_root / "tasks" / f"{task_id}.json", task.to_dict())


def _edit_in_process(cfg, task_id: str, dependency_id: str, ready, output) -> None:
    ready.wait()
    try:
        edit_dependencies(cfg, task_id, [dependency_id])
    except Exception as exc:  # pragma: no cover - assertion is in the parent
        output.put(("error", type(exc).__name__, str(exc)))
    else:
        output.put(("committed", task_id))


def test_fresh_root_enables_project_dependencies_and_empty_projection(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    schema = read_json(cfg.shared_root / "schema" / "version.json")["schema"]

    assert "task-dependencies-v1" in schema["required_capabilities"]
    assert "task-dependencies-v2" in schema["required_capabilities"]
    state = read_json(cfg.shared_root / "indexes" / "task-dependencies" / "state.json")
    assert state["dependency_projection"]["state"] == "active"


def test_cross_group_dependency_waits_then_becomes_claimable(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "train"], task_id="train-a", group="train", working_dir=tmp_path)
    child = submit(
        cfg,
        ["echo", "eval"],
        task_id="eval-c",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )

    assert dependency_gate(cfg, child).state == "waiting"
    assert claim_task(cfg, child.task_id, [0]) is None

    _set_projection(cfg, parent.task_id, "succeeded")

    assert dependency_gate(cfg, load_task(cfg, child.task_id)).state == "ready"
    assert claim_task(cfg, child.task_id, [0]) is not None


@pytest.mark.parametrize(
    ("parent_group", "child_group"),
    [(None, "eval"), ("train", None)],
)
def test_grouped_and_ungrouped_tasks_may_depend_in_either_direction(
    tmp_path: Path,
    parent_group: str | None,
    child_group: str | None,
) -> None:
    cfg = _config(tmp_path)
    parent = submit(
        cfg,
        ["echo", "parent"],
        task_id="parent",
        group=parent_group,
        working_dir=tmp_path,
    )
    child = submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group=child_group,
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )

    assert child.depends_on_task_ids == [parent.task_id]
    assert dependency_gate(cfg, child).state == "waiting"
    _set_projection(cfg, parent.task_id, "succeeded")
    assert dependency_gate(cfg, load_task(cfg, child.task_id)).state == "ready"


def test_explicit_fan_in_and_transitive_ancestor_semantics(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = submit(cfg, ["echo", "a"], task_id="a", group="train", working_dir=tmp_path)
    second = submit(cfg, ["echo", "b"], task_id="b", group="eval", working_dir=tmp_path)
    fan_in = submit(
        cfg,
        ["echo", "fan-in"],
        task_id="fan-in",
        working_dir=tmp_path,
        depends_on_task_ids=[second.task_id, first.task_id],
    )

    _set_projection(cfg, first.task_id, "succeeded")
    assert dependency_gate(cfg, load_task(cfg, fan_in.task_id)).state == "waiting"
    _set_projection(cfg, second.task_id, "succeeded")
    assert dependency_gate(cfg, load_task(cfg, fan_in.task_id)).state == "ready"

    middle = submit(
        cfg,
        ["echo", "middle"],
        task_id="middle",
        group="train",
        working_dir=tmp_path,
        depends_on_task_ids=[first.task_id],
    )
    tail = submit(
        cfg,
        ["echo", "tail"],
        task_id="tail",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[middle.task_id],
    )
    assert tail.depends_on_task_ids == [middle.task_id]
    assert dependency_gate(cfg, tail).state == "waiting"
    _set_projection(cfg, middle.task_id, "succeeded")
    assert dependency_gate(cfg, load_task(cfg, tail.task_id)).state == "ready"


def test_cross_group_cycle_edit_is_atomic(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = submit(cfg, ["echo", "a"], task_id="a", group="train", working_dir=tmp_path)
    second = submit(
        cfg,
        ["echo", "b"],
        task_id="b",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[first.task_id],
    )

    with pytest.raises(ValueError, match="cycle"):
        edit_dependencies(cfg, first.task_id, [second.task_id])

    assert load_task(cfg, first.task_id).depends_on_task_ids == []
    assert load_task(cfg, second.task_id).depends_on_task_ids == [first.task_id]


def test_concurrent_opposite_edits_commit_at_most_one_edge(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    submit(cfg, ["echo", "a"], task_id="a", group="train", working_dir=tmp_path)
    submit(cfg, ["echo", "b"], task_id="b", group="eval", working_dir=tmp_path)
    context = get_context("fork")
    ready = context.Event()
    output = context.Queue()
    processes = [
        context.Process(target=_edit_in_process, args=(cfg, "a", "b", ready, output)),
        context.Process(target=_edit_in_process, args=(cfg, "b", "a", ready, output)),
    ]
    for process in processes:
        process.start()
    ready.set()
    results = [output.get(timeout=10) for _ in processes]
    for process in processes:
        process.join(10)
        assert process.exitcode == 0

    assert sum(result[0] == "committed" for result in results) == 1
    assert sum(result[0] == "error" and "cycle" in result[2] for result in results) == 1
    assert not (load_task(cfg, "a").depends_on_task_ids == ["b"] and load_task(cfg, "b").depends_on_task_ids == ["a"])


@pytest.mark.parametrize("child_group", ["eval", None])
def test_cleanup_rejects_project_wide_direct_dependents(tmp_path: Path, child_group: str | None) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="train", working_dir=tmp_path)
    submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group=child_group,
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )
    cancel(cfg, parent.task_id)

    with pytest.raises(ValueError, match="referenced by: child"):
        clean(cfg, task_id=parent.task_id)


def test_group_lifecycle_actions_do_not_propagate_across_dependency(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="train", working_dir=tmp_path)
    child = submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )

    group_control(cfg, "train", "seal")

    assert load_task(cfg, parent.task_id).state["projection"] == "queued"
    assert load_task(cfg, child.task_id).state["projection"] == "queued"
    assert dependency_gate(cfg, load_task(cfg, child.task_id)).state == "waiting"


def test_project_local_missing_dependency_does_not_search_another_root(tmp_path: Path) -> None:
    cfg = _config(tmp_path / "one")
    other = _config(tmp_path / "two")
    foreign = submit(other, ["echo", "foreign"], task_id="foreign", group="train", working_dir=tmp_path)

    with pytest.raises(ValueError, match="foreign.*does not exist"):
        submit(
            cfg,
            ["echo", "child"],
            task_id="child",
            group="eval",
            working_dir=tmp_path,
            depends_on_task_ids=[foreign.task_id],
        )


def test_process_without_v2_capability_rejects_root_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qqtools.plugins.qexp import layout

    cfg = _config(tmp_path)
    monkeypatch.setattr(
        layout,
        "SUPPORTED_REQUIRED_CAPABILITIES",
        layout.SUPPORTED_REQUIRED_CAPABILITIES - {"task-dependencies-v2"},
    )

    with pytest.raises(RuntimeError, match="unsupported capabilities: task-dependencies-v2"):
        submit(cfg, ["echo", "old-writer"], task_id="old-writer", working_dir=tmp_path)

    assert not (cfg.shared_root / "tasks" / "old-writer.json").exists()


def test_ungrouped_same_submission_candidate_dependency_commits_atomically(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    from qqtools.plugins.qexp.runtime.submission import submit_specs

    result = submit_specs(
        cfg,
        [
            {"task_id": "parent", "command": ["echo", "parent"]},
            {
                "task_id": "child",
                "command": ["echo", "child"],
                "depends_on_task_ids": ["parent"],
            },
        ],
        group_name=None,
        idempotency_key="ungrouped-graph",
    )

    assert [task.task_id for task in result] == ["parent", "child"]
    assert load_task(cfg, "child").depends_on_task_ids == ["parent"]


def test_interrupted_submission_rollback_preserves_degraded_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    original_save = submission_runtime.save_task

    def fail_second_task(cfg_value, task):
        if task.task_id == "child":
            raise OSError("test interrupted submission")
        return original_save(cfg_value, task)

    monkeypatch.setattr(submission_runtime, "save_task", fail_second_task)
    with pytest.raises(OSError, match="test interrupted submission"):
        submission_runtime.submit_specs(
            cfg,
            [
                {"task_id": "parent", "command": ["echo", "parent"]},
                {
                    "task_id": "child",
                    "command": ["echo", "child"],
                    "depends_on_task_ids": ["parent"],
                },
            ],
            group_name=None,
            idempotency_key="interrupted-project-graph",
        )

    assert not (cfg.shared_root / "tasks" / "parent.json").exists()
    assert not (cfg.shared_root / "tasks" / "child.json").exists()
    state = read_projection_state(cfg)
    assert state["state"] == "degraded"
    assert "canonical dependency mutation" in state["failure_reason"]


def test_degraded_projection_blocks_cleanup_but_not_dependency_gate(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="train", working_dir=tmp_path)
    child = submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )
    cancel(cfg, parent.task_id)
    state_path = cfg.shared_root / "indexes" / "task-dependencies" / "state.json"
    state = read_json(state_path)
    state["dependency_projection"]["state"] = "degraded"
    atomic_replace(state_path, state)

    assert dependency_gate(cfg, load_task(cfg, child.task_id)).state == "blocked"
    with pytest.raises((RuntimeError, ValueError), match="projection|coverage|degraded"):
        clean(cfg, task_id=parent.task_id)


def test_cleanup_projection_mismatch_durably_degrades_coverage(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="train", working_dir=tmp_path)
    child = submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )
    cancel(cfg, parent.task_id)
    state = read_projection_state(cfg)
    record_path = (
        cfg.shared_root
        / "indexes"
        / "task-dependencies"
        / "generations"
        / state["active_generation"]
        / "records"
        / f"{parent.task_id}.json"
    )
    record = read_json(record_path)
    record["task_dependency_record"]["dependent_task_ids"] = []
    atomic_replace(record_path, record)

    with pytest.raises((RuntimeError, ValueError), match="projection|coverage"):
        clean(cfg, task_id=parent.task_id)

    degraded = read_projection_state(cfg)
    assert degraded["state"] == "degraded"
    assert "cleanup projection validation failed" in degraded["failure_reason"]
    assert load_task(cfg, child.task_id).depends_on_task_ids == [parent.task_id]


def test_projection_rebuild_uses_bounded_cursors_and_restores_reverse_coverage(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="train", working_dir=tmp_path)
    submit(
        cfg,
        ["echo", "child"],
        task_id="child",
        group="eval",
        working_dir=tmp_path,
        depends_on_task_ids=[parent.task_id],
    )
    with dependency_lock(cfg.shared_root):
        mark_degraded(cfg, "test rebuild")
        state = start_rebuild(cfg)
    assert state["cursor"] == {"offset": 0}

    results = []
    for _ in range(20):
        with dependency_lock(cfg.shared_root):
            result = advance_rebuild(cfg, max_items=1)
        results.append(result)
        assert result["work_items"] <= 1
        if result["state"] == "complete":
            break
    else:  # pragma: no cover - bounded failure diagnostic
        pytest.fail(f"projection rebuild did not converge: {results[-1]}")

    assert read_projection_state(cfg)["state"] == "active"
    assert projected_dependents(cfg, parent.task_id) == ["child"]
    assert all("task_ids" not in (item.get("cursor") or {}) for item in results)


def test_legacy_task_creation_invalidates_an_in_progress_projection_build(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-dependencies-v2")
    atomic_replace(schema_path, schema)
    with dependency_lock(cfg.shared_root):
        mark_degraded(cfg, "test legacy rebuild")
        assert start_rebuild(cfg)["state"] == "building"

    submit(cfg, ["echo", "legacy"], task_id="legacy", group="train", working_dir=tmp_path)

    state = read_projection_state(cfg)
    assert state["state"] == "degraded"
    assert "legacy dependency mutation" in state["failure_reason"]


def test_upgrade_recovers_projection_invalidated_before_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qqtools.plugins.qexp.runtime.upgrade import task_dependencies_v2

    cfg = _config(tmp_path)
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("task-dependencies-v2")
    atomic_replace(schema_path, schema)
    monkeypatch.setattr(
        task_dependencies_v2,
        "__version__",
        task_dependencies_v2.TASK_DEPENDENCIES_V2_WRITER_FLOOR,
    )
    coordinator = UpgradeCoordinator(cfg)
    assert coordinator.discover()["pending"]
    for _ in range(100):
        status = coordinator.status()
        migration = next(
            (item for item in status["migrations"] if item["name"] == "task-dependencies-v2"),
            None,
        )
        if migration is not None and migration["phase"] == "activation":
            break
        coordinator.advance(force_retry=True)
    else:  # pragma: no cover - bounded failure diagnostic
        pytest.fail(f"Task dependency migration did not reach activation: {coordinator.status()}")

    with dependency_lock(cfg.shared_root):
        mark_degraded(cfg, "test invalidation after audit")

    for _ in range(100):
        status = coordinator.advance(force_retry=True)
        if not status["pending"]:
            break
    else:  # pragma: no cover - bounded failure diagnostic
        pytest.fail(f"Task dependency migration did not converge: {coordinator.status()}")

    assert read_projection_state(cfg)["state"] == "active"
    capabilities = read_json(schema_path)["schema"]["required_capabilities"]
    assert "task-dependencies-v2" in capabilities


def test_interrupted_edge_projection_stays_degraded_until_bounded_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path)
    parent = submit(cfg, ["echo", "parent"], task_id="parent", group="train", working_dir=tmp_path)
    child = submit(cfg, ["echo", "child"], task_id="child", group="eval", working_dir=tmp_path)

    def interrupt_projection(*_args, **_kwargs):
        raise OSError("projection write interrupted")

    monkeypatch.setattr("qqtools.plugins.qexp.commands.task.synchronize_task_edges", interrupt_projection)
    with pytest.raises(OSError, match="projection write interrupted"):
        edit_dependencies(cfg, child.task_id, [parent.task_id])
    monkeypatch.undo()

    assert load_task(cfg, child.task_id).depends_on_task_ids == [parent.task_id]
    assert read_projection_state(cfg)["state"] == "degraded"
    assert dependency_gate(cfg, load_task(cfg, child.task_id)).state == "waiting"

    for _ in range(20):
        with dependency_lock(cfg.shared_root):
            result = advance_rebuild(cfg, max_items=1)
        if result["state"] == "complete":
            break
    else:  # pragma: no cover - bounded failure diagnostic
        pytest.fail("projection repair did not converge")
    assert projected_dependents(cfg, parent.task_id) == [child.task_id]

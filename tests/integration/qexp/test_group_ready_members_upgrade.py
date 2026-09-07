from pathlib import Path
from threading import Barrier, Thread

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.cli import main
from qqtools.plugins.qexp.commands.cleanup import clean
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.doctor import repair_metadata, verify_integrity
from qqtools.plugins.qexp.group_ready_members_upgrade import (
    attest_group_ready_members_upgrade,
    resume_group_ready_members_upgrade,
    start_group_ready_members_upgrade,
)
from qqtools.plugins.qexp.runtime.locks import is_schema_narrow_protocol_active
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.ready import advance_ready_index_build, is_primary_ready_index_active
from qqtools.plugins.qexp.runtime.ready.group_members import (
    group_ready_members_state,
    mark_group_ready_members_degraded,
    read_group_ready_members,
    retire_group_ready_member,
)
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.scheduler import claim_task, fail_attempt

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_group_ready_members_upgrade_cli_exposes_status(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    assert (
        main(
            [
                "--shared-root",
                str(cfg.shared_root),
                "upgrade",
                "group-ready-members",
                "status",
            ]
        )
        == 0
    )


def test_new_root_publishes_and_retires_exact_group_ready_members(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "one"], task_id="member-task", group="exp")

    members = read_group_ready_members(cfg, "exp")
    assert [(item["task_id"], item["generation"]) for item in members] == [(task.task_id, task.ready_generation)]

    from qqtools.plugins.qexp.scheduler import claim_task

    assert claim_task(cfg, task.task_id, [0]) is not None
    assert read_group_ready_members(cfg, "exp") == []


# QQTOOLS-COMPAT-0008: legacy roots activate the member protocol through a resumable upgrader.
def test_legacy_root_backfills_then_jointly_activates_narrow_schema_protocol(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "legacy"], task_id="legacy-member", group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()

    session = start_group_ready_members_upgrade(cfg)
    assert group_ready_members_state(cfg) == "building"
    assert not is_schema_narrow_protocol_active(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )
    first = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    completed = first
    while completed["phase"] == "building":
        completed = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )

    assert completed["phase"] == "completed"
    assert is_schema_narrow_protocol_active(cfg)
    assert [(item["task_id"], item["generation"]) for item in read_group_ready_members(cfg, "exp")] == [
        (task.task_id, task.ready_generation)
    ]


def test_active_group_change_uses_members_and_exact_primary_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    from qqtools.plugins.qexp.commands import group as group_commands
    from qqtools.plugins.qexp.runtime.ready import index as ready_runtime

    state = advance_ready_index_build(cfg)
    while state["state"] == "building":
        state = advance_ready_index_build(cfg)
    assert is_primary_ready_index_active(cfg)

    def fail_task_scan(path):
        if Path(path) == cfg.shared_root / "tasks":
            raise AssertionError("active Group projection must not scan all Tasks")
        return iter(())

    monkeypatch.setattr(group_commands, "iter_json", fail_task_scan)
    primary_routes = shared_paths(cfg.shared_root)["ready_primary"] / "routes"
    original_scandir = ready_runtime.os.scandir

    def fail_primary_route_scan(path):
        if isinstance(path, (str, Path)) and Path(path) == primary_routes:
            raise AssertionError("active Group projection must not scan all primary routes")
        return original_scandir(path)

    monkeypatch.setattr(ready_runtime.os, "scandir", fail_primary_route_scan)
    change_worker(cfg, "exp", "gpu-2", "add", role="primary")


def test_group_members_are_paged_without_historical_task_scan(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for index in range(65):
        submit(cfg, ["echo", str(index)], task_id=f"member-{index}", group="exp")

    members = read_group_ready_members(cfg, "exp")
    assert len(members) == 65
    assert {item["task_id"] for item in members} == {f"member-{index}" for index in range(65)}


def test_retired_empty_pages_reuse_a_bounded_directory_slot(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    tasks = [submit(cfg, ["echo", str(index)], task_id=f"member-{index}", group="exp") for index in range(65)]
    from qqtools.plugins.qexp.runtime.ready import group_members

    for task in tasks:
        assert retire_group_ready_member(cfg, "exp", task.task_id, task.ready_generation)
    state = read_json(group_members._group_state_path(cfg, "exp"))["group_ready_members"]
    assert state["directory_page_count"] == 1
    assert state["next_member_page"] == 2
    assert state["writable_member_page"] in {0, 1}

    submit(cfg, ["echo", "reused"], task_id="member-reused", group="exp")
    state = read_json(group_members._group_state_path(cfg, "exp"))["group_ready_members"]
    assert state["directory_page_count"] == 1
    assert state["next_member_page"] == 2


def test_writable_page_index_reuses_every_retired_page_across_churn(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    from qqtools.plugins.qexp.runtime.ready import group_members

    for cycle in range(4):
        tasks = [
            submit(cfg, ["echo", str(index)], task_id=f"cycle-{cycle}-{index}", group="exp")
            for index in range(65)
        ]
        for task in tasks:
            assert retire_group_ready_member(cfg, "exp", task.task_id, task.ready_generation)
        state = read_json(group_members._group_state_path(cfg, "exp"))["group_ready_members"]
        assert state["member_count"] == 0
        assert state["next_member_page"] == 2
        assert state["directory_page_count"] == 1
        assert state["writable_index_count"] <= 1


def test_archive_cleanup_reclaims_writable_index_pages(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    tasks = [
        submit(cfg, ["echo", str(index)], task_id=f"archive-{index}", group="exp")
        for index in range(129)
    ]
    for task in tasks[:128]:
        assert retire_group_ready_member(cfg, "exp", task.task_id, task.ready_generation)
    from qqtools.plugins.qexp.runtime.ready import group_members

    writable_root = group_members._group_root(cfg, "exp") / "writable-pages"
    assert len(list(writable_root.glob("*.json"))) >= 1
    mark_group_ready_members_degraded(cfg, "test")
    repair_metadata(cfg, max_work_items=64)
    for _ in range(200):
        result = clean(cfg, dry_run=False, max_work_items=64)
        cleanup = result["group_ready_member_archive_cleanup"]
        if cleanup["state"] == "completed" and cleanup["archive_count"] == 0:
            break
    else:
        raise AssertionError("archive cleanup did not converge")


def test_verify_detects_missing_writable_index_page(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    tasks = [
        submit(cfg, ["echo", str(index)], task_id=f"audit-index-{index}", group="exp")
        for index in range(129)
    ]
    for task in tasks[:128]:
        assert retire_group_ready_member(cfg, "exp", task.task_id, task.ready_generation)
    from qqtools.plugins.qexp.runtime.ready import group_members

    writable_root = group_members._group_root(cfg, "exp") / "writable-pages"
    index_path = next(writable_root.glob("*.json"))
    index_path.unlink()
    result = verify_integrity(cfg, max_work_items=64)
    while not result["complete"]:
        result = verify_integrity(cfg, max_work_items=64)
    assert result["healthy"] is False
    assert "group_ready_members_inconsistent" in {item["code"] for item in result["issues"]}
    assert group_ready_members_state(cfg) == "degraded"


def test_verify_detects_full_current_writable_page_pointer(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for index in range(64):
        submit(cfg, ["echo", str(index)], task_id=f"full-pointer-{index}", group="exp")

    from qqtools.plugins.qexp.runtime.ready import group_members

    state_path = group_members._group_state_path(cfg, "exp")
    state = read_json(state_path)
    state["group_ready_members"]["writable_member_page"] = 0
    atomic_replace(state_path, state)

    result = verify_integrity(cfg, max_work_items=64)
    while not result["complete"]:
        result = verify_integrity(cfg, max_work_items=64)
    assert result["healthy"] is False
    assert "group_ready_members_inconsistent" in {item["code"] for item in result["issues"]}
    assert group_ready_members_state(cfg) == "degraded"


def test_verify_detects_directory_page_cycle(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for index in range(65):
        submit(cfg, ["echo", str(index)], task_id=f"directory-cycle-{index}", group="exp")

    from qqtools.plugins.qexp.runtime.ready import group_members

    directory_path = group_members._directory_path(cfg, "exp", 0)
    directory = read_json(directory_path)
    directory["group_ready_member_directory"]["next_page"] = 0
    atomic_replace(directory_path, directory)

    result = verify_integrity(cfg, max_work_items=64)
    for _ in range(20):
        if result["complete"]:
            break
        result = verify_integrity(cfg, max_work_items=64)
    assert result["complete"] is True
    assert result["healthy"] is False
    assert "group_ready_members_inconsistent" in {item["code"] for item in result["issues"]}
    assert group_ready_members_state(cfg) == "degraded"


def test_locator_damage_degrades_then_doctor_rebuilds_active_projection(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    from qqtools.plugins.qexp.runtime.ready import group_members

    entry = read_group_ready_members(cfg, "exp")[0]
    locator_path = group_members._locator_path(cfg, "exp", entry["identity"])
    locator = read_json(locator_path)
    locator["group_ready_member_locator"]["page"] = 1
    atomic_replace(locator_path, locator)

    with pytest.raises(RuntimeError, match="ready-member projection is invalid"):
        read_group_ready_members(cfg, "exp")
    assert group_ready_members_state(cfg) == "degraded"

    result = repair_metadata(cfg)
    while result["group_ready_members"]["state"] == "building":
        result = repair_metadata(cfg)

    assert result["group_ready_members"]["state"] == "active"
    assert [(item["task_id"], item["generation"]) for item in read_group_ready_members(cfg, "exp")] == [
        (task.task_id, task.ready_generation)
    ]


def test_doctor_repair_advances_only_one_member_projection_slice(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for index in range(65):
        submit(cfg, ["echo", str(index)], task_id=f"member-{index}", group="exp")
    mark_group_ready_members_degraded(cfg, "injected projection damage")

    result = repair_metadata(cfg)

    projection = result["group_ready_members"]
    assert projection["state"] == "building"
    assert projection["build"]["phase"] == "backfill"
    assert projection["build"]["watermark"]["capture"]["task_count"] == 64
    assert "group_ready_members" not in result["repaired"]
    assert "rerun doctor repair" in result["message"]


def test_doctor_verify_degrades_when_a_group_member_directory_is_missing(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    import shutil

    from qqtools.plugins.qexp.runtime.ready import group_members

    shutil.rmtree(group_members._group_root(cfg, "exp"))

    result = verify_integrity(cfg, max_work_items=64)
    assert result["complete"] is False
    result = verify_integrity(cfg, max_work_items=64)

    assert "group_ready_members_inconsistent" in {item["code"] for item in result["issues"]}
    assert group_ready_members_state(cfg) == "degraded"


def test_verify_reports_bounded_progress(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")

    first = verify_integrity(cfg, max_work_items=1)

    assert first["complete"] is False
    assert first["healthy"] is False
    assert first["group_ready_members"]["verification"]["state"] == "building"


def test_completed_audit_starts_a_new_audit_without_degrading_projection(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")

    result = verify_integrity(cfg, max_work_items=64)
    while not result["complete"]:
        result = verify_integrity(cfg, max_work_items=64)
    first_audit = result["group_ready_members"]["verification"]["audit_id"]
    assert result["healthy"] is True
    assert group_ready_members_state(cfg) == "active"

    result = verify_integrity(cfg, max_work_items=1)
    verification = result["group_ready_members"]["verification"]
    assert verification["state"] == "building"
    assert verification["audit_id"] != first_audit
    assert "group_ready_members_inconsistent" not in {item["code"] for item in result["issues"]}
    assert group_ready_members_state(cfg) == "active"

    while not result["complete"]:
        result = verify_integrity(cfg, max_work_items=64)
    assert result["healthy"] is True
    assert group_ready_members_state(cfg) == "active"
    second_audit = result["group_ready_members"]["verification"]["audit_id"]

    from qqtools.plugins.qexp.runtime.ready.group_members_rebuild import cleanup_group_ready_member_archives

    builds = shared_paths(cfg.shared_root)["ready_group_members"] / "builds"
    for _ in range(20):
        cleanup_group_ready_member_archives(cfg, max_work_items=1)
        if not (builds / first_audit).exists() and not (builds / second_audit).exists():
            break
    assert not (builds / first_audit).exists()
    assert not (builds / second_audit).exists()


def test_clean_reclaims_parked_member_archive_in_bounded_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    mark_group_ready_members_degraded(cfg, "test")

    repair_metadata(cfg, max_work_items=1)
    state = read_json(shared_paths(cfg.shared_root)["ready_group_members"] / "state.json")["group_ready_members"]
    assert state["archive_count"] == 1

    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    monkeypatch.setattr(
        group_members_rebuild.os,
        "walk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("archive cleanup must not recurse")),
        raising=False,
    )

    result = clean(cfg, dry_run=False, max_work_items=1)
    while result["group_ready_member_archive_cleanup"]["state"] == "building":
        result = clean(cfg, dry_run=False, max_work_items=1)

    assert result["group_ready_member_archive_cleanup"]["archive_count"] == 0


def test_concurrent_archive_cleaners_share_one_owner_without_errors(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    mark_group_ready_members_degraded(cfg, "test")
    repair_metadata(cfg, max_work_items=1)

    from qqtools.plugins.qexp.runtime.ready.group_members_rebuild import cleanup_group_ready_member_archives

    barrier = Barrier(2)
    errors: list[Exception] = []

    def run_cleaner() -> None:
        try:
            barrier.wait()
            cleanup_group_ready_member_archives(cfg, max_work_items=1)
        except Exception as exc:
            errors.append(exc)

    first = Thread(target=run_cleaner)
    second = Thread(target=run_cleaner)
    first.start()
    second.start()
    first.join()
    second.join()
    assert errors == []

    result = cleanup_group_ready_member_archives(cfg, max_work_items=1)
    while result["state"] == "building":
        result = cleanup_group_ready_member_archives(cfg, max_work_items=1)
    assert result["archive_count"] == 0


def test_member_audit_does_not_start_a_member_page_after_spending_the_directory_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    cursor = {
        "group_page": 0,
        "group_offset": 0,
        "group_page_count": 0,
        "group_name": "exp",
        "directory_page": 0,
        "directory_offset": 0,
        "directory_pages_seen": 0,
        "directory_page_count": 1,
        "member_page": None,
        "entry_offset": 0,
        "seen_count": 0,
        "seen_digest": "0" * 64,
        "membership_revision": 1,
    }

    monkeypatch.setattr(
        group_members_rebuild,
        "load_group_ready_member_page",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("member page exceeded N=1")),
    )
    _cursor, processed, complete = group_members_rebuild._advance_member_audit(cfg, "unused", cursor, 1)

    assert processed == 1
    assert complete is False


def test_degraded_projection_blocks_group_worker_truth_mutation(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    mark_group_ready_members_degraded(cfg, "test")

    with pytest.raises(RuntimeError, match="projection is degraded"):
        change_worker(cfg, "exp", "gpu-2", "add", role="primary")

    workers = read_json(cfg.shared_root / "groups" / "exp.json")["group"]["worker_set"]
    assert "gpu-2" not in workers


@pytest.mark.parametrize("failure_point", ["locator", "partition", "catalog", "group", "global"])
def test_member_publication_write_failure_persists_degraded_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    from qqtools.plugins.qexp.runtime.ready import group_members

    original_atomic_replace = group_members.atomic_replace
    has_failed = False

    def fail_publication_write(path, value):
        nonlocal has_failed
        path = Path(path)
        should_fail = {
            "locator": path.parent.name == "locators",
            "partition": path.parent.name == "partitions",
            "catalog": path.parent.name == "catalog",
            "group": path.name == "state.json" and path.parent.name != "group-members",
            "global": path == shared_paths(cfg.shared_root)["ready_group_members"] / "state.json",
        }[failure_point]
        if should_fail and not has_failed:
            has_failed = True
            raise OSError(f"injected {failure_point} publication failure")
        return original_atomic_replace(path, value)

    monkeypatch.setattr(group_members, "atomic_replace", fail_publication_write)

    with pytest.raises(RuntimeError, match="publication failed"):
        submit(cfg, ["echo", "one"], task_id="member-task", group="exp")

    assert has_failed
    assert group_ready_members_state(cfg) == "degraded"


@pytest.mark.parametrize("failure_point", ["locator", "partition", "catalog", "group", "global"])
def test_member_retirement_write_failure_persists_degraded_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    first = submit(cfg, ["echo", "one"], task_id="member-one", group="exp")
    submit(cfg, ["echo", "two"], task_id="member-two", group="exp")
    from qqtools.plugins.qexp.runtime.ready import group_members

    original_atomic_replace = group_members.atomic_replace
    has_failed = False

    def fail_retirement_write(path, value):
        nonlocal has_failed
        path = Path(path)
        should_fail = {
            "locator": path.parent.name == "locators",
            "partition": path.parent.name == "partitions",
            "catalog": path.parent.name == "catalog",
            "group": path.name == "state.json" and path.parent.name != "group-members",
            "global": path == shared_paths(cfg.shared_root)["ready_group_members"] / "state.json",
        }[failure_point]
        if should_fail and not has_failed:
            has_failed = True
            raise OSError(f"injected {failure_point} retirement failure")
        return original_atomic_replace(path, value)

    monkeypatch.setattr(group_members, "atomic_replace", fail_retirement_write)

    with pytest.raises(RuntimeError, match="retirement is disabled"):
        retire_group_ready_member(cfg, "exp", first.task_id, first.ready_generation)

    assert has_failed
    assert group_ready_members_state(cfg) == "degraded"


def test_stale_member_revision_update_cannot_reopen_degraded_gate(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    from qqtools.plugins.qexp.runtime.ready import group_members

    state, catalog, partition = group_members._load_group(cfg, "exp")
    mark_group_ready_members_degraded(cfg, "concurrent_projection_failure")

    with pytest.raises(RuntimeError, match="projection is degraded"):
        group_members._write_group(cfg, "exp", state, catalog, partition)

    assert group_ready_members_state(cfg) == "degraded"


def test_concurrent_group_projection_updates_do_not_lose_the_global_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "first")
    create_group(cfg, "second")
    from qqtools.plugins.qexp.runtime.ready import group_members

    original_write_group = group_members._write_group
    barrier = Barrier(2)
    failures: list[BaseException] = []

    def synchronized_write_group(*args, **kwargs):
        barrier.wait(timeout=10)
        return original_write_group(*args, **kwargs)

    monkeypatch.setattr(group_members, "_write_group", synchronized_write_group)

    def write_group(group_name: str) -> None:
        try:
            state, catalog, partition = group_members._load_group(cfg, group_name)
            group_members._write_group(cfg, group_name, state, catalog, partition)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    first = Thread(target=write_group, args=("first",))
    second = Thread(target=write_group, args=("second",))
    first.start()
    second.start()
    first.join(timeout=15)
    second.join(timeout=15)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == [], [f"{exc!r}: {exc.__cause__!r}" for exc in failures]
    assert group_members.read_group_ready_members_state(cfg)["revision"] == 2


def test_retirement_uses_its_exact_locator_without_group_catalog_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "one"], task_id="member-task", group="exp")
    from qqtools.plugins.qexp.runtime.ready import group_members

    def fail_catalog_scan(*_args, **_kwargs):
        raise AssertionError("retirement must not scan all Group member pages")

    monkeypatch.setattr(group_members, "_read_entries", fail_catalog_scan)

    assert retire_group_ready_member(cfg, "exp", task.task_id, task.ready_generation) is True
    assert group_ready_members_state(cfg) == "active"


def test_first_upgrade_resume_captures_only_its_declared_io_budget(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for index in range(3):
        submit(cfg, ["echo", str(index)], task_id=f"member-{index}", group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()

    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )
    first = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )

    assert first["phase"] == "building"
    watermark = first["projection"]["build"]["watermark"]
    assert watermark["is_complete"] is False
    assert watermark["capture"]["task_count"] == 1
    assert not (
        cfg.shared_root
        / "indexes"
        / "ready"
        / "group-members"
        / "builds"
        / first["projection"]["build"]["build_id"]
        / "watermark"
    ).exists()


def test_upgrade_capture_commit_and_activation_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    group_members_rebuild.begin_group_ready_members_build(cfg)
    original_atomic_replace = group_members_rebuild.atomic_replace
    has_failed = False

    def fail_capture_commit(path, value):
        nonlocal has_failed
        record = value.get("group_ready_members") if isinstance(value, dict) else None
        if (
            path == shared_paths(cfg.shared_root)["ready_group_members"] / "state.json"
            and isinstance(record, dict)
            and record.get("state") == "building"
            and not has_failed
        ):
            has_failed = True
            raise OSError("injected watermark commit failure")
        return original_atomic_replace(path, value)

    monkeypatch.setattr(group_members_rebuild, "atomic_replace", fail_capture_commit)
    assert group_members_rebuild.advance_group_ready_members_build(cfg, max_tasks=1)["state"] == "degraded"
    assert has_failed

    cfg = init_shared_root(
        tmp_path / "second" / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "runtime-second",
    )
    create_group(cfg, "exp")
    submit(cfg, ["echo", "audit"], task_id="audit-task", group="exp")
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    group_members_rebuild.begin_group_ready_members_build(cfg)
    assert group_members_rebuild.advance_group_ready_members_build(cfg, max_tasks=1)["state"] == "building"
    monkeypatch.setattr(
        group_members_rebuild,
        "_audit_task_member",
        lambda *_args: (_ for _ in ()).throw(OSError("injected activation audit failure")),
    )
    for _ in range(16):
        result = group_members_rebuild.advance_group_ready_members_build(cfg, max_tasks=1)
        if result["state"] == "degraded":
            break
    assert result["state"] == "degraded"


def test_doctor_repair_crash_preserves_the_degraded_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    mark_group_ready_members_degraded(cfg, "injected projection damage")
    original_atomic_replace = group_members_rebuild.atomic_replace

    def fail_repair_start(path, value):
        record = value.get("group_ready_members") if isinstance(value, dict) else None
        if (
            path == shared_paths(cfg.shared_root)["ready_group_members"] / "state.json"
            and isinstance(record, dict)
            and record.get("state") == "building"
        ):
            raise OSError("injected doctor repair start failure")
        return original_atomic_replace(path, value)

    monkeypatch.setattr(group_members_rebuild, "atomic_replace", fail_repair_start)

    with pytest.raises(OSError, match="doctor repair start failure"):
        group_members_rebuild.repair_group_ready_members(cfg, max_tasks=1)

    assert group_ready_members_state(cfg) == "degraded"


def test_upgrade_dual_publishes_tasks_created_after_the_capture_cursor(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "before"], task_id="before-upgrade", group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()

    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )
    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    assert state["phase"] == "building"
    submit(cfg, ["echo", "during"], task_id="during-upgrade", group="exp")
    while state["phase"] == "building":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )

    assert state["phase"] == "completed"
    assert is_schema_narrow_protocol_active(cfg)
    assert {entry["task_id"] for entry in read_group_ready_members(cfg, "exp")} == {
        "before-upgrade",
        "during-upgrade",
    }


def test_resume_consumes_durable_watermark_without_reenumerating_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for task_id in ("member-one", "member-two"):
        submit(cfg, ["echo", task_id], task_id=task_id, group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()

    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )
    first = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    assert first["phase"] == "building"
    capture = first["projection"]["build"]["watermark"]["capture"]
    assert capture["offset"] > 0

    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    original_seekdir = group_members_rebuild._LIBC.seekdir
    resumed_offsets: list[int] = []

    def track_capture_resume(directory_handle, offset):
        resumed_offsets.append(offset)
        original_seekdir(directory_handle, offset)

    monkeypatch.setattr(group_members_rebuild._LIBC, "seekdir", track_capture_resume)
    completed = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    while completed["phase"] == "building":
        completed = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )

    assert completed["phase"] == "completed"
    assert capture["offset"] in resumed_offsets


def test_upgrade_treats_a_normally_cleaned_watermark_task_as_a_tombstone(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    task = submit(cfg, ["echo", "clean"], task_id="cleaned-watermark", group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(cfg, activation_id=session["activation_id"], machine_name=cfg.machine_name)
    resume_group_ready_members_upgrade(cfg, activation_id=session["activation_id"], max_tasks=1)

    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert fail_attempt(cfg, task.task_id, attempt.attempt_id, attempt.current_fencing_token, "test")
    assert clean(cfg, task_id=task.task_id)["removed"]

    result = session
    for _ in range(32):
        result = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
        if result["phase"] != "building":
            break
    assert result["phase"] == "completed"
    assert group_ready_members_state(cfg) == "active"


def test_primary_rebuild_crash_before_member_activation_resumes_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "crash"], task_id="crash-window", group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(cfg, activation_id=session["activation_id"], machine_name=cfg.machine_name)

    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    while state["projection"]["build"]["phase"] != "primary-rebuild":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    from qqtools.plugins.qexp.runtime.ready import group_members_rebuild

    original_commit = group_members_rebuild._commit_build_record

    def crash_after_primary(*args, **kwargs):
        if kwargs.get("should_activate"):
            raise KeyboardInterrupt("simulated process exit")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(group_members_rebuild, "_commit_build_record", crash_after_primary)
    with pytest.raises(KeyboardInterrupt, match="simulated process exit"):
        resume_group_ready_members_upgrade(cfg, activation_id=session["activation_id"], max_tasks=1)
    monkeypatch.setattr(group_members_rebuild, "_commit_build_record", original_commit)

    result = resume_group_ready_members_upgrade(cfg, activation_id=session["activation_id"], max_tasks=1)
    assert result["phase"] == "completed"
    assert group_ready_members_state(cfg) == "active"


def test_primary_rebuild_dual_writes_ready_tasks_created_after_its_watermark(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for task_id in ("old-one", "old-two"):
        submit(cfg, ["echo", task_id], task_id=task_id, group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )

    state = session
    while state.get("projection", {}).get("build", {}).get("phase") != "primary-rebuild":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    assert state["projection"]["build"]["phase"] == "primary-rebuild"

    concurrent = submit(cfg, ["echo", "during"], task_id="during-primary-rebuild", group="exp")
    from qqtools.plugins.qexp.runtime.ready.primary_candidates import candidate_path
    from qqtools.plugins.qexp.runtime.ready.routes import reference_for_generation

    reference = reference_for_generation(cfg, concurrent.task_id, concurrent.ready_generation)
    assert reference is not None
    primary_candidate_path = candidate_path(cfg, f"home.{cfg.machine_name}", reference.identity)
    assert primary_candidate_path.exists()

    while state["phase"] == "building":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )

    assert state["phase"] == "completed"
    assert primary_candidate_path.exists()


def test_final_task_watermark_starts_primary_rebuild_before_later_task_writes(
    tmp_path: Path,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    submit(cfg, ["echo", "old"], task_id="old-task", group="exp")
    primary = advance_ready_index_build(cfg)
    while primary["state"] == "building":
        primary = advance_ready_index_build(cfg)
    assert is_primary_ready_index_active(cfg)

    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )

    state = session
    while state.get("projection", {}).get("build", {}).get("phase") != "audit-task-capture":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=64,
    )
    assert state["projection"]["build"]["phase"] == "audit-tasks"

    concurrent = submit(cfg, ["echo", "late"], task_id="after-final-watermark", group="exp")
    from qqtools.plugins.qexp.runtime.ready.primary_candidates import candidate_path
    from qqtools.plugins.qexp.runtime.ready.routes import reference_for_generation

    reference = reference_for_generation(cfg, concurrent.task_id, concurrent.ready_generation)
    assert reference is not None
    primary_candidate_path = candidate_path(cfg, f"home.{cfg.machine_name}", reference.identity)
    assert primary_candidate_path.exists()

    while state["phase"] == "building":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )

    assert state["phase"] == "completed"
    assert primary_candidate_path.exists()


def test_primary_rebuild_parks_existing_routes_without_enumerating_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    routes = shared_paths(cfg.shared_root)["ready_primary"] / "routes"
    route = routes / "home.gpu-1"
    route.mkdir(parents=True)
    for index in range(128):
        atomic_replace(route / f"candidate-{index}.json", {"candidate": index})

    from qqtools.plugins.qexp.runtime.ready import primary_candidates

    original_scandir = primary_candidates.os.scandir

    def reject_candidate_enumeration(path):
        if Path(path) == routes or Path(path) == route:
            raise AssertionError("primary rebuild cutover must not enumerate candidates")
        return original_scandir(path)

    monkeypatch.setattr(primary_candidates.os, "scandir", reject_candidate_enumeration)
    primary_candidates.begin_primary_ready_index_rebuild(cfg, "bounded-cutover")
    monkeypatch.setattr(primary_candidates.os, "scandir", original_scandir)

    assert routes.is_dir()
    assert list(routes.iterdir()) == []
    replaced = shared_paths(cfg.shared_root)["ready_primary"] / "replaced-routes" / "bounded-cutover" / "home.gpu-1"
    assert len(list(replaced.iterdir())) == 128


def test_primary_rebuild_park_resumes_without_replacing_the_new_route_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    routes = shared_paths(cfg.shared_root)["ready_primary"] / "routes"
    old_route = routes / "home.gpu-1"
    old_route.mkdir(parents=True)
    atomic_replace(old_route / "old.json", {"candidate": "old"})

    from qqtools.plugins.qexp.runtime.ready import primary_candidates

    original_atomic_replace = primary_candidates.atomic_replace
    is_crash_injected = False

    def crash_before_cleared_commit(path, value):
        nonlocal is_crash_injected
        primary = value.get("primary_ready_index")
        if (
            not is_crash_injected
            and isinstance(primary, dict)
            and primary.get("state") == "rebuilding"
            and primary.get("cleared") is True
        ):
            is_crash_injected = True
            raise OSError("simulated crash after primary route park")
        return original_atomic_replace(path, value)

    monkeypatch.setattr(primary_candidates, "atomic_replace", crash_before_cleared_commit)
    with pytest.raises(OSError, match="simulated crash"):
        primary_candidates.begin_primary_ready_index_rebuild(cfg, "crash-cutover")

    new_route = routes / "home.gpu-2"
    new_route.mkdir(parents=True)
    atomic_replace(new_route / "new.json", {"candidate": "new"})
    primary_candidates.begin_primary_ready_index_rebuild(cfg, "crash-cutover")

    assert (new_route / "new.json").exists()
    replaced = (
        shared_paths(cfg.shared_root)["ready_primary"] / "replaced-routes" / "crash-cutover" / "home.gpu-1" / "old.json"
    )
    assert replaced.exists()


def test_primary_rebuild_accepts_group_role_updates_for_new_members(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    change_worker(cfg, "exp", "gpu-1", "set", role="borrow")
    for task_id in ("old-one", "old-two"):
        submit(cfg, ["echo", task_id], task_id=task_id, group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )

    state = session
    while state.get("projection", {}).get("build", {}).get("phase") != "primary-rebuild":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    concurrent = submit(cfg, ["echo", "during"], task_id="during-role-update", group="exp")
    from qqtools.plugins.qexp.runtime.ready.primary_candidates import candidate_path
    from qqtools.plugins.qexp.runtime.ready.routes import reference_for_generation

    reference = reference_for_generation(cfg, concurrent.task_id, concurrent.ready_generation)
    assert reference is not None
    primary_candidate_path = candidate_path(cfg, f"home.{cfg.machine_name}", reference.identity)
    assert not primary_candidate_path.exists()

    change_worker(cfg, "exp", "gpu-1", "set", role="primary")
    assert primary_candidate_path.exists()

    while state["phase"] == "building":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    assert state["phase"] == "completed"
    assert primary_candidate_path.exists()


def test_member_audit_restarts_a_group_after_a_legal_retirement(tmp_path: Path) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "exp")
    for task_id in ("audit-one", "audit-two"):
        submit(cfg, ["echo", task_id], task_id=task_id, group="exp")
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove("group-ready-members-v1")
    atomic_replace(schema_path, schema)
    (cfg.shared_root / "indexes" / "ready" / "group-members" / "state.json").unlink()
    session = start_group_ready_members_upgrade(cfg)
    attest_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        machine_name=cfg.machine_name,
    )

    state = session
    while state.get("projection", {}).get("build", {}).get("phase") != "audit-members":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    assert state["projection"]["build"]["audit_member_cursor"]["group_name"] == "exp"
    state = resume_group_ready_members_upgrade(
        cfg,
        activation_id=session["activation_id"],
        max_tasks=1,
    )
    attempt = claim_task(cfg, "audit-one", [0])
    assert attempt is not None
    from qqtools.plugins.qexp.runtime.ready import group_members

    while state["projection"]["build"]["audit_member_cursor"]["seen_count"] != 1:
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )
    cursor = state["projection"]["build"]["audit_member_cursor"]
    revision = read_json(group_members._group_state_path(cfg, "exp"))["group_ready_members"]
    assert cursor["seen_count"] == 1
    assert cursor["membership_revision"] < revision["membership_revision"]

    while state["phase"] == "building":
        state = resume_group_ready_members_upgrade(
            cfg,
            activation_id=session["activation_id"],
            max_tasks=1,
        )

    assert state["phase"] == "completed"
    assert group_ready_members_state(cfg) == "active"
    assert [entry["task_id"] for entry in read_group_ready_members(cfg, "exp")] == ["audit-two"]

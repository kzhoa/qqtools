"""Bounded archive replay preserves the terminal cleanup durability barrier."""

from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.runtime.claim_archive_scan import ClaimArchiveScan
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_archive_backlog_is_replayed_in_bounded_pages_before_clear(tmp_path):
    cfg = SimpleNamespace(shared_root=tmp_path / "shared")
    paths = shared_paths(cfg.shared_root)
    for token in range(41):
        atomic_replace(
            paths["claim_pending"] / "task" / f"{token}.json",
            {"claim_archive": {"task_id": "task", "fencing_token": token}},
        )
    scan = ClaimArchiveScan(cfg, "task")
    try:
        for turn in range(10):
            diagnostics = RuntimeDiagnostics()
            with activate_diagnostics(diagnostics):
                is_clear = scan.step(8)
            assert diagnostics.counters["store.read_json.calls"] <= 8
            assert diagnostics.counters["store.create_if_absent.calls"] <= 8
            if turn == 0:
                assert not is_clear
            if is_clear:
                break
        assert is_clear
        assert not list((paths["claim_pending"] / "task").glob("*.json"))
        for token in range(41):
            assert (
                read_json(paths["claim_archive"] / "task" / f"{token}.json")["claim_archive"]["fencing_token"] == token
            )
    finally:
        scan.close()


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ([], "Expected JSON object"),
        (None, "Expected JSON object"),
        ({"claim_archive": []}, "identity"),
        ({"claim_archive": {"task_id": "wrong", "fencing_token": 0}}, "identity"),
    ],
)
def test_malformed_pending_record_blocks_clear_without_starving_other_records(tmp_path, invalid, message):
    cfg = SimpleNamespace(shared_root=tmp_path / "shared")
    paths = shared_paths(cfg.shared_root)
    pending = paths["claim_pending"] / "task"
    atomic_replace(pending / "0.json", invalid)
    atomic_replace(pending / "1.json", {"claim_archive": {"task_id": "task", "fencing_token": 1}})
    scan = ClaimArchiveScan(cfg, "task")
    try:
        with pytest.raises(ValueError, match=message):
            scan.step(8)
        assert (pending / "0.json").exists()
        assert not (pending / "1.json").exists()
        assert (paths["claim_archive"] / "task" / "1.json").exists()
        with pytest.raises(ValueError, match=message):
            scan.step(8)
    finally:
        scan.close()


def test_empty_proof_rejects_directory_mutation_and_restart_replays(tmp_path):
    cfg = SimpleNamespace(shared_root=tmp_path / "shared")
    pending = shared_paths(cfg.shared_root)["claim_pending"] / "task"
    pending.mkdir(parents=True)
    (pending / "unrelated.txt").touch()
    scan = ClaimArchiveScan(cfg, "task")
    try:
        assert not scan.step(1)
        atomic_replace(pending / "1.json", {"claim_archive": {"task_id": "task", "fencing_token": 1}})
        assert not scan.step(1)
    finally:
        scan.close()
    (pending / "unrelated.txt").unlink()
    restarted = ClaimArchiveScan(cfg, "task")
    try:
        for _ in range(8):
            if restarted.step(1):
                break
        else:
            pytest.fail("restarted archive scan did not clear finite work")
        assert not (pending / "1.json").exists()
    finally:
        restarted.close()


def test_failed_archive_publication_retains_pending_record_and_retries(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime import claim_archive_scan

    cfg = SimpleNamespace(shared_root=tmp_path / "shared")
    paths = shared_paths(cfg.shared_root)
    pending = paths["claim_pending"] / "task" / "1.json"
    value = {"claim_archive": {"task_id": "task", "fencing_token": 1}}
    atomic_replace(pending, value)
    scan = ClaimArchiveScan(cfg, "task")
    original = claim_archive_scan.create_if_absent

    def unavailable(*_args):
        raise OSError("archive publication unavailable")

    try:
        monkeypatch.setattr(claim_archive_scan, "create_if_absent", unavailable)
        with pytest.raises(OSError, match="publication unavailable"):
            scan.step()
        assert read_json(pending) == value
        monkeypatch.setattr(claim_archive_scan, "create_if_absent", original)
        assert scan.step()
        assert not pending.exists()
        assert read_json(paths["claim_archive"] / "task" / "1.json") == value
    finally:
        scan.close()


def test_archive_discovery_reaches_valid_records_across_more_than_256_tasks(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.authority_work import AuthorityWork
    from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan

    cfg = SimpleNamespace(shared_root=tmp_path / "shared", runtime_root=tmp_path / "runtime")
    paths = shared_paths(cfg.shared_root)
    task_ids = [f"task-{number}" for number in range(257)]
    for task_id in task_ids:
        for token in range(25):
            atomic_replace(
                paths["claim_pending"] / task_id / f"{token}.json",
                {"claim_archive": {"task_id": "wrong-task", "fencing_token": token}},
            )
        atomic_replace(
            paths["claim_pending"] / task_id / "25.json",
            {"claim_archive": {"task_id": task_id, "fencing_token": 25}},
        )
    failures = []
    visited = [0]
    original_take = EvidenceScan.take

    def measured_take(scan, limit):
        page = original_take(scan, limit)
        visited[0] += page.entries_visited
        return page

    monkeypatch.setattr(EvidenceScan, "take", measured_take)
    work = AuthorityWork(
        SimpleNamespace(
            cfg=cfg,
            _materialize_unverified_intent=lambda _path: None,
            _record_diagnostic=lambda *_args: failures.append(_args),
        )
    )
    try:
        for _ in range(6):
            for task_id in task_ids:
                diagnostics = RuntimeDiagnostics()
                before = visited[0]
                with activate_diagnostics(diagnostics):
                    try:
                        assert not work.reconcile_archives(task_id)
                    except ValueError as exc:
                        assert "identity" in str(exc)
                assert diagnostics.counters["store.read_json.calls"] <= 8
                assert diagnostics.counters["store.create_if_absent.calls"] <= 8
                assert visited[0] - before <= 8
        assert work._archives.cursor_count <= 2
        assert failures
        for task_id in task_ids:
            assert (paths["claim_archive"] / task_id / "25.json").exists()
            assert len(list((paths["claim_pending"] / task_id).glob("*.json"))) == 25
    finally:
        work.close()


def test_archive_discovery_rotates_after_unreadable_task_directory(tmp_path, monkeypatch):
    from qqtools.plugins.qexp.runtime.claim_archive_scan import ClaimArchiveDiscovery

    cfg = SimpleNamespace(shared_root=tmp_path / "shared")
    paths = shared_paths(cfg.shared_root)
    for task_id in ("unreadable", "healthy"):
        atomic_replace(
            paths["claim_pending"] / task_id / "1.json",
            {"claim_archive": {"task_id": task_id, "fencing_token": 1}},
        )
    from qqtools.plugins.qexp.runtime.authority_scan import EvidenceScan

    original = EvidenceScan.take

    def take(scan, limit):
        if scan.directory == paths["claim_pending"] / "unreadable":
            raise OSError("injected unreadable Task directory")
        return original(scan, limit)

    monkeypatch.setattr(EvidenceScan, "take", take)
    discovery = ClaimArchiveDiscovery(cfg)
    try:
        for _ in range(12):
            try:
                discovery.step()
            except OSError as exc:
                assert "unreadable Task" in str(exc)
        assert (paths["claim_archive"] / "healthy" / "1.json").exists()
        assert (paths["claim_pending"] / "unreadable" / "1.json").exists()
    finally:
        discovery.close()


def test_project_archive_discovery_resumes_after_restart_and_new_task(tmp_path):
    from qqtools.plugins.qexp.runtime.claim_archive_scan import ClaimArchiveDiscovery, is_pending_archive_clear

    cfg = SimpleNamespace(shared_root=tmp_path / "shared")
    paths = shared_paths(cfg.shared_root)
    for token in range(11):
        atomic_replace(
            paths["claim_pending"] / "existing" / f"{token}.json",
            {"claim_archive": {"task_id": "existing", "fencing_token": token}},
        )
    discovery = ClaimArchiveDiscovery(cfg)
    try:
        for _ in range(3):
            discovery.step(1)
        assert not is_pending_archive_clear(cfg, "existing")
    finally:
        discovery.close()
    atomic_replace(
        paths["claim_pending"] / "arrived" / "1.json",
        {"claim_archive": {"task_id": "arrived", "fencing_token": 1}},
    )
    restarted = ClaimArchiveDiscovery(cfg)
    try:
        for _ in range(40):
            restarted.step(1)
        assert is_pending_archive_clear(cfg, "existing")
        assert is_pending_archive_clear(cfg, "arrived")
        assert len(list((paths["claim_archive"] / "existing").glob("*.json"))) == 11
        assert (paths["claim_archive"] / "arrived" / "1.json").exists()
    finally:
        restarted.close()

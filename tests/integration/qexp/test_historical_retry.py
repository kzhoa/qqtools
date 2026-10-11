"""Cancelled retry rejection preserves authoritative history and queue state."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.task import retry
from qqtools.plugins.qexp.runtime.historical_retry import HistoricalRetryError
from qqtools.plugins.qexp.runtime.paths import attempt_path, shared_paths
from qqtools.plugins.qexp.runtime.records import AttemptRecord
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task, save_task
from qqtools.plugins.qexp.scheduler import claim_task

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _cancelled_case(tmp_path):
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    task = submit(cfg, ["echo", "frozen"])
    attempt = claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    path = attempt_path(cfg.shared_root, task.task_id, attempt.attempt_number)
    stored = AttemptRecord.from_dict(read_json(path))
    stored.phase = "cancelled"
    stored.result.update(exit_code=None, category="cancelled", reason="terminated_by_agent")
    stored.termination.update(result="terminated")
    atomic_replace(path, stored.to_dict())
    task = load_task(cfg, task.task_id)
    task.state = {"projection": "cancelled", "reason": "terminated_by_agent"}
    task.claim_control["active_claim"] = None
    save_task(cfg, task)
    return cfg, task, path


def test_concurrent_historical_retry_rejects_without_successor(tmp_path: Path):
    cfg, task, path = _cancelled_case(tmp_path)
    before_task, before_attempt = load_task(cfg, task.task_id).to_dict(), path.read_bytes()
    before_files = sorted(path.parent.glob("*.json"))

    def attempt_retry(_):
        with pytest.raises(HistoricalRetryError) as caught:
            retry(cfg, task.task_id)
        return caught.value.reason

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(attempt_retry, range(2))) == ["unsupported_evidence"] * 2
    assert load_task(cfg, task.task_id).to_dict() == before_task
    assert path.read_bytes() == before_attempt
    assert sorted(path.parent.glob("*.json")) == before_files


@pytest.mark.parametrize("damage", ["missing", "malformed", "wrong_identity", "injected_receipt"])
def test_untrusted_or_missing_evidence_never_queues_execution(tmp_path: Path, damage):
    cfg, task, path = _cancelled_case(tmp_path)
    expected = "termination_evidence_conflict"
    if damage == "missing":
        path.unlink()
        expected = "termination_recovery_pending"
    elif damage == "malformed":
        path.write_text("{invalid", encoding="utf-8")
    elif damage == "wrong_identity":
        value = read_json(path)
        value["attempt"]["attempt_id"] = "other-attempt"
        atomic_replace(path, value)
    else:
        # A previously proposed receipt is not a cause/intent authority.
        receipt = cfg.shared_root / "indexes" / "termination-receipts-v1" / "forged.json"
        atomic_replace(receipt, {"state": "published", "cause": "holder_safe_deadline_elapsed"})
        expected = "unsupported_evidence"
    before = load_task(cfg, task.task_id).to_dict()
    with pytest.raises(HistoricalRetryError) as caught:
        retry(cfg, task.task_id)
    assert caught.value.reason == expected
    assert load_task(cfg, task.task_id).to_dict() == before
    assert len(list(shared_paths(cfg.shared_root)["attempts"].rglob("*.json"))) <= 1


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_cli_rejection_names_attempt_and_inspection_command(tmp_path: Path, monkeypatch, capsys, output_format):
    cfg, task, path = _cancelled_case(tmp_path)
    machine_root = tmp_path / "machine-runtime"
    MachineRuntime(machine_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    monkeypatch.setattr("qqtools.plugins.qexp.cli.project_handlers.ensure_local_agent_active", lambda *a, **k: False)
    before = path.read_bytes()
    assert (
        main(
            [
                "--project",
                str(cfg.shared_root),
                "--machine",
                cfg.machine_name,
                "--runtime-root",
                str(cfg.runtime_root),
                "--machine-runtime-root",
                str(machine_root),
                "task",
                "retry",
                task.task_id,
                "--format",
                output_format,
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    attempt_id = read_json(path)["attempt"]["attempt_id"]
    if output_format == "json":
        error = json.loads(captured.out)["error"]
        assert error["code"] == "unsupported_evidence"
        assert attempt_id in error["message"]
        assert "task show" in error["next_action"]
        assert captured.err == ""
    else:
        assert attempt_id in captured.err
        assert "task show" in captured.err
        assert "Traceback" not in captured.err
    assert path.read_bytes() == before
    assert load_task(cfg, task.task_id).state["projection"] == "cancelled"

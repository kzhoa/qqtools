from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest
from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.commands.task import offer
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.tasks import load_task
from tests.support.qexp.architecture import SingleHostMachineLab

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.machine_lab
def test_machine_lab_participants_have_independent_local_identity(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-identity")
    first = lab.start("first")
    second = lab.start("second")
    try:
        first_identity = first.request("identity")
        second_identity = second.request("identity")
        assert first_identity["pid"] != second_identity["pid"]
        assert first_identity["tmpdir"] != second_identity["tmpdir"]
        assert first_identity["runtime_root"] != second_identity["runtime_root"]
        assert lab.shared_root.exists()
        assert first.request("scheduler_authority")["acquired"] is True
        assert second.request("scheduler_authority")["acquired"] is True
    finally:
        lab.close()


@pytest.mark.machine_lab
def test_machine_lab_records_participant_lifecycle_in_its_resource_ledger(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-ledger")
    participant = lab.start("first")
    participant.request("identity")
    lab.close()
    records = [
        json.loads(path.read_text(encoding="utf-8"))["resource"]
        for path in participant.scope.resource_ledger_root.glob("*.json")
    ]
    kinds = {record["kind"] for record in records}
    assert {"participant-intent", "participant", "participant-exit"} <= kinds


@pytest.mark.machine_lab
def test_machine_lab_checkpoint_allows_participant_interleaving_and_restart(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-checkpoint")
    first = lab.start("first")
    second = lab.start("second")
    try:
        assert first.request("checkpoint")["checkpoint"] == "reached"
        second_identity = second.request("identity")
        assert first.request("continue")["checkpoint"] == "continued"
        first.kill()
        restarted = lab.restart("first")
        restarted_identity = restarted.request("identity")
        assert restarted_identity["pid"] != first.process.pid
        assert restarted_identity["tmpdir"] != second_identity["tmpdir"]
        assert [event.kind for event in lab.trace.events].count("participant.command") == 4
    finally:
        lab.close()


@pytest.mark.machine_lab
def test_machine_lab_two_participants_produce_one_real_task_claim(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-claim")
    cfg = init_shared_root(lab.shared_root, "gpu-1", runtime_root=tmp_path / "bootstrap-runtime")
    create_group(cfg, "workers")
    change_worker(cfg, "workers", "gpu-2", "add")
    task = submit(cfg, ["echo", "claim"], group="workers", sharing_mode="spillover")
    offer(cfg, task.task_id)
    first = lab.start("first")
    second = lab.start("second")

    def claim(participant, machine_name: str) -> dict[str, object]:
        return participant.request(
            "claim_task",
            payload={"machine_name": machine_name, "task_id": task.task_id, "gpu_ids": [0]},
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda item: claim(*item), ((first, "gpu-1"), (second, "gpu-2"))))
        claims = [result for result in results if result["claimed"] is True]
        assert len(claims) == 1
        assert isinstance(claims[0]["attempt_id"], str)
        assert claims[0]["fencing_token"] == 1
    finally:
        lab.close()


@pytest.mark.machine_lab
def test_machine_lab_cas_race_preserves_the_single_winning_value(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-cas")
    record = lab.shared_root / "cas-record.json"
    atomic_replace(record, {"meta": {"revision": 0}, "value": "initial"})
    first = lab.start("first")
    second = lab.start("second")

    def update(participant, value: str) -> dict[str, object]:
        return participant.request(
            "cas_update",
            payload={"record_name": record.name, "expected_revision": 0, "value": value},
        )

    try:
        assert first.request("read_revision", payload={"record_name": record.name}) == {"revision": 0}
        assert second.request("read_revision", payload={"record_name": record.name}) == {"revision": 0}
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda item: update(*item), ((first, "first"), (second, "second"))))
        assert sorted(result["committed"] for result in results) == [False, True]
        stored = read_json(record)
        assert stored["meta"]["revision"] == 1
        assert stored["value"] in {"first", "second"}
    finally:
        lab.close()


@pytest.mark.machine_lab
def test_machine_lab_cancel_and_launch_gate_have_one_lock_order(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-cancel-launch")
    cfg = init_shared_root(lab.shared_root, "gpu-1", runtime_root=tmp_path / "bootstrap-runtime")
    create_group(cfg, "workers")
    change_worker(cfg, "workers", "gpu-2", "add")
    task = submit(cfg, ["echo", "race"], group="workers", sharing_mode="spillover")
    offer(cfg, task.task_id)
    first = lab.start("first")
    second = lab.start("second")
    try:
        claim = first.request("claim_task", payload={"machine_name": "gpu-1", "task_id": task.task_id, "gpu_ids": [0]})
        assert claim["claimed"] is True
        assert isinstance(claim["attempt_id"], str)
        assert claim["fencing_token"] == 1
        with ThreadPoolExecutor(max_workers=2) as executor:
            launch = executor.submit(
                first.request,
                "authorize_launch",
                payload={
                    "machine_name": "gpu-1",
                    "task_id": task.task_id,
                    "attempt_id": claim["attempt_id"],
                    "fencing_token": claim["fencing_token"],
                },
            )
            cancellation = executor.submit(
                second.request, "cancel_task", payload={"machine_name": "gpu-2", "task_id": task.task_id}
            )
            launch_result = launch.result()
            cancellation.result()
        state = load_task(cfg, task.task_id).state["projection"]
        assert launch_result["authorized"] == (state == "running")
        assert state in {"running", "cancelled"}
    finally:
        lab.close()


@pytest.mark.machine_lab
def test_machine_lab_rejects_a_stale_fencing_token_after_a_successor_claim(tmp_path: Path) -> None:
    lab = SingleHostMachineLab(tmp_path, "machine-lab-fencing")
    cfg = init_shared_root(lab.shared_root, "gpu-1", runtime_root=tmp_path / "bootstrap-runtime")
    create_group(cfg, "workers")
    change_worker(cfg, "workers", "gpu-2", "add")
    task = submit(cfg, ["echo", "fencing"], group="workers", sharing_mode="spillover")
    offer(cfg, task.task_id)
    first = lab.start("first")
    second = lab.start("second")
    try:
        original = first.request(
            "claim_task", payload={"machine_name": "gpu-1", "task_id": task.task_id, "gpu_ids": [0]}
        )
        assert original["claimed"] is True
        assert isinstance(original["attempt_id"], str)
        assert original["fencing_token"] == 1
        assert (
            first.request(
                "fail_attempt",
                payload={
                    "machine_name": "gpu-1",
                    "task_id": task.task_id,
                    "attempt_id": original["attempt_id"],
                    "fencing_token": original["fencing_token"],
                },
            )["failed"]
            is True
        )
        assert (
            second.request("retry_task", payload={"machine_name": "gpu-2", "task_id": task.task_id})["state"]
            == "queued"
        )
        successor = first.request(
            "claim_task", payload={"machine_name": "gpu-1", "task_id": task.task_id, "gpu_ids": [0]}
        )
        assert successor["claimed"] is True
        assert successor["fencing_token"] == 2
        stale = first.request(
            "authorize_launch",
            payload={
                "machine_name": "gpu-1",
                "task_id": task.task_id,
                "attempt_id": original["attempt_id"],
                "fencing_token": original["fencing_token"],
            },
        )
        assert stale == {"authorized": False}
        assert load_task(cfg, task.task_id).claim_control["active_claim"]["fencing_token"] == 2
    finally:
        lab.close()

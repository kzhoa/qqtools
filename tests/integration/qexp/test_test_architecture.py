from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import select
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import change_worker, create_group
from qqtools.plugins.qexp.commands.task import offer
from qqtools.plugins.qexp.runtime.tasks import load_task
from tests.qexp_test_support import TestResourceScope
from tests.qexp_architecture import SingleHostMachineLab


pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.qexp_fast
def test_resource_scope_isolates_child_environment(qexp_resource_scope: TestResourceScope) -> None:
    environment = qexp_resource_scope.child_environment({"PATH": "test-path"})

    assert environment["PATH"] == "test-path"
    assert environment["TMPDIR"] == str(qexp_resource_scope.local_temp_root)
    assert environment["HOME"] == str(qexp_resource_scope.home_root)
    assert environment["QEXP_MACHINE_RUNTIME_ROOT"] == str(qexp_resource_scope.runtime_root)


@pytest.mark.qexp_fast
def test_resource_scope_records_test_owned_resources_and_cleanup_diagnostics(
    qexp_resource_scope: TestResourceScope,
) -> None:
    resource = qexp_resource_scope.record_resource("participant", {"pid": 1234})
    diagnostic = qexp_resource_scope.record_cleanup_diagnostic("timeout", {"pid": 1234})

    assert json.loads(resource.read_text(encoding="utf-8")) == {
        "resource": {"identity": {"pid": 1234}, "kind": "participant"}
    }
    assert json.loads(diagnostic.read_text(encoding="utf-8")) == {
        "diagnostic": {"pid": 1234}
    }


@pytest.mark.qexp_fast
def test_machine_participants_use_distinct_frozen_authority_roots(tmp_path: Path) -> None:
    first_scope = TestResourceScope.create(tmp_path, "first")
    second_scope = TestResourceScope.create(tmp_path, "second")
    source_root = Path(__file__).parents[3] / "src"
    participant = """
import json
import sys
from pathlib import Path
from qqtools.plugins.qexp.machine_runtime import MachineRuntime

runtime = MachineRuntime(sys.argv[1])
result = Path(sys.argv[2])
with runtime.scheduler_authority(blocking=False) as acquired:
    result.write_text(json.dumps({"acquired": acquired}), encoding="utf-8")
    if acquired:
        sys.stdin.readline()
"""

    processes: list[subprocess.Popen[str]] = []
    result_paths = []
    for scope in (first_scope, second_scope):
        result_path = scope.root / "result.json"
        environment = scope.child_environment()
        environment["PYTHONPATH"] = str(source_root)
        processes.append(
            subprocess.Popen(
                [sys.executable, "-c", participant, str(scope.runtime_root), str(result_path)],
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
        result_paths.append(result_path)

    try:
        deadline = time.monotonic() + 2.0
        while any(not path.exists() for path in result_paths) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all(path.exists() for path in result_paths)
        acquired = [
            json.loads(path.read_text(encoding="utf-8"))["acquired"] for path in result_paths
        ]
        assert acquired == [True, True]
    finally:
        for process in processes:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=1.0)


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
def test_machine_lab_checkpoint_allows_participant_interleaving_and_restart(
    tmp_path: Path,
) -> None:
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
            results = list(
                executor.map(lambda item: claim(*item), ((first, "gpu-1"), (second, "gpu-2")))
            )

        claims = [result for result in results if result["claimed"] is True]
        assert len(claims) == 1
        assert isinstance(claims[0]["attempt_id"], str)
        assert claims[0]["fencing_token"] == 1
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
        claim = first.request(
            "claim_task",
            payload={"machine_name": "gpu-1", "task_id": task.task_id, "gpu_ids": [0]},
        )
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
                second.request,
                "cancel_task",
                payload={"machine_name": "gpu-2", "task_id": task.task_id},
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
            "claim_task",
            payload={"machine_name": "gpu-1", "task_id": task.task_id, "gpu_ids": [0]},
        )
        assert original["claimed"] is True
        assert isinstance(original["attempt_id"], str)
        assert original["fencing_token"] == 1
        assert first.request(
            "fail_attempt",
            payload={
                "machine_name": "gpu-1",
                "task_id": task.task_id,
                "attempt_id": original["attempt_id"],
                "fencing_token": original["fencing_token"],
            },
        )["failed"] is True
        assert second.request(
            "retry_task",
            payload={"machine_name": "gpu-2", "task_id": task.task_id},
        )["state"] == "queued"
        successor = first.request(
            "claim_task",
            payload={"machine_name": "gpu-1", "task_id": task.task_id, "gpu_ids": [0]},
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


@pytest.mark.host_exclusive
def test_host_exclusive_authority_allows_only_one_runtime_per_os_user(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[3] / "src"
    shared_tmpdir = tempfile.gettempdir()
    participant = """
import json
import sys
from qqtools.plugins.qexp.machine_runtime import MachineRuntime

runtime = MachineRuntime(sys.argv[1])
with runtime.scheduler_authority(blocking=False) as acquired:
    print(json.dumps({"acquired": acquired}), flush=True)
    if acquired:
        sys.stdin.readline()
"""

    def start(scope: TestResourceScope) -> subprocess.Popen[str]:
        environment = scope.child_environment()
        environment.update(
            {
                "PYTHONPATH": str(source_root),
                "TMPDIR": shared_tmpdir,
                "TMP": shared_tmpdir,
                "TEMP": shared_tmpdir,
            }
        )
        return subprocess.Popen(
            [sys.executable, "-c", participant, str(scope.runtime_root)],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def response(process: subprocess.Popen[str]) -> dict[str, object]:
        assert process.stdout is not None
        ready, _, _ = select.select([process.stdout], [], [], 2.0)
        if not ready:
            stderr = "participant is still running"
            if process.stderr is not None and process.poll() is not None:
                stderr = process.stderr.read()
            pytest.fail(f"authority participant did not respond within 2 seconds: {stderr}")
        line = process.stdout.readline()
        if not line:
            stderr = "stderr unavailable"
            if process.stderr is not None:
                stderr = process.stderr.read()
            pytest.fail(f"authority participant exited before responding: {stderr}")
        return json.loads(line)

    first = start(TestResourceScope.create(tmp_path, "first"))
    second = start(TestResourceScope.create(tmp_path, "second"))
    third: subprocess.Popen[str] | None = None
    try:
        assert response(first)["acquired"] is True
        assert response(second)["acquired"] is False

        assert first.stdin is not None
        first.stdin.close()
        assert first.wait(timeout=2.0) == 0

        third = start(TestResourceScope.create(tmp_path, "third"))
        assert response(third)["acquired"] is True
    finally:
        for process in (first, second, third):
            if process is None:
                continue
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=2.0)

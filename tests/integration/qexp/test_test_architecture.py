from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from tests.qexp_test_support import TestResourceScope
from tests.qexp_architecture import SingleHostMachineLab


pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_resource_scope_isolates_child_environment(qexp_resource_scope: TestResourceScope) -> None:
    environment = qexp_resource_scope.child_environment({"PATH": "test-path"})

    assert environment["PATH"] == "test-path"
    assert environment["TMPDIR"] == str(qexp_resource_scope.local_temp_root)
    assert environment["HOME"] == str(qexp_resource_scope.home_root)
    assert environment["QEXP_MACHINE_RUNTIME_ROOT"] == str(qexp_resource_scope.runtime_root)


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

from __future__ import annotations

import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from tests.helpers.qexp.resources import TestResourceScope

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def test_resource_scope_isolates_child_environment(qexp_resource_scope: TestResourceScope) -> None:
    environment = qexp_resource_scope.child_environment({"PATH": "test-path"})

    assert environment["PATH"] == "test-path"
    assert environment["TMPDIR"] == str(qexp_resource_scope.local_temp_root)
    assert environment["HOME"] == str(qexp_resource_scope.home_root)
    assert environment["QEXP_MACHINE_RUNTIME_ROOT"] == str(qexp_resource_scope.runtime_root)


def test_resource_scope_derives_tmux_root_from_selected_test_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_base = tmp_path / "fallback-test-base"
    monkeypatch.setenv("QQTOOLS_TEST_TMUX_BASE", str(selected_base))

    scope = TestResourceScope.create(tmp_path / "resources", "tmux-fallback")

    assert scope.tmux_root.parent == selected_base


def test_resource_scope_records_test_owned_resources_and_cleanup_diagnostics(
    qexp_resource_scope: TestResourceScope,
) -> None:
    resource = qexp_resource_scope.record_resource("participant-intent", {"pid": 1234})
    diagnostic = qexp_resource_scope.record_cleanup_diagnostic("timeout", {"pid": 1234})

    assert json.loads(resource.read_text(encoding="utf-8")) == {
        "resource": {"identity": {"pid": 1234}, "kind": "participant-intent"}
    }
    assert json.loads(diagnostic.read_text(encoding="utf-8")) == {
        "consumed": False,
        "diagnostic": {"pid": 1234},
    }
    assert any(
        "unconsumed cleanup diagnostic" in violation
        for violation in TestResourceScope.cleanup_violations(qexp_resource_scope.root)
    )
    qexp_resource_scope.consume_cleanup_diagnostic(diagnostic)


def test_resource_scope_reports_an_active_test_owned_tmux_socket(
    qexp_resource_scope: TestResourceScope,
) -> None:
    server = socket.socket(socket.AF_UNIX)
    try:
        server.bind(str(qexp_resource_scope.tmux_socket))
        server.listen()
        assert any(
            "tmux socket remains active" in violation
            for violation in TestResourceScope.cleanup_violations(qexp_resource_scope.root)
        )
    finally:
        server.close()


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
        acquired = [json.loads(path.read_text(encoding="utf-8"))["acquired"] for path in result_paths]
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


def test_isolated_authority_allows_only_one_runtime_at_a_time(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[3] / "src"
    shared_tmpdir = tmp_path / "shared-authority-tmp"
    shared_tmpdir.mkdir()
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
        process = subprocess.Popen(
            [sys.executable, "-c", participant, str(scope.runtime_root)],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        scope.record_resource(
            "participant",
            {"pid": process.pid, "process_group_id": process.pid},
        )
        return process

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

    first_scope = TestResourceScope.create(tmp_path, "first")
    second_scope = TestResourceScope.create(tmp_path, "second")
    third_scope = TestResourceScope.create(tmp_path, "third")
    processes: list[tuple[TestResourceScope, subprocess.Popen[str]]] = [(first_scope, start(first_scope))]
    try:
        first = processes[0][1]
        assert response(first)["acquired"] is True
        second = start(second_scope)
        processes.append((second_scope, second))
        assert response(second)["acquired"] is False

        assert first.stdin is not None
        first.stdin.close()
        assert first.wait(timeout=2.0) == 0

        third = start(third_scope)
        processes.append((third_scope, third))
        assert response(third)["acquired"] is True
    finally:
        for _scope, process in processes:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2.0)

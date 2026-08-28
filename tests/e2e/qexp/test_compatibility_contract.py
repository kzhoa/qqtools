from __future__ import annotations

import json
from pathlib import Path

from qexp_e2e import (
    TASK_TERMINAL_TIMEOUT_SECONDS,
    ensure_site_packages_import,
    is_machine_agent_running,
    jrun,
    make_env,
    run,
    stop_agent,
    wait_for,
)


def _common(shared_root: Path, runtime_root: Path, machine_runtime_root: Path) -> list[str]:
    return [
        "qexp",
        "--shared-root",
        str(shared_root),
        "--machine",
        "gpu-1",
        "--runtime-root",
        str(runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]


def _wait_for_terminal_task(
    common: list[str], task_id: str, env: dict[str, str]
) -> dict[str, object]:
    def is_done() -> bool:
        task = jrun([*common, "task", "show", task_id], env=env)
        return task["task"]["state"]["projection"] in {"succeeded", "failed", "cancelled"}

    wait_for(
        is_done,
        timeout=TASK_TERMINAL_TIMEOUT_SECONDS,
        label="protected workflow task completion",
    )
    return jrun([*common, "task", "show", task_id], env=env)


def _assert_current_project_is_registered(
    common: list[str], shared_root: Path, env: dict[str, str]
) -> None:
    projects = jrun([*common, "agent", "list-projects"], env=env)["projects"]
    assert any(project["shared_root"] == str(shared_root) for project in projects)


# Protected compatibility test: New project workflow
def test_new_project_activation_uses_registered_binding_without_add_project(tmp_path: Path) -> None:
    base = tmp_path / "protected-new-project-activation"
    shared_root = base / ".qexp"
    runtime_root = base / "project-runtime"
    machine_runtime_root = base / "machine-runtime"
    env = make_env(base)
    common = _common(shared_root, runtime_root, machine_runtime_root)

    try:
        run([*common, "init"], env=env)
        _assert_current_project_is_registered(common, shared_root, env)
        started = run([*common, "agent", "start", "--format=json"], env=env)
        status = json.loads(started.stdout)
        assert status["action"] == "started"
        assert is_machine_agent_running(common, env=env)

        task_id = run(
            [*common, "submit", "--", "python", "-c", "print('protected start')"], env=env
        ).stdout.strip()
        task = _wait_for_terminal_task(common, task_id, env)

        assert "site-packages" in ensure_site_packages_import()
        assert task["task"]["state"]["projection"] == "succeeded"
    finally:
        stop_agent(common, env=env)


# Protected compatibility test: New project submission workflow
def test_new_project_submit_activates_agent_without_manual_registration(tmp_path: Path) -> None:
    base = tmp_path / "protected-new-project-submit"
    shared_root = base / ".qexp"
    runtime_root = base / "project-runtime"
    machine_runtime_root = base / "machine-runtime"
    env = make_env(base)
    common = _common(shared_root, runtime_root, machine_runtime_root)

    try:
        run([*common, "init"], env=env)
        _assert_current_project_is_registered(common, shared_root, env)
        assert not is_machine_agent_running(common, env=env)

        task_id = run(
            [*common, "submit", "--", "python", "-c", "print('protected submit')"], env=env
        ).stdout.strip()
        task = _wait_for_terminal_task(common, task_id, env)

        assert "site-packages" in ensure_site_packages_import()
        assert is_machine_agent_running(common, env=env)
        assert task["task"]["state"]["projection"] == "succeeded"
    finally:
        stop_agent(common, env=env)


# Protected compatibility test: Legacy migration workflow
def test_legacy_metadata_requires_migration_without_touching_unrelated_resources(
    tmp_path: Path,
) -> None:
    base = tmp_path / "protected-legacy-migration"
    shared_root = base / "legacy-project" / ".qexp"
    runtime_root = base / "legacy-runtime"
    bootstrap_runtime_root = base / "bootstrap-machine-runtime"
    machine_runtime_root = base / "machine-runtime"
    env = make_env(base)
    bootstrap_common = _common(shared_root, runtime_root, bootstrap_runtime_root)
    common = _common(shared_root, runtime_root, machine_runtime_root)
    record_path = shared_root / "machines" / "gpu-1" / "machine.json"
    legacy_reservation_path = (
        runtime_root / "reservations" / "provisional" / "legacy-reservation.json"
    )
    legacy_evidence_path = runtime_root / "processes" / "legacy-attempt.json"
    unrelated_reservation_path = machine_runtime_root / "reservations" / "active" / "unrelated.json"

    try:
        run([*bootstrap_common, "init"], env=env)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["machine"].pop("agent_runtime")
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        original_legacy_record = record_path.read_bytes()
        legacy_reservation_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_reservation_path.write_text(
            json.dumps(
                {
                    "reservation": {
                        "reservation_id": "legacy-reservation",
                        "gpu_ids": [0],
                        "state": "provisional",
                        "task_id": "legacy-task",
                    }
                }
            ),
            encoding="utf-8",
        )
        legacy_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_evidence_path.write_text(
            json.dumps({"process": {"attempt_id": "legacy-attempt"}}), encoding="utf-8"
        )

        other_root = base / "other-project" / ".qexp"
        other_runtime_root = base / "other-runtime"
        run(
            [
                "qexp",
                "--shared-root",
                str(other_root),
                "--machine",
                "gpu-other",
                "--runtime-root",
                str(other_runtime_root),
                "--machine-runtime-root",
                str(machine_runtime_root),
                "init",
            ],
            env=env,
        )
        unrelated_reservation_path.parent.mkdir(parents=True, exist_ok=True)
        unrelated_reservation_path.write_text(
            json.dumps(
                {
                    "reservation": {
                        "reservation_id": "unrelated-reservation",
                        "project_id": "unrelated-project",
                        "gpu_ids": [1],
                        "state": "active",
                        "task_id": "unrelated-task",
                    }
                }
            ),
            encoding="utf-8",
        )
        original_unrelated_reservation = unrelated_reservation_path.read_bytes()

        rejected = run(
            [
                "qexp",
                "--shared-root",
                str(shared_root),
                "--machine",
                "gpu-2",
                "--runtime-root",
                str(runtime_root),
                "--machine-runtime-root",
                str(machine_runtime_root),
                "init",
            ],
            env=env,
            check=False,
        )
        assert rejected.returncode == 2
        assert "qexp agent migrate-project" in rejected.stderr
        assert record_path.read_bytes() == original_legacy_record
        assert not (shared_root / "machines" / "gpu-2" / "machine.json").exists()

        migrated = jrun([*common, "agent", "migrate-project"], env=env)
        assert migrated["action"] == "project_migrated"
        assert migrated["enabled"] is True
        assert record_path.read_bytes() != original_legacy_record
        migrated_record = json.loads(record_path.read_text(encoding="utf-8"))
        assert migrated_record["machine"]["agent_runtime"] == "machine"
        assert not legacy_reservation_path.exists()
        assert not legacy_evidence_path.exists()
        assert unrelated_reservation_path.read_bytes() == original_unrelated_reservation

        registry = json.loads((machine_runtime_root / "registry.json").read_text(encoding="utf-8"))
        bindings = registry["registry"]["bindings"]
        target = next(binding for binding in bindings if binding["shared_root"] == str(shared_root))
        assert target["enabled"] is True
        migrated_reservation = (
            machine_runtime_root / "reservations" / "provisional" / "legacy-reservation.json"
        )
        migrated_evidence = (
            machine_runtime_root
            / "projects"
            / target["project_id"]
            / "processes"
            / "legacy-attempt.json"
        )
        assert migrated_reservation.exists()
        assert migrated_evidence.exists()
        assert "site-packages" in ensure_site_packages_import()
    finally:
        stop_agent(common, env=env)

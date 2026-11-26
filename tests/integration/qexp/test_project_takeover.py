"""Exercise takeover discovery and registration observation through the CLI."""

import json
import shlex

import pytest

from qqtools.plugins.qexp.agent import takeover_intent, takeover_requests
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_takeover import (
    TakeoverUsageError,
    confirm_project_takeover,
    preview_project_takeover,
)
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.agent.takeover_intent import read_takeover_intent
from qqtools.plugins.qexp.agent.takeover_requests import accept_project_takeover, request_takeover_cancellation
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = pytest.mark.integration


@pytest.fixture
def enrolled(tmp_path):
    runtime = MachineRuntime(tmp_path / "successor runtime")
    previous = MachineRuntime(tmp_path / "previous runtime")
    initialize_machine(runtime, "global-name")
    initialize_machine(previous, "g2")
    project = tmp_path / "affected project"
    other = tmp_path / "other project"
    for root in (project, other):
        initialize_project(root)
        assert register_projects(runtime, [root], machine_name="g3")["projects"][0]["status"] == "registered"
    assert register_projects(previous, [project])["projects"][0]["status"] == "registered"
    return runtime, previous, project, other


def preview(runtime, project, target="g2", *, global_args=()):
    return main(
        [
            "--machine-runtime-root",
            str(runtime.root),
            *global_args,
            "project",
            "takeover",
            str(project),
            "--machine",
            target,
            "--plan",
            "--format=json",
        ]
    )


def snapshot(roots):
    return {
        str(path): (path.stat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
        for root in roots
        for path in (root, *root.rglob("*"))
    }


def test_expired_target_preview_and_status_discovery_are_read_only(enrolled, capsys):
    runtime, previous, project, other = enrolled
    binding = runtime.resolve_project_binding(project / ".qexp")
    registration_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    document = json.loads(registration_path.read_bytes())
    target = document["registration"]
    target["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(registration_path, document)
    roots = (runtime.root, previous.root, project, other)
    before = snapshot(roots)

    status_args = ["--machine-runtime-root", str(runtime.root), "--project", str(project), "status"]
    assert main([*status_args, "--format=json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["local_participation"]["state"] == "registered"
    assert main(status_args) == 0
    assert status["takeover_preview_help"] in capsys.readouterr().out
    help_args = shlex.split(status["takeover_preview_help"])
    assert help_args == [
        "qexp",
        "--machine-runtime-root",
        str(runtime.root),
        "project",
        "takeover",
        str(project),
        "--help",
    ]
    with pytest.raises(SystemExit) as result:
        main(help_args[1:])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    assert "--machine" in help_text and "--plan" in help_text

    assert preview(runtime, project) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "preview"
    assert result["project_id"] == binding.project_id
    assert result["runtime_id"] == binding.runtime_instance_id
    assert result["current_binding"] == {
        "machine_name": "g3",
        "registration_generation": binding.registration_generation,
        "enabled": True,
    }
    assert result["target_registration"] == {
        "machine_name": "g2",
        "state": "expired",
        "generation": target["generation"],
        "runtime_id": target["runtime_instance_id"],
        "runtime_root": target["runtime_root"],
        "eligibility_expires_at": target["eligibility_expires_at"],
    }
    assert result["blockers"] == []
    assert snapshot(roots) == before


@pytest.mark.parametrize(
    ("target", "expected_blocker"),
    [
        ("g2", "target_registration_eligible"),
        ("g3", "target_is_current_binding"),
        ("missing", "target_registration_missing"),
    ],
)
def test_registration_blockers_preserve_the_binding(enrolled, capsys, target, expected_blocker):
    runtime, previous, project, other = enrolled
    roots = (runtime.root, previous.root, project, other)
    before = snapshot(roots)

    assert preview(runtime, project, target) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "blocked"
    assert expected_blocker in {blocker["code"] for blocker in result["blockers"]}
    assert result["current_binding"]["machine_name"] == "g3"
    assert snapshot(roots) == before


@pytest.mark.parametrize("side", ["departing", "target"])
def test_malformed_registration_fails_closed(enrolled, capsys, side):
    runtime, previous, project, other = enrolled
    machine = "g3" if side == "departing" else "g2"
    registration_path = project / ".qexp" / "machines" / machine / "registration.json"
    registration_path.write_text('{"registration": false}', encoding="utf-8")
    roots = (runtime.root, previous.root, project, other)
    before = snapshot(roots)

    assert preview(runtime, project) == 1

    assert "error" in json.loads(capsys.readouterr().out)
    assert snapshot(roots) == before


def test_preview_does_not_initialize_an_absent_runtime(enrolled, capsys, tmp_path):
    _, _, project, _ = enrolled
    unused_runtime = MachineRuntime(tmp_path / "unused")
    assert preview(unused_runtime, project) == 1
    result = json.loads(capsys.readouterr().out)
    assert "uninitialized" in result["error"]["message"]
    assert not unused_runtime.root.exists()


def test_named_target_is_required_explicitly(enrolled, capsys):
    runtime, _, project, _ = enrolled
    assert (
        main(
            [
                "--machine-runtime-root",
                str(runtime.root),
                "--machine",
                "g2",
                "project",
                "takeover",
                str(project),
                "--plan",
                "--format=json",
            ]
        )
        == 2
    )
    assert "--machine" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_preview_allows_initialized_runtime_without_a_local_project_binding(enrolled, capsys, tmp_path):
    _, previous, project, other = enrolled
    successor = MachineRuntime(tmp_path / "fresh runtime")
    initialize_machine(successor, "replacement")
    registration_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    document = json.loads(registration_path.read_bytes())
    document["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(registration_path, document)
    roots = (successor.root, previous.root, project, other)
    before = snapshot(roots)

    assert preview(successor, project) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["current_binding"] is None
    assert result["target_registration"]["state"] == "expired"
    assert snapshot(roots) == before


def test_global_assertion_is_independent_of_the_named_target(enrolled, capsys):
    runtime, _, project, _ = enrolled

    assert preview(runtime, project, global_args=("--expect-machine", "global-name")) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["target_registration"]["machine_name"] == "g2"
    assert result["blockers"][0]["code"] == "target_registration_eligible"


def test_preview_uses_durable_project_identity(tmp_path, capsys):
    project = tmp_path / "project"
    initialize_project(project)
    shared_root = project / ".qexp"
    identity_path = shared_root / "project" / "identity.json"
    document = json.loads(identity_path.read_bytes())
    project_id = "b0ae06daa2ce4b67"
    assert document["project"]["project_id"] != project_id
    document["project"]["project_id"] = project_id
    atomic_replace(identity_path, document)
    runtime = MachineRuntime(tmp_path / "machine")
    previous = MachineRuntime(tmp_path / "previous")
    initialize_machine(runtime, "g3")
    initialize_machine(previous, "g2")
    for machine in (runtime, previous):
        assert register_projects(machine, [project])["projects"][0]["status"] == "registered"

    assert preview(runtime, project) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["project_id"] == project_id
    assert result["target_registration"]["state"] == "eligible"
    assert result["current_binding"]["machine_name"] == "g3"


@pytest.mark.parametrize(
    ("path_key", "expected_code"),
    [
        ("registration_transaction", "registration_recovery_pending"),
        ("replacement_transaction", "machine_replacement_pending"),
    ],
)
def test_preview_preserves_pending_ownership_transactions(enrolled, capsys, path_key, expected_code):
    runtime, _, project, _ = enrolled
    journal = runtime.paths[path_key]
    journal.write_bytes(b"malformed transaction")

    assert preview(runtime, project) == 1

    result = json.loads(capsys.readouterr().out)
    assert expected_code in {blocker["code"] for blocker in result["blockers"]}
    assert journal.read_bytes() == b"malformed transaction"


def test_preview_rejects_a_relative_target_runtime_root(enrolled, capsys):
    runtime, _, project, _ = enrolled
    registration_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    registration = json.loads(registration_path.read_bytes())
    registration["registration"]["runtime_root"] = "../other"
    atomic_replace(registration_path, registration)

    assert preview(runtime, project) == 1

    assert "canonical absolute path" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_preview_returns_an_operational_error_for_unreadable_machine_assertion(enrolled, capsys):
    runtime, _, project, _ = enrolled
    config_path = runtime.paths["global_config"]
    config_path.write_bytes(b'{"invalid"')

    assert preview(runtime, project, global_args=("--expect-machine", "global-name")) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] == "operational_failure"
    assert config_path.read_bytes() == b'{"invalid"'


@pytest.mark.parametrize("is_missing", [False, True])
def test_preview_detects_invalid_reservation_for_the_local_transaction(enrolled, capsys, is_missing):
    runtime, _, project, _ = enrolled
    _, bindings = runtime.load_registry_uncached()
    binding = next(binding for binding in bindings if binding.shared_root.parent == project)
    reservation_path = runtime.paths["active"] / "existing-work.json"
    atomic_replace(
        reservation_path,
        {"reservation": {"reservation_id": "existing-work", "project_id": binding.project_id}},
    )
    result = register_projects(runtime, [project], machine_name="g4", replace_current=True)
    assert result["status"] == "replacement_pending"
    target_reservation = project / ".qexp" / "project" / "registration-reservations" / "g4.json"
    if is_missing:
        target_reservation.unlink()
    else:
        reservation = json.loads(target_reservation.read_bytes())
        reservation["registration_reservation"]["old_generation"] = "different-generation"
        atomic_replace(target_reservation, reservation)
    before = snapshot((runtime.root, project))

    assert preview(runtime, project, "g4") == 1

    reason = "reservation is missing" if is_missing else "another transaction"
    assert reason in json.loads(capsys.readouterr().out)["error"]["message"]
    assert snapshot((runtime.root, project)) == before


def test_confirmation_is_fresh_and_binds_target_expiry_and_local_enablement(enrolled):
    runtime, previous, project, _ = enrolled
    target_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    target = json.loads(target_path.read_bytes())
    target["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(target_path, target)
    roots = (runtime.root, previous.root, project)
    before = snapshot(roots)

    first = preview_project_takeover(runtime, project, "g2")
    second = preview_project_takeover(runtime, project, "g2")
    assert first["plan_id"] != second["plan_id"]
    assert confirm_project_takeover(runtime, project, "g2", first["plan_id"]).original_enabled
    assert snapshot(roots) == before

    old_path = project / ".qexp" / "machines" / "g3" / "registration.json"
    departing = json.loads(old_path.read_bytes())
    departing["registration"]["eligibility_expires_at"] = "2099-01-01T00:00:00Z"
    atomic_replace(old_path, departing)
    assert (
        confirm_project_takeover(runtime, project, "g2", second["plan_id"]).departing.generation
        == (first["current_binding"]["registration_generation"])
    )

    runtime.set_enabled(project / ".qexp", False)
    with pytest.raises(TakeoverUsageError, match="--plan"):
        confirm_project_takeover(runtime, project, "g2", second["plan_id"])
    disabled_plan = preview_project_takeover(runtime, project, "g2")["plan_id"]
    assert confirm_project_takeover(runtime, project, "g2", disabled_plan).original_enabled is False

    # A timestamp change must stale the confirmation even if the target is still expired.
    target["registration"]["eligibility_expires_at"] = "2001-01-01T00:00:00Z"
    atomic_replace(target_path, target)
    with pytest.raises(TakeoverUsageError, match="--plan"):
        confirm_project_takeover(runtime, project, "g2", disabled_plan)


def test_acceptance_persists_exact_operation_and_cancel_request_without_changing_admission(enrolled):
    runtime, previous, project, other = enrolled
    target_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    target = json.loads(target_path.read_bytes())
    target["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(target_path, target)
    preview_result = preview_project_takeover(runtime, project, "g2")
    plan_id = preview_result["plan_id"]
    registry = runtime.load_registry_uncached()
    before = snapshot((previous.root, project, other))

    accepted = accept_project_takeover(runtime, project, "g2", plan_id)
    fresh_runtime = MachineRuntime(runtime.root)
    assert accepted.phase == "accepted"
    assert read_takeover_intent(runtime.root) == accepted
    assert accept_project_takeover(fresh_runtime, project / ".qexp", "g2", plan_id) == accepted
    with pytest.raises(TakeoverUsageError):
        request_takeover_cancellation(fresh_runtime, other, "g2", plan_id)
    with pytest.raises(TakeoverUsageError):
        accept_project_takeover(fresh_runtime, project, "g3", plan_id)

    cancel_requested = request_takeover_cancellation(fresh_runtime, project, "g2", plan_id)
    assert cancel_requested.phase == "accepted"
    assert cancel_requested.is_cancel_requested
    assert accept_project_takeover(fresh_runtime, project, "g2", plan_id) == cancel_requested
    assert runtime.load_registry_uncached() == registry
    assert snapshot((previous.root, project, other)) == before

    identity_path = runtime.paths["identity"]
    identity = json.loads(identity_path.read_bytes())
    identity["machine_runtime"]["runtime_id"] = "0" * 64
    atomic_replace(identity_path, identity)
    with pytest.raises(TakeoverUsageError, match="different machine runtime identity"):
        accept_project_takeover(fresh_runtime, project, "g2", plan_id)
    with pytest.raises(TakeoverUsageError, match="different machine runtime identity"):
        request_takeover_cancellation(fresh_runtime, project, "g2", plan_id)
    assert read_takeover_intent(runtime.root) == cancel_requested


def test_acceptance_rechecks_local_enablement_at_durable_boundary(enrolled, monkeypatch):
    runtime, _, project, _ = enrolled
    registration_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    target = json.loads(registration_path.read_bytes())
    target["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(registration_path, target)
    plan_id = preview_project_takeover(runtime, project, "g2")["plan_id"]

    def confirm_then_disable(*args, **kwargs):
        confirmed = confirm_project_takeover(*args, **kwargs)
        runtime.set_enabled(project / ".qexp", False)
        return confirmed

    monkeypatch.setattr(takeover_requests, "confirm_project_takeover", confirm_then_disable)
    with pytest.raises(TakeoverUsageError, match="--plan"):
        accept_project_takeover(runtime, project, "g2", plan_id)

    assert read_takeover_intent(runtime.root) is None
    assert not runtime.resolve_project_binding(project / ".qexp").enabled


def test_retry_repairs_a_visible_request_whose_durability_barrier_failed(enrolled, monkeypatch):
    runtime, _, project, _ = enrolled
    registration_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    target = json.loads(registration_path.read_bytes())
    target["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(registration_path, target)
    plan_id = preview_project_takeover(runtime, project, "g2")["plan_id"]
    write_intent = takeover_intent._write_intent
    sync_directory = takeover_intent.DurableIO.sync_directory
    replayed = []

    def fail_after_publication(path, intent):
        write_intent(path, intent)
        raise OSError("injected journal publication failure")

    def observe_sync(self, path, label):
        if label == "takeover_intent_replay":
            replayed.append(path)
        return sync_directory(self, path, label)

    monkeypatch.setattr(takeover_intent.DurableIO, "sync_directory", observe_sync)
    with monkeypatch.context() as fault:
        fault.setattr(takeover_intent, "_write_intent", fail_after_publication)
        with pytest.raises(RuntimeError, match="publication failure"):
            accept_project_takeover(runtime, project, "g2", plan_id)
    accepted = accept_project_takeover(runtime, project, "g2", plan_id)
    assert accepted == read_takeover_intent(runtime.root)
    assert replayed == [runtime.root]

    with monkeypatch.context() as fault:
        fault.setattr(takeover_intent, "_write_intent", fail_after_publication)
        with pytest.raises(RuntimeError, match="publication failure"):
            request_takeover_cancellation(runtime, project, "g2", plan_id)
    replayed.clear()
    assert request_takeover_cancellation(runtime, project, "g2", plan_id).is_cancel_requested
    assert replayed == [runtime.root]


def test_unenrolled_takeover_preserves_disabled_inventory_and_rejects_reinitialized_project(enrolled, tmp_path, capsys):
    from qqtools.plugins.qexp.agent.inventory import ProjectInventoryEntry, save_inventory_locked

    _, _, project, _ = enrolled
    runtime = MachineRuntime(tmp_path / "fresh runtime")
    initialize_machine(runtime, "replacement")
    registration_path = project / ".qexp" / "machines" / "g2" / "registration.json"
    document = json.loads(registration_path.read_bytes())
    document["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(registration_path, document)
    shared_root = project / ".qexp"
    old_entry = ProjectInventoryEntry("another-project", shared_root, False)
    with runtime.inventory_guard():
        save_inventory_locked(runtime, 1, [old_entry])

    assert preview(runtime, project) == 1
    blocked = json.loads(capsys.readouterr().out)
    assert "takeover_inventory_unavailable" in {item["code"] for item in blocked["blockers"]}
    assert blocked["plan_id"] is None

    actual_id = document["registration"]["project_id"]
    with runtime.inventory_guard():
        save_inventory_locked(runtime, 2, [ProjectInventoryEntry(actual_id, shared_root, False)])
    plan_id = preview_project_takeover(runtime, project, "g2")["plan_id"]
    assert plan_id is not None
    intent = accept_project_takeover(runtime, project, "g2", plan_id)
    assert not intent.original_enabled
    assert intent.transition.departing is None

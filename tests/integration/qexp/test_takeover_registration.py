"""Verify the shared takeover transaction against real Project records."""

import time
from dataclasses import replace

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.project_io_controller import ProjectIOController
from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor
from qqtools.plugins.qexp.agent.project_io_protocol import ProjectIORequest
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.agent.takeover_intent import TakeoverIntent, save_takeover_intent
from qqtools.plugins.qexp.agent.takeover_transport import run_takeover_step
from qqtools.plugins.qexp.layout import load_machine_record, load_machine_registration, load_root_config
from qqtools.plugins.qexp.runtime import takeover_registration as takeover
from qqtools.plugins.qexp.runtime.paths import machine_project_paths, machine_registration_path
from qqtools.plugins.qexp.runtime.registration_protocol import reservation_path
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = pytest.mark.integration


@pytest.fixture
def transition(tmp_path):
    project = tmp_path / "project"
    successor = MachineRuntime(tmp_path / "successor")
    old_owner = MachineRuntime(tmp_path / "old owner")
    initialize_machine(successor, "g3")
    initialize_machine(old_owner, "g2")
    initialize_project(project)
    for owner in (old_owner, successor):
        assert register_projects(owner, [project])["projects"][0]["status"] == "registered"
    root = project / ".qexp"
    current = successor.resolve_project_binding(root)
    prior = old_owner.resolve_project_binding(root)
    target_cfg = load_root_config(root, "g2")
    departing_cfg = load_root_config(root, "g3")
    target = load_machine_registration(target_cfg)
    target["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(machine_registration_path(root, "g2"), target)
    request = takeover.TakeoverRegistration(
        project_id=current.project_id,
        shared_root=root,
        transaction_id="takeover-operation",
        successor_runtime_id=current.runtime_instance_id,
        successor_runtime_root=successor.root,
        successor_generation="new-registration-generation",
        target=takeover.RegistrationWitness.from_record(target["registration"]),
        departing=takeover.RegistrationWitness.from_record(load_machine_registration(departing_cfg)["registration"]),
    )
    return request, old_owner, prior, target_cfg, departing_cfg


def do_not_fence():
    """The test thread exclusively owns its local administrative intent."""


def test_target_renewal_during_drain_rejects_the_confirmed_takeover(transition):
    request, old_owner, prior, target, departing = transition
    departing_before = load_machine_registration(departing)
    assert takeover.observe_takeover_registration(request) == "unchanged"
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)

    assert old_owner.reactivate_binding(prior)
    renewed = load_machine_registration(target)
    assert takeover.observe_takeover_registration(request) == "changed"
    with pytest.raises(takeover.TakeoverRegistrationConflict):
        takeover.commit_takeover_registration(request, before_mutation=do_not_fence)

    assert load_machine_registration(target) == renewed
    assert load_machine_registration(departing) == departing_before
    assert takeover.release_takeover_registration(request, before_mutation=do_not_fence)
    assert not reservation_path(target).exists()


def test_replay_finishes_exact_successor_after_metadata_publication_failure(transition, monkeypatch):
    request, _, _, target, departing = transition
    previous_machine = load_machine_record(target)
    previous_departing = load_machine_registration(departing)
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)

    def fail_metadata(_, __):
        raise OSError("injected metadata publication failure")

    with monkeypatch.context() as fault:
        fault.setattr(takeover, "save_machine_record", fail_metadata)
        with pytest.raises(OSError, match="injected metadata"):
            takeover.commit_takeover_registration(request, before_mutation=do_not_fence)

    committed = load_machine_registration(target)["registration"]
    assert committed["generation"] == request.successor_generation
    assert committed["runtime_instance_id"] == request.successor_runtime_id
    assert load_machine_record(target) == previous_machine
    assert load_machine_registration(departing) == previous_departing
    assert takeover.observe_takeover_registration(request) == "committed"

    takeover.commit_takeover_registration(request, before_mutation=do_not_fence)

    assert load_machine_registration(target)["registration"]["generation"] == request.successor_generation
    machine = load_machine_record(target)["machine"]
    assert machine["runtime_root"] == str(
        machine_project_paths(request.successor_runtime_root, request.project_id)["root"]
    )
    assert machine["machine_name"] == "g2"
    old_registration = load_machine_registration(departing)["registration"]
    assert old_registration["state"] == "superseded"
    assert old_registration["generation"] == request.departing.generation
    assert old_registration["superseded_by_generation"] == request.successor_generation
    assert takeover.release_takeover_registration(request, before_mutation=do_not_fence)
    assert not takeover.release_takeover_registration(request, before_mutation=do_not_fence)


def test_competing_takeover_cannot_release_or_acquire_the_first_reservation(transition):
    request, _, _, target, _ = transition
    competing = replace(request, transaction_id="competing-operation", successor_generation="competing-generation")
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)
    reservation = reservation_path(target)
    before = reservation.read_bytes()

    with pytest.raises(takeover.TakeoverRegistrationConflict):
        takeover.reserve_takeover_registration(competing, before_mutation=do_not_fence)
    assert not takeover.release_takeover_registration(competing, before_mutation=do_not_fence)
    assert reservation.read_bytes() == before
    assert takeover.observe_takeover_registration(request) == "unchanged"

    takeover.commit_takeover_registration(request, before_mutation=do_not_fence)
    assert takeover.observe_takeover_registration(request) == "committed"
    assert takeover.observe_takeover_registration(competing) == "changed"


def test_takeover_can_grant_to_a_runtime_without_a_departing_binding(transition, tmp_path):
    request, _, _, target, departing = transition
    new_runtime = MachineRuntime(tmp_path / "fresh")
    initialize_machine(new_runtime, "replacement")
    fresh = replace(
        request,
        departing=None,
        successor_runtime_id=new_runtime.instance_id,
        successor_runtime_root=new_runtime.root,
    )
    old_binding_record = load_machine_registration(departing)
    takeover.reserve_takeover_registration(fresh, before_mutation=do_not_fence)
    takeover.commit_takeover_registration(fresh, before_mutation=do_not_fence)

    assert load_machine_registration(departing) == old_binding_record
    assert load_machine_registration(target)["registration"]["runtime_instance_id"] == new_runtime.instance_id


def test_replaying_same_operation_cannot_replace_its_confirmed_target_witness(transition):
    request, _, _, target, _ = transition
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)
    original_reservation = reservation_path(target).read_bytes()
    replacement = load_machine_registration(target)
    replacement["registration"]["eligibility_expires_at"] = "2001-01-01T00:00:00Z"
    atomic_replace(machine_registration_path(request.shared_root, "g2"), replacement)
    forged_replay = replace(request, target=takeover.RegistrationWitness.from_record(replacement["registration"]))

    with pytest.raises(takeover.TakeoverRegistrationConflict):
        takeover.reserve_takeover_registration(forged_replay, before_mutation=do_not_fence)
    with pytest.raises(takeover.TakeoverRegistrationConflict):
        takeover.commit_takeover_registration(forged_replay, before_mutation=do_not_fence)

    assert not takeover.release_takeover_registration(forged_replay, before_mutation=do_not_fence)
    assert reservation_path(target).read_bytes() == original_reservation
    assert load_machine_registration(target) == replacement


def test_takeover_preserves_the_target_name_recovery_protocol(transition):
    request, _, _, target, _ = transition
    from qqtools.plugins.qexp.runtime.registration_protocol import RECOVERY_PROTOCOL

    document = load_machine_registration(target)
    document["registration"]["recovery_protocol"] = RECOVERY_PROTOCOL
    atomic_replace(machine_registration_path(request.shared_root, "g2"), document)

    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)
    takeover.commit_takeover_registration(request, before_mutation=do_not_fence)

    assert load_machine_registration(target)["registration"]["recovery_protocol"] == RECOVERY_PROTOCOL


def test_takeover_probe_cannot_create_a_missing_project_root(transition):
    request, _, _, _, _ = transition
    root = request.shared_root
    root.rename(root.parent / "unmounted-project")

    with pytest.raises((OSError, takeover.TakeoverRegistrationConflict)):
        takeover.observe_takeover_registration(request)

    assert not root.exists()


@pytest.mark.parametrize("relative", ["locks", "project/registration-reservations", "machines/g2"])
def test_takeover_does_not_use_redirected_shared_directories(transition, tmp_path, relative):
    request, _, _, target, _ = transition
    directory = request.shared_root / relative
    directory.mkdir(exist_ok=True)
    moved = tmp_path / "redirected"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    before = sorted(str(path.relative_to(moved)) for path in moved.rglob("*"))

    with pytest.raises(takeover.TakeoverRegistrationConflict):
        takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)

    assert sorted(str(path.relative_to(moved)) for path in moved.rglob("*")) == before
    assert not reservation_path(target).exists()


def test_takeover_honors_competing_reservation_on_departing_name(transition, tmp_path):
    request, _, _, target, departing = transition
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)
    document = load_machine_registration(departing)
    document["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(machine_registration_path(request.shared_root, "g3"), document)
    contender = replace(
        request,
        transaction_id="takeover-departing",
        successor_runtime_id="elsewhere",
        successor_runtime_root=tmp_path / "another machine",
        successor_generation="successor-on-departing",
        target=takeover.RegistrationWitness.from_record(document["registration"]),
        departing=None,
    )
    takeover.reserve_takeover_registration(contender, before_mutation=do_not_fence)
    target_before = load_machine_registration(target)

    with pytest.raises(takeover.TakeoverRegistrationConflict, match="Departing name"):
        takeover.commit_takeover_registration(request, before_mutation=do_not_fence)

    assert load_machine_registration(target) == target_before
    assert load_machine_registration(departing) == document


def test_takeover_validates_supported_project_schema_before_reserving(transition):
    from qqtools.plugins.qexp.runtime.store import read_json

    request, _, _, target, _ = transition
    path = request.shared_root / "schema" / "version.json"
    schema = read_json(path)
    schema["schema"]["version"] = 999
    atomic_replace(path, schema)

    with pytest.raises(takeover.TakeoverRegistrationConflict, match="schema version"):
        takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)

    assert not reservation_path(target).exists()


def test_takeover_mutation_requires_a_fence(transition):
    request, _, _, target, _ = transition

    with pytest.raises(ValueError, match="mutation fence"):
        takeover.reserve_takeover_registration(request, before_mutation=None)

    assert not reservation_path(target).exists()


def consume_worker(executor, request):
    executor.start(request.request_id)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        executor.poll()
        result = executor.consume(request.request_id, request)
        if result is not None:
            return result
        time.sleep(0.02)
    raise AssertionError(f"takeover request {request.request_id} did not produce a consumable result")


def test_takeover_shared_steps_run_through_the_isolated_executor(transition):
    takeover_request, _, _, target, _ = transition
    root = takeover_request.successor_runtime_root
    runtime = MachineRuntime(root)
    revision, _ = runtime.load_registry_uncached()
    intent = TakeoverIntent(plan_id="c" * 64, transition=takeover_request, original_enabled=True)
    save_takeover_intent(root, intent, expected=None)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        reserve = executor.prepare_takeover_registration(intent, revision, "reserve")
        result = consume_worker(executor, reserve)
        assert result.status == "completed"
        assert result.evidence["outcome"] == "reserved"
        assert reservation_path(target).exists()

        for phase in ("reserved", "draining", "verifying_drain", "committing"):
            updated = replace(intent, phase=phase)
            save_takeover_intent(root, updated, expected=intent)
            intent = updated
        commit = executor.prepare_takeover_registration(intent, revision, "commit")
        result = consume_worker(executor, commit)
        assert result.status == "completed"
        assert result.evidence["outcome"] == "committed"
        assert load_machine_registration(target)["registration"]["generation"] == takeover_request.successor_generation

        observe = executor.prepare_takeover_registration(intent, revision, "observe")
        result = consume_worker(executor, observe)
        assert result.status == "completed"
        assert result.evidence["outcome"] == "successor_observed"
    finally:
        executor.shutdown()


def test_prepared_takeover_worker_rejects_later_cancel_request(transition):
    takeover_request, _, _, target, _ = transition
    root = takeover_request.successor_runtime_root
    revision, _ = MachineRuntime(root).load_registry_uncached()
    intent = TakeoverIntent(plan_id="d" * 64, transition=takeover_request, original_enabled=True)
    save_takeover_intent(root, intent, expected=None)
    executor = ProjectIOExecutor(MachineRuntime(root))
    executor.begin_epoch()
    try:
        request = executor.prepare_takeover_registration(intent, revision, "reserve")
        tampered = request.to_dict()
        tampered["project_io_request"]["parameters"]["transition"]["successor_generation"] = "different"
        with pytest.raises(ValueError):
            ProjectIORequest.from_dict(tampered)
        updated = replace(intent, is_cancel_requested=True)
        save_takeover_intent(root, updated, expected=intent)

        result = consume_worker(executor, request)

        assert result.status == "completed"
        assert result.evidence["outcome"] == "intent_changed"
        assert not reservation_path(target).exists()
    finally:
        executor.shutdown()


def test_postcommit_replay_never_regrants_a_reverted_registration(transition):
    takeover_request, _, _, target, _ = transition
    root = takeover_request.successor_runtime_root
    runtime = MachineRuntime(root)
    revision, _ = runtime.load_registry_uncached()
    intent = TakeoverIntent(plan_id="e" * 64, transition=takeover_request, original_enabled=True)
    save_takeover_intent(root, intent, expected=None)
    takeover.reserve_takeover_registration(takeover_request, before_mutation=do_not_fence)
    original = load_machine_registration(target)
    takeover.commit_takeover_registration(takeover_request, before_mutation=do_not_fence)
    for phase in ("reserved", "draining", "verifying_drain", "committing", "shared_committed"):
        updated = replace(intent, phase=phase)
        save_takeover_intent(root, updated, expected=intent)
        intent = updated
    atomic_replace(machine_registration_path(takeover_request.shared_root, "g2"), original)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    try:
        request = executor.prepare_takeover_registration(intent, revision, "commit")
        executor.start(request.request_id)
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            executor.poll()
            result = executor.load_result(request.request_id)
            time.sleep(0.02)

        assert result is not None and result.status == "outcome_unknown"
        assert load_machine_registration(target) == original
    finally:
        executor.shutdown()


def test_existing_agent_finishes_accepted_takeover_and_preserves_other_project(transition, tmp_path):
    from qqtools.plugins.qexp.agent.lifecycle import get_machine_agent_status, start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.agent.takeover_intent import read_takeover_intent, read_takeover_result
    from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
    from qqtools.plugins.qexp.runtime.responsibility_store import Ledger
    from tests.helpers.qexp.lifecycle import wait_until

    takeover_request, _, _, target, departing_cfg = transition
    root = takeover_request.successor_runtime_root
    runtime = MachineRuntime(root)
    peer = tmp_path / "peer"
    initialize_project(peer)
    assert register_projects(runtime, [peer])["projects"][0]["status"] == "registered"
    peer_binding = runtime.resolve_project_binding(peer / ".qexp")
    old_root = runtime.project_paths(takeover_request.project_id)["root"]
    local_receipt = old_root / "preserved-local-receipt.json"
    atomic_replace(local_receipt, {"receipt": {"keep": True}})
    ledger = Ledger.open_or_create(responsibility_root(old_root))
    ledger.capture_local("control-plane", {"task_id": None, "attempt_number": None})
    diagnostic = {
        "authority_diagnostic": {
            "attempt_id": "control-plane",
            "reason": "cleanup_replay_unavailable",
            "at": "2026-01-01T00:00:00Z",
            "error_type": "RuntimeError",
            "error": "historical incident diagnostic",
        }
    }
    atomic_replace(
        runtime.project_paths(takeover_request.project_id)["authority_diagnostics"] / "control-plane.json", diagnostic
    )
    intent = TakeoverIntent(plan_id="f" * 64, transition=takeover_request, original_enabled=True)
    save_takeover_intent(root, intent, expected=None)

    assert get_machine_agent_status(runtime)["is_running"] is False
    worker = start_machine_agent(runtime, available_gpus=[], loop_interval=0.05)
    try:

        def completed():
            result = read_takeover_result(root, intent.plan_id)
            return result is not None and result.phase == "completed" and read_takeover_intent(root) is None

        wait_until("takeover-completed", completed, timeout=25, on_timeout=lambda: read_takeover_intent(root))
        assert get_machine_agent_status(runtime)["is_running"]
        assert worker.poll() is None
        binding = runtime.resolve_project_binding(takeover_request.shared_root)
        assert binding.machine_name == "g2"
        assert binding.enabled
        assert binding.registration_generation == takeover_request.successor_generation
        assert runtime.resolve_project_binding(peer / ".qexp") == peer_binding
        assert load_machine_registration(departing_cfg)["registration"]["state"] == "superseded"
        assert load_machine_registration(target)["registration"]["generation"] == takeover_request.successor_generation
        receipt = root / "archives" / "project-takeovers" / intent.plan_id / "partition" / local_receipt.name
        assert receipt.is_file() and not local_receipt.exists()
        archived_ledger = Ledger(responsibility_root(receipt.parent), is_observation=True)
        assert archived_ledger.lookup("control-plane")["payload"] == {"task_id": None, "attempt_number": None}
        assert not reservation_path(target).exists()
    finally:
        stop_machine_agent(runtime)


def test_takeover_abort_releases_only_its_reservation_and_verifies_departing_authority(transition):
    request, old_owner, prior, target, departing = transition
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)
    assert old_owner.reactivate_binding(prior)
    renewed = load_machine_registration(target)
    assert takeover.abort_takeover_registration(request, before_mutation=do_not_fence) == "departing_current"
    assert not reservation_path(target).exists()
    assert load_machine_registration(target) == renewed

    departing_record = load_machine_registration(departing)
    departing_record["registration"]["generation"] = "another-valid-generation"
    atomic_replace(machine_registration_path(request.shared_root, "g3"), departing_record)

    assert takeover.abort_takeover_registration(request, before_mutation=do_not_fence) == "departing_changed"
    assert load_machine_registration(departing) == departing_record


def test_takeover_abort_observes_the_grant_after_a_metadata_failure(transition, monkeypatch):
    request, _, _, target, departing = transition
    departing_before = load_machine_registration(departing)
    takeover.reserve_takeover_registration(request, before_mutation=do_not_fence)
    reservation_before = reservation_path(target).read_bytes()

    def fail_metadata(_, __):
        raise OSError("injected metadata publication failure")

    with monkeypatch.context() as fault:
        fault.setattr(takeover, "save_machine_record", fail_metadata)
        with pytest.raises(OSError, match="injected metadata"):
            takeover.commit_takeover_registration(request, before_mutation=do_not_fence)

    root = request.successor_runtime_root
    intent = TakeoverIntent(plan_id="f" * 64, transition=request, original_enabled=True)
    save_takeover_intent(root, intent, expected=None)
    cancelling = replace(intent, phase="cancelling", is_cancel_requested=True)
    save_takeover_intent(root, cancelling, expected=intent)
    parameters = {"plan_id": intent.plan_id, "transition": request.to_dict(), "step": "release"}
    assert run_takeover_step(root, parameters, before_mutation=do_not_fence) == {"outcome": "intent_changed"}
    assert run_takeover_step(root, {**parameters, "step": "abort"}, before_mutation=do_not_fence) == {
        "outcome": "successor_observed"
    }
    assert reservation_path(target).read_bytes() == reservation_before

    assert takeover.abort_takeover_registration(request, before_mutation=do_not_fence) == "committed"
    assert reservation_path(target).read_bytes() == reservation_before
    assert load_machine_registration(departing) == departing_before
    takeover.commit_takeover_registration(request, before_mutation=do_not_fence, is_replay_only=True)
    assert load_machine_registration(departing)["registration"]["state"] == "superseded"


def test_takeover_arbiter_admits_exact_intent_with_no_local_project_binding(transition, tmp_path):
    takeover_request, _, _, target, _ = transition
    runtime = MachineRuntime(tmp_path / "fresh runtime")
    initialize_machine(runtime, "replacement")
    revision, bindings = runtime.load_registry_uncached()
    assert not bindings
    takeover_request = replace(
        takeover_request,
        departing=None,
        successor_runtime_id=runtime.instance_id,
        successor_runtime_root=runtime.root,
    )
    intent = TakeoverIntent(plan_id="c" * 64, transition=takeover_request, original_enabled=True)
    save_takeover_intent(runtime.root, intent, expected=None)
    executor = ProjectIOExecutor(runtime)
    executor.begin_epoch()
    controller = ProjectIOController(runtime, executor)
    try:
        with controller.admission_turn():
            assert controller.advance_takeover_registration(intent, revision, "reserve") is None
            assert not executor.unresolved_requests()
        requests = executor.unresolved_requests()
        assert len(requests) == 1
        assert requests[0].parameters["step"] == "reserve"
        assert requests[0].registration_generation == takeover_request.successor_generation

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            # Do not restart reserve if the controller has since chosen to stop.
            settled = controller.advance_takeover_registration(intent, revision, None)
            if settled is not None:
                break
            time.sleep(0.02)
        else:
            raise AssertionError("the shared arbiter did not settle the exact reserve request")
        assert settled == ("reserve", "reserved")
        assert reservation_path(target).exists()
        assert not executor.unresolved_requests()
    finally:
        executor.shutdown()


@pytest.mark.parametrize("action", ["drain", "cancel", "restart"])
def test_takeover_drains_or_cancels_without_interrupting_running_work(
    transition, tmp_path, monkeypatch, capsys, action
):
    import json
    import sys

    from qqtools.plugins.qexp.agent.lifecycle import start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.agent.project_takeover import preview_project_takeover
    from qqtools.plugins.qexp.agent.takeover_intent import read_takeover_intent, read_takeover_result
    from qqtools.plugins.qexp.agent.takeover_requests import accept_project_takeover, request_takeover_cancellation
    from qqtools.plugins.qexp.cli.entrypoint import main
    from qqtools.plugins.qexp.commands.task import submit
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from tests.helpers.qexp.lifecycle import wait_until

    should_cancel = action == "cancel"
    request, _, _, target, departing = transition
    runtime = MachineRuntime(request.successor_runtime_root)
    root = request.shared_root.parent
    marker = tmp_path / "running"
    finish = tmp_path / "finish"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path\nimport time\n"
        f"Path({str(marker)!r}).touch()\nfinish=Path({str(finish)!r})\n"
        "deadline=time.monotonic()+45\n"
        "while not finish.exists():\n"
        "    if time.monotonic()>deadline: raise SystemExit(99)\n"
        "    time.sleep(.02)\n",
    ]
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0")
    training = submit(departing, command, task_id="training", working_dir=root)
    start_machine_agent(runtime, available_gpus=[0], loop_interval=0.05)
    try:

        def is_training():
            return marker.exists() and load_task(departing, training.task_id).state["projection"] == "running"

        wait_until("departing-training-running", is_training, timeout=30)
        plan = preview_project_takeover(runtime, root, "g2")
        intent = accept_project_takeover(runtime, root, "g2", plan["plan_id"])

        def is_draining():
            active = read_takeover_intent(runtime.root)
            binding = runtime.resolve_project_binding(request.shared_root)
            return (
                active is not None
                and active.phase == "draining"
                and binding.machine_name == "g3"
                and not binding.enabled
            )

        wait_until("takeover-draining", is_draining, timeout=25, on_timeout=lambda: read_takeover_intent(runtime.root))
        assert is_training()
        status_args = ["--machine-runtime-root", str(runtime.root), "--project", str(root), "status", "--format=json"]
        assert main(status_args) == 0
        project_status = json.loads(capsys.readouterr().out)
        assert project_status["takeover"]["phase"] == "draining"
        assert project_status["takeover"]["plan_id"] == intent.plan_id
        assert project_status["local_participation"]["binding"]["enabled"] is False
        if should_cancel:
            requested = request_takeover_cancellation(runtime, root, "g2", intent.plan_id)
            assert requested.is_cancel_requested
        elif action == "restart":
            assert stop_machine_agent(runtime)
            assert is_training()
            finish.touch()
            start_machine_agent(MachineRuntime(runtime.root), available_gpus=[0], loop_interval=0.05)
        else:
            finish.touch()

        def is_done():
            result = read_takeover_result(runtime.root, intent.plan_id)
            expected = "cancelled" if should_cancel else "completed"
            return result is not None and result.phase == expected and read_takeover_intent(runtime.root) is None

        wait_until("takeover-done", is_done, timeout=30, on_timeout=lambda: read_takeover_intent(runtime.root))
        binding = runtime.resolve_project_binding(request.shared_root)
        assert binding.enabled and binding.machine_name == ("g3" if should_cancel else "g2")
        assert not reservation_path(target).exists()
        if should_cancel:
            assert is_training()
            finish.touch()
        wait_until(
            "departing-training-succeeded",
            lambda: load_task(departing, training.task_id).state["projection"] == "succeeded",
            timeout=20,
        )
    finally:
        finish.touch()
        stop_machine_agent(runtime)


def test_takeover_cli_runs_private_canary_and_preserves_concurrent_group_edit(transition, monkeypatch, capsys):
    import json

    from qqtools.plugins.qexp.agent.lifecycle import stop_machine_agent
    from qqtools.plugins.qexp.agent.takeover_intent import read_takeover_intent, read_takeover_result
    from qqtools.plugins.qexp.agent.takeover_requests import accept_project_takeover
    from qqtools.plugins.qexp.cli.entrypoint import main
    from qqtools.plugins.qexp.commands.group import create_group
    from qqtools.plugins.qexp.commands.task import submit
    from qqtools.plugins.qexp.runtime.paths import group_path
    from qqtools.plugins.qexp.runtime.resources.cpu_lane import set_cpu_lane_capacity
    from qqtools.plugins.qexp.runtime.store import read_json
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from tests.helpers.qexp.lifecycle import wait_until

    request, old_owner, prior, target, _ = transition
    runtime = MachineRuntime(request.successor_runtime_root)
    root = request.shared_root.parent
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0")
    set_cpu_lane_capacity(runtime.root, capacity=1)
    assert old_owner.reactivate_binding(prior)
    for group, task in (("approved", "canary"), ("edited", "blocked")):
        create_group(target, group, ["g2", "g3"])
        submit(target, ["echo", task], requested_gpus=0, requested_cpus=1, task_id=task, group=group, working_dir=root)
        path = group_path(request.shared_root, group)
        document = read_json(path)
        document["group"]["worker_set"].pop("g2")
        atomic_replace(path, document)
    expired = load_machine_registration(target)
    expired["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(machine_registration_path(request.shared_root, "g2"), expired)
    command = [
        "--machine-runtime-root",
        str(runtime.root),
        "project",
        "takeover",
        str(root),
        "--machine=g2",
        "--format=json",
    ]
    selection = ["--add-primary-worker=approved", "--add-primary-worker=edited"]
    assert main([*command, "--plan", *selection]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [change["group_name"] for change in plan["selected_group_changes"]] == ["approved", "edited"]
    assert "g2" not in read_json(group_path(request.shared_root, "approved"))["group"]["worker_set"]

    plan_id = plan["plan_id"]
    accept_project_takeover(runtime, root, "g2", plan_id, selected_groups=("approved", "edited"))
    edited_path = group_path(request.shared_root, "edited")
    edited = read_json(edited_path)
    edited["meta"]["revision"] += 1
    atomic_replace(edited_path, edited)

    try:
        assert main([*command, f"--apply={plan_id}", *selection]) == 0
        assert json.loads(capsys.readouterr().out)["phase"] == "accepted"

        def completed():
            return read_takeover_result(runtime.root, plan_id) is not None

        def progress():
            paths = runtime.paths
            requests = [read_json(path)["project_io_request"] for path in paths["project_io_requests"].glob("*.json")]
            resolutions = [
                read_json(path)["project_io_resolution"]
                for path in sorted(paths["project_io_resolved"].glob("*.json"))[-8:]
            ]
            return {
                "intent": read_takeover_intent(runtime.root),
                "waiting": [(row["operation_kind"], row["parameters"]) for row in requests],
                "recent_outcomes": [(row["resolution"], (row["result"] or {}).get("status")) for row in resolutions],
            }

        wait_until("takeover-group-results", completed, timeout=75, on_timeout=progress)
        assert main([*command, f"--operation-status={plan_id}"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["phase"] == "completed"
        assert [change["outcome"] for change in result["group_approvals"]] == ["applied", "stale"]
        current_edited = read_json(group_path(request.shared_root, "edited"))
        assert current_edited["meta"]["revision"] >= edited["meta"]["revision"]
        assert current_edited["group"]["worker_set"] == edited["group"]["worker_set"]
        assert load_task(target, "blocked").state["projection"] == "queued"
        workers = read_json(group_path(request.shared_root, "approved"))["group"]["worker_set"]
        assert set(workers) == {"g2", "g3"}
        assert workers["g2"]["scheduling_role"] == "primary"
        assert workers["g2"]["takeover_plan_id"] == plan_id
        wait_until(
            "private-canary-succeeded",
            lambda: load_task(target, "canary").state["projection"] == "succeeded",
            timeout=30,
        )
        assert load_task(target, "canary").attempt_control["current_attempt_number"] == 1
    finally:
        stop_machine_agent(runtime)


def test_takeover_drain_retains_a_runner_that_has_no_published_ledger(transition, monkeypatch, tmp_path):
    from qqtools.plugins.qexp.agent.takeover_drain import observe_takeover_drain
    from qqtools.plugins.qexp.runtime import responsibility_process_capture as processes

    request, _, _, _, _ = transition
    runtime = MachineRuntime(request.successor_runtime_root)
    root = runtime.project_paths(request.project_id)["root"]
    root.mkdir(parents=True, exist_ok=True)
    proc = tmp_path / "proc"
    pid = 123456
    process = proc / str(pid)
    process.mkdir(parents=True)
    (process / "stat").write_text(f"{pid} (runner) " + " ".join(["S", *(["0"] * 18), "123"]))
    flags = {
        "--shared-root": str(request.shared_root),
        "--runtime-root": str(root),
        "--machine": request.departing.machine_name,
        "--task-id": "runner",
        "--attempt-id": "runner-attempt-1",
        "--fencing-token": "1",
        "--launch-id": "runner-launch",
    }
    argv = ["python", "-m", processes.RUNNER_MODULE, *(item for pair in flags.items() for item in pair)]
    (process / "cmdline").write_bytes(("\0".join(argv) + "\0").encode())
    monkeypatch.setattr(processes, "PROC_ROOT", proc)
    monkeypatch.setattr(processes, "_process_scope", lambda: {"pid_namespace": 1})

    assert observe_takeover_drain(request) == "drain_pending"
    (process / "stat").write_text(f"{pid} (runner) " + " ".join(["Z", *(["0"] * 18), "123"]))
    assert observe_takeover_drain(request) == "drained"


def test_drain_requires_durable_legacy_source_release_before_retirement(transition, monkeypatch, tmp_path):
    from qqtools.plugins.qexp.agent import takeover_drain
    from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
    from qqtools.plugins.qexp.runtime.responsibility_capture import CAPTURE_FILE
    from qqtools.plugins.qexp.runtime.responsibility_completion import (
        COMPLETION_FILE,
        SOURCE_RELEASE_FILE,
        _source_release_receipt,
    )
    from qqtools.plugins.qexp.runtime.responsibility_store import Ledger

    request, _, _, _, _ = transition
    departing = request.departing
    assert departing is not None
    root = machine_project_paths(request.successor_runtime_root, request.project_id)["root"]
    root.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "legacy"
    source.mkdir()
    ledger = Ledger.open_or_create(responsibility_root(root))
    assert not ledger.has_members()
    proof = {
        "project_id": request.project_id,
        "shared_root": str(request.shared_root),
        "machine_name": departing.machine_name,
        "owner_root": departing.runtime_root,
        "owner_instance": departing.runtime_id,
        "registration_generation": departing.generation,
        "legacy_source": str(source),
    }
    for name in (COMPLETION_FILE, CAPTURE_FILE, "responsibility-capture-backfill.json"):
        atomic_replace(root / name, {"retained": True})
    atomic_replace(root / SOURCE_RELEASE_FILE, _source_release_receipt(proof))
    atomic_replace(
        root / "migration.json",
        {
            "migration": {
                "state": "active",
                "project_id": request.project_id,
                "shared_root": str(request.shared_root),
                "machine_name": departing.machine_name,
                "legacy_runtime_root": str(source),
            }
        },
    )
    monkeypatch.setattr(takeover_drain, "read_capture_completion", lambda *_args, **_kwargs: proof)
    monkeypatch.setattr(takeover_drain, "has_live_runner", lambda _cfg: False)
    sync = takeover_drain.DurableIO.sync_directory

    def failed_source_sync(self, directory, label):
        if directory == source:
            raise OSError("injected source directory fsync failure")
        return sync(self, directory, label)

    with monkeypatch.context() as fault:
        fault.setattr(takeover_drain.DurableIO, "sync_directory", failed_source_sync)
        assert takeover_drain.observe_takeover_drain(request) == "drain_unavailable"

    assert takeover_drain.observe_takeover_drain(request) == "drained"
    assert (root / SOURCE_RELEASE_FILE).is_file()

"""Replacement retains runtime identity and uses existing removal obligations."""

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.config import load_agent_config
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.inventory import load_inventory
from qqtools.plugins.qexp.agent.setup import (
    initialize_machine,
    initialize_project,
    register_projects,
    set_project_enablement,
)
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = pytest.mark.integration


@pytest.fixture
def enrolled(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "global-name")
    root = tmp_path / "project"
    initialize_project(root)
    result = register_projects(runtime, [root], machine_name="g3")
    assert result["projects"][0]["status"] == "registered"
    binding = runtime.resolve_project_binding(root / ".qexp")
    return runtime, root, binding


def invoke(runtime, root, *options):
    return main(
        [
            "--machine-runtime-root",
            str(runtime.root),
            "project",
            "register",
            str(root),
            *options,
            "--format=json",
        ]
    )


def replacement(runtime, root, name="g2"):
    return register_projects(runtime, [root], machine_name=name, replace_current=True)


def hold_reservation(runtime, binding):
    path = runtime.paths["active"] / "existing-work.json"
    atomic_replace(
        path,
        {
            "reservation": {
                "reservation_id": "existing-work",
                "project_id": binding.project_id,
                "runtime_instance_id": runtime.instance_id,
            }
        },
    )
    return path


@pytest.mark.parametrize("enabled", [False, True])
def test_idle_replacement_preserves_runtime_default_and_unrelated_project(enrolled, tmp_path, enabled):
    runtime, root, before = enrolled
    peer = tmp_path / "other"
    initialize_project(peer)
    register_projects(runtime, [peer], machine_name="peer")
    peer_before = runtime.resolve_project_binding(peer / ".qexp")
    if not enabled:
        assert set_project_enablement(runtime, root, False)["status"] == "committed"
    runtime_id = runtime.instance_id
    global_name = load_agent_config(runtime).name
    result = replacement(runtime, root)
    assert result["action"] == "project_registration_replacement"
    assert result["status"] == "replacement_completed"
    assert result["runtime_id"] == runtime_id == runtime.instance_id
    assert result["old_machine_name"] == "g3"
    assert result["requested_machine_name"] == "g2"
    assert result["cancellable"] is False
    after = runtime.resolve_project_binding(root / ".qexp")
    assert after.machine_name == "g2"
    assert after.registration_generation != before.registration_generation
    assert result["registration_generation"] == after.registration_generation
    assert after.enabled is enabled
    assert load_agent_config(runtime).name == global_name
    assert runtime.resolve_project_binding(peer / ".qexp") == peer_before
    entry = next(item for item in load_inventory(runtime)[1] if item.project_id == after.project_id)
    assert (entry.name_source, entry.name_override, entry.enabled) == ("explicit", "g2", enabled)
    assert not (runtime.root / "project-registration-replacement.json").exists()


def test_same_name_does_not_allocate_generation_or_disable(enrolled):
    runtime, root, binding = enrolled
    result = replacement(runtime, root, "g3")
    assert result["status"] == "unchanged"
    assert runtime.resolve_project_binding(root / ".qexp") == binding
    assert not (runtime.root / "project-registration-replacement.json").exists()


def test_reservation_blocks_retirement_and_exact_retry_converges(enrolled, capsys):
    runtime, root, binding = enrolled
    evidence = hold_reservation(runtime, binding)
    before = evidence.read_bytes()
    assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "replacement_pending"
    assert result["phase"] == "draining"
    assert result["cancellable"] is True
    assert result["blockers"]
    current = runtime.resolve_project_binding(root / ".qexp")
    assert current.enabled is False
    assert current.machine_name == "g3"
    assert current.registration_generation == binding.registration_generation
    assert evidence.read_bytes() == before
    repeat = replacement(runtime, root)
    assert repeat["status"] == "replacement_pending"
    evidence.unlink()
    completed = replacement(runtime, root)
    assert completed["status"] == "replacement_completed"
    generation = completed["registration_generation"]
    assert replacement(runtime, root)["status"] == "unchanged"
    assert runtime.resolve_project_binding(root / ".qexp").registration_generation == generation


@pytest.mark.parametrize("enabled", [False, True])
def test_cancellation_restores_original_enablement_and_preserves_evidence(enrolled, enabled):
    runtime, root, binding = enrolled
    if not enabled:
        set_project_enablement(runtime, root, False)
    evidence = hold_reservation(runtime, binding)
    before = evidence.read_bytes()
    assert replacement(runtime, root)["status"] == "replacement_pending"
    result = register_projects(runtime, [root], cancel_replacement=True)
    assert result["status"] == "replacement_cancelled"
    current = runtime.resolve_project_binding(root / ".qexp")
    assert current.machine_name == "g3"
    assert current.registration_generation == binding.registration_generation
    assert current.enabled is enabled
    assert evidence.read_bytes() == before
    assert not (runtime.root / "project-registration-replacement.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        ("--replace-current",),
        ("--replace-current", "--machine", "g2", "--adopt-existing"),
        ("--replace-current", "--machine", "g2", "--from-pool"),
        ("--replace-current", "--machine", "g2", "--name-source", "default"),
        ("--cancel-replacement", "--machine", "g2"),
        ("--cancel-replacement", "--replace-current"),
        ("--cancel-replacement", "--adopt-existing"),
        ("--cancel-replacement", "--name-source", "explicit"),
    ],
)
def test_invalid_options_fail_without_mutation(enrolled, capsys, options):
    runtime, root, _binding = enrolled
    registry = runtime.paths["registry"].read_bytes()
    inventory = runtime.paths["inventory"].read_bytes()
    assert invoke(runtime, root, *options) == 2
    capsys.readouterr()
    assert runtime.paths["registry"].read_bytes() == registry
    assert runtime.paths["inventory"].read_bytes() == inventory
    assert not (runtime.root / "project-registration-replacement.json").exists()


def test_competing_eligible_target_fails_before_drain(enrolled, tmp_path, capsys):
    runtime, root, binding = enrolled
    competitor = MachineRuntime(tmp_path / "competitor")
    initialize_machine(competitor, "competitor")
    register_projects(competitor, [root], machine_name="g2")
    assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 1
    capsys.readouterr()
    assert runtime.resolve_project_binding(root / ".qexp") == binding
    assert not (runtime.root / "project-registration-replacement.json").exists()


def test_reserved_target_rejects_competing_registration(enrolled, tmp_path):
    runtime, root, binding = enrolled
    hold_reservation(runtime, binding)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    competitor = MachineRuntime(tmp_path / "competitor")
    initialize_machine(competitor, "competitor")
    result = register_projects(competitor, [root], machine_name="g2")
    assert result["projects"][0]["status"] == "conflicting"
    assert competitor.load_registry()[1] == []
    assert runtime.resolve_project_binding(root / ".qexp").machine_name == "g3"


@pytest.mark.parametrize(
    "phase",
    [
        "preparing",
        "reserved",
        "draining",
        "old_retirement_committed",
        "target_published",
        "binding_published",
        "completed",
    ],
)
@pytest.mark.parametrize("after_write", [False, True])
def test_interrupted_replacement_resumes_exact_generation(enrolled, monkeypatch, capsys, phase, after_write):
    import os
    from pathlib import Path

    runtime, root, old = enrolled
    publish = os.replace
    observed = []

    class Interrupted(BaseException):
        pass

    def interrupt(source, destination, *args, **kwargs):
        if Path(destination).name == "project-registration-replacement.json":
            document = json.loads(Path(source).read_text())
            intent = next(value for value in document.values() if isinstance(value, dict) and "phase" in value)
            observed.append(intent)
            if intent["phase"] == phase:
                if after_write:
                    publish(source, destination, *args, **kwargs)
                raise Interrupted
        return publish(source, destination, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", interrupt)
        with pytest.raises(Interrupted):
            replacement(runtime, root)
    bindings = runtime.load_registry()[1]
    assert len([item for item in bindings if item.project_id == old.project_id and item.enabled]) <= 1
    restarted = MachineRuntime(runtime.root)
    intent_path = runtime.root / "project-registration-replacement.json"
    if intent_path.exists():
        document = json.loads(intent_path.read_text())
        intent = next(value for value in document.values() if isinstance(value, dict) and "phase" in value)
        if intent["phase"] in {"old_retirement_committed", "target_published", "binding_published", "completed"}:
            before = intent_path.read_bytes()
            assert invoke(restarted, root, "--cancel-replacement") == 1
            error_output = capsys.readouterr()
            assert "--replace-current" in error_output.out + error_output.err
            assert intent_path.read_bytes() == before
    result = replacement(restarted, root)
    assert result["status"] == "replacement_completed"
    current = restarted.resolve_project_binding(root / ".qexp")
    assert current.machine_name == "g2"
    assert current.runtime_instance_id == old.runtime_instance_id
    if phase != "preparing" or after_write:
        assert current.registration_generation == observed[0]["target_generation"]
    assert current.enabled
    assert not (runtime.root / "project-registration-replacement.json").exists()


@pytest.mark.parametrize("phase", ["cancelling", "reservation_release_committed"])
@pytest.mark.parametrize("after_write", [False, True])
def test_interrupted_cancellation_restores_exact_old_identity(enrolled, monkeypatch, phase, after_write):
    import os
    from pathlib import Path

    runtime, root, old = enrolled
    evidence = hold_reservation(runtime, old)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    publish = os.replace

    class Interrupted(BaseException):
        pass

    def interrupt(source, destination, *args, **kwargs):
        if Path(destination).name == "project-registration-replacement.json":
            document = json.loads(Path(source).read_text())
            intent = next(value for value in document.values() if isinstance(value, dict) and "phase" in value)
            if intent["phase"] == phase:
                if after_write:
                    publish(source, destination, *args, **kwargs)
                raise Interrupted
        return publish(source, destination, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", interrupt)
        with pytest.raises(Interrupted):
            register_projects(runtime, [root], cancel_replacement=True)
    result = register_projects(MachineRuntime(runtime.root), [root], cancel_replacement=True)
    assert result["status"] == "replacement_cancelled"
    assert runtime.resolve_project_binding(root / ".qexp") == old
    assert evidence.exists()


def test_unactivated_project_rejects_replacement_before_acceptance(enrolled, capsys):
    runtime, root, old = enrolled
    path = root / ".qexp" / "schema" / "version.json"
    schema = json.loads(path.read_text())
    schema["schema"]["required_capabilities"].remove("registration-replacement-v1")
    atomic_replace(path, schema)
    assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 1
    capsys.readouterr()
    assert runtime.resolve_project_binding(root / ".qexp") == old
    assert not (runtime.root / "project-registration-replacement.json").exists()


@pytest.mark.parametrize("ordinary_resume", [False, True])
def test_cancellation_after_release_does_not_delete_successor_reservation(enrolled, monkeypatch, ordinary_resume):
    import os
    from pathlib import Path

    runtime, root, binding = enrolled
    hold_reservation(runtime, binding)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    claim = root / ".qexp" / "project" / "registration-reservations" / "g2.json"
    original_claim = json.loads(claim.read_text())
    unlink = os.unlink

    class Interrupted(BaseException):
        pass

    def interrupt(path, *args, **kwargs):
        result = unlink(path, *args, **kwargs)
        if Path(path) == claim:
            raise Interrupted
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "unlink", interrupt)
        with pytest.raises(Interrupted):
            register_projects(runtime, [root], cancel_replacement=True)
    successor = original_claim
    successor["registration_reservation"]["transaction_id"] = "successor-transaction"
    successor["registration_reservation"]["runtime_id"] = "successor-runtime"
    atomic_replace(claim, successor)
    before = claim.read_bytes()
    restarted = MachineRuntime(runtime.root)
    result = (
        replacement(restarted, root)
        if ordinary_resume
        else register_projects(restarted, [root], cancel_replacement=True)
    )
    assert result["status"] == "replacement_cancelled"
    assert claim.read_bytes() == before
    assert runtime.resolve_project_binding(root / ".qexp") == binding


def test_pending_replacement_fences_identity_enablement_and_removal(enrolled):
    from qqtools.plugins.qexp.agent.setup import remove_project

    runtime, root, binding = enrolled
    hold_reservation(runtime, binding)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    before = runtime.instance_id
    for operation in (
        lambda: initialize_machine(runtime, "new-runtime", confirmed=True),
        lambda: set_project_enablement(runtime, root, True),
        lambda: remove_project(runtime, root),
    ):
        with pytest.raises((RuntimeError, ValueError)):
            operation()
    assert runtime.instance_id == before
    assert runtime.resolve_project_binding(root / ".qexp").enabled is False


@pytest.mark.parametrize("interrupt_cleanup", [False, True])
def test_running_work_finishes_under_old_name_before_replacement(enrolled, monkeypatch, tmp_path, interrupt_cleanup):
    import sys
    import time

    from qqtools.plugins.qexp import submit
    from qqtools.plugins.qexp.agent.lifecycle import start_machine_agent, stop_machine_agent
    from qqtools.plugins.qexp.layout import load_root_config
    from qqtools.plugins.qexp.runtime.paths import attempt_path
    from qqtools.plugins.qexp.runtime.store import read_json
    from qqtools.plugins.qexp.runtime.tasks import load_task
    from tests.helpers.qexp.lifecycle import wait_until

    runtime, root, old = enrolled
    monkeypatch.setenv("QEXP_VISIBLE_GPUS", "0,1")
    cfg = load_root_config(root / ".qexp", "g3")
    marker = tmp_path / "running"
    finish = tmp_path / "finish"
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path\nimport time\n"
        f"Path({str(marker)!r}).touch()\nfinish=Path({str(finish)!r})\n"
        "deadline=time.monotonic()+60\n"
        "while not finish.exists():\n"
        "    if time.monotonic()>deadline: raise SystemExit(99)\n"
        "    time.sleep(.02)\n",
    ]
    task = submit(cfg, command, working_dir=root)
    process = start_machine_agent(runtime, available_gpus=[0, 1], loop_interval=0.05)
    try:
        wait_until(
            "old-name-running",
            lambda: marker.exists() and load_task(cfg, task.task_id).state["projection"] == "running",
            timeout=30,
        )
        pending = replacement(runtime, root)
        assert pending["status"] == "replacement_pending"
        assert runtime.resolve_project_binding(root / ".qexp").machine_name == "g3"
        assert load_task(cfg, task.task_id).state["projection"] == "running"
        next_marker = tmp_path / "new-admission"
        next_task = submit(
            cfg,
            [sys.executable, "-c", f"from pathlib import Path; Path({str(next_marker)!r}).touch()"],
            working_dir=root,
        )
        observe_until = time.monotonic() + 1.0
        while time.monotonic() < observe_until:
            assert not next_marker.exists(), "new admission occurred while the old binding was draining"
            time.sleep(0.02)
        finish.touch()
        wait_until(
            "old-name-terminal",
            lambda: load_task(cfg, task.task_id).state["projection"] == "succeeded",
            timeout=30,
        )
        completed = []
        latest = []
        cleanup_faults = []
        if interrupt_cleanup:
            from qqtools.plugins.qexp.runtime.responsibility_store import DurableIO

            sync = DurableIO.sync_directory

            def fail_cleanup_barrier(self, path, label):
                if label == "cleanup_evidence_directories" and not cleanup_faults:
                    cleanup_faults.append(label)
                    raise OSError("interrupted cleanup directory barrier")
                return sync(self, path, label)

            monkeypatch.setattr(DurableIO, "sync_directory", fail_cleanup_barrier)

        def converge():
            result = replacement(runtime, root)
            latest[:] = [result]
            if result["status"] == "replacement_completed":
                completed.append(result)
                return True
            return False

        wait_until("replacement-completed", converge, timeout=30, on_timeout=lambda: latest)
        if interrupt_cleanup:
            assert cleanup_faults == ["cleanup_evidence_directories"]
        assert completed[0]["runtime_id"] == old.runtime_instance_id
        attempt = read_json(attempt_path(cfg.shared_root, task.task_id, 1))["attempt"]
        assert attempt["machine_name"] == "g3"
        assert attempt["result"]["exit_code"] == 0
        assert runtime.resolve_project_binding(root / ".qexp").machine_name == "g2"
        old_registration = read_json(cfg.shared_root / "machines/g3/registration.json")["registration"]
        assert old_registration["generation"] == old.registration_generation
        assert old_registration["state"] == "superseded"
        assert not next_marker.exists()
        new_cfg = load_root_config(root / ".qexp", "g2")
        new_marker = tmp_path / "new-name-admission"
        new_task = submit(
            new_cfg,
            [sys.executable, "-c", f"from pathlib import Path; Path({str(new_marker)!r}).touch()"],
            working_dir=root,
        )
        wait_until(
            "new-name-terminal",
            lambda: load_task(new_cfg, new_task.task_id).state["projection"] == "succeeded",
            timeout=30,
        )
        next_attempt = read_json(attempt_path(cfg.shared_root, new_task.task_id, 1))["attempt"]
        assert next_attempt["machine_name"] == "g2"
        assert new_marker.exists()
        assert not next_marker.exists()
        assert load_task(cfg, next_task.task_id).attempt_control.get("current_attempt_id") is None
    finally:
        finish.touch()
        stop_machine_agent(runtime, timeout=10)
        process.wait(timeout=10)


@pytest.mark.parametrize(
    "field,value", [("generation", "successor-generation"), ("runtime_instance_id", "successor-runtime")]
)
def test_changed_old_authority_fences_resume_and_cancel(enrolled, capsys, field, value):
    runtime, root, binding = enrolled
    evidence = hold_reservation(runtime, binding)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    evidence.unlink()
    registration_path = root / ".qexp" / "machines" / "g3" / "registration.json"
    registration = json.loads(registration_path.read_text())
    registration["registration"][field] = value
    atomic_replace(registration_path, registration)
    before = registration_path.read_bytes()
    for flags in (("--machine", "g2", "--replace-current"), ("--cancel-replacement",)):
        assert invoke(runtime, root, *flags) == 1
        capsys.readouterr()
        assert registration_path.read_bytes() == before
        current = runtime.resolve_project_binding(root / ".qexp")
        assert current.machine_name == "g3"
        assert current.enabled is False
        assert (runtime.root / "project-registration-replacement.json").exists()


def test_existing_ineligible_target_still_requires_separate_adoption(enrolled, tmp_path, capsys):
    runtime, root, binding = enrolled
    competitor = MachineRuntime(tmp_path / "competitor")
    initialize_machine(competitor, "competitor")
    register_projects(competitor, [root], machine_name="g2")
    path = root / ".qexp" / "machines" / "g2" / "registration.json"
    record = json.loads(path.read_text())
    record["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(path, record)
    before = path.read_bytes()
    assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 1
    capsys.readouterr()
    assert path.read_bytes() == before
    assert runtime.resolve_project_binding(root / ".qexp") == binding


def test_pending_replacement_remains_visible_to_list(enrolled, capsys):
    runtime, root, binding = enrolled
    hold_reservation(runtime, binding)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    assert main(["--machine-runtime-root", str(runtime.root), "project", "list", "--format=json"]) == 0
    rendered = capsys.readouterr().out
    document = json.loads(rendered)
    assert document
    assert "g2" in rendered and "draining" in rendered
    assert runtime.resolve_project_binding(root / ".qexp").enabled is False


def test_missing_local_binding_does_not_turn_replacement_into_enrollment(tmp_path, capsys):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "default")
    root = tmp_path / "project"
    initialize_project(root)
    assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 1
    capsys.readouterr()
    assert runtime.load_registry()[1] == []
    assert not (runtime.root / "project-registration-replacement.json").exists()


def test_pending_replacement_cannot_be_retargeted(enrolled, capsys):
    runtime, root, binding = enrolled
    hold_reservation(runtime, binding)
    assert replacement(runtime, root)["status"] == "replacement_pending"
    path = runtime.root / "project-registration-replacement.json"
    before = path.read_bytes()
    assert invoke(runtime, root, "--machine", "g4", "--replace-current") == 1
    capsys.readouterr()
    assert path.read_bytes() == before
    assert runtime.resolve_project_binding(root / ".qexp").machine_name == "g3"


def test_human_and_json_report_same_pending_names_and_phase(enrolled, capsys):
    runtime, root, binding = enrolled
    hold_reservation(runtime, binding)
    assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["project_id"] == binding.project_id
    assert result["shared_root"] == str(root / ".qexp")
    assert result["phase"] == "draining"
    assert result["cancellable"] is True
    assert (
        main(
            [
                "--machine-runtime-root",
                str(runtime.root),
                "project",
                "register",
                str(root),
                "--machine",
                "g2",
                "--replace-current",
                "--format=human",
            ]
        )
        == 0
    )
    human = capsys.readouterr().out
    for value in ("g3", "g2", "draining", "--replace-current", "--cancel-replacement"):
        assert value in human


@pytest.mark.parametrize(
    "effect", ["reservation", "old_retired", "target", "disable", "remove", "bind", "enable", "inventory"]
)
def test_failed_effect_directory_barrier_converges_without_new_generation(enrolled, monkeypatch, capsys, effect):
    import os
    from pathlib import Path

    runtime, root, old = enrolled
    shared = root / ".qexp"
    publish, sync = os.replace, os.fsync
    armed = []
    failed = []

    def observe(source, destination, *args, **kwargs):
        path = Path(destination)
        should_fail = False
        if effect == "reservation":
            should_fail = path == shared / "project/registration-reservations/g2.json"
        elif effect == "old_retired":
            should_fail = path == shared / "machines/g3/registration.json"
        elif effect == "target":
            should_fail = path == shared / "machines/g2/registration.json"
        elif effect == "inventory":
            should_fail = path == runtime.paths["inventory"] and '"g2"' in Path(source).read_text()
        elif path == runtime.paths["registry"]:
            bindings = json.loads(Path(source).read_text())["registry"]["bindings"]
            match = {
                "disable": any(b["machine_name"] == "g3" and not b["enabled"] for b in bindings),
                "remove": not bindings,
                "bind": any(b["machine_name"] == "g2" and not b["enabled"] for b in bindings),
                "enable": any(b["machine_name"] == "g2" and b["enabled"] for b in bindings),
            }
            should_fail = match.get(effect, False)
        result = publish(source, destination, *args, **kwargs)
        if should_fail and not failed:
            armed.append(path.parent)
        return result

    def fail_directory(descriptor):
        if armed and Path(os.readlink(f"/proc/self/fd/{descriptor}")) == armed[-1]:
            failed.append(armed.pop())
            raise OSError("injected directory durability failure")
        return sync(descriptor)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", observe)
        scoped.setattr(os, "fsync", fail_directory)
        assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 1
    capsys.readouterr()
    assert failed, effect
    document = json.loads((runtime.root / "project-registration-replacement.json").read_text())
    intent = next(value for value in document.values() if isinstance(value, dict) and "phase" in value)
    result = replacement(MachineRuntime(runtime.root), root)
    assert result["status"] == "replacement_completed"
    assert result["registration_generation"] == intent["target_generation"]
    current = runtime.resolve_project_binding(shared)
    assert current.machine_name == "g2" and current.enabled
    assert current.runtime_instance_id == old.runtime_instance_id


def test_interrupted_partition_removal_resumes_after_retirement(enrolled, monkeypatch, capsys):
    import shutil
    from pathlib import Path

    runtime, root, old = enrolled
    partition = runtime.project_paths(old.project_id)["root"]
    scratch = partition / "disposable-cache"
    scratch.mkdir(parents=True)
    (scratch / "entry").write_text("cached")
    remove = shutil.rmtree

    def partial(path, *args, **kwargs):
        if Path(path) == partition:
            remove(scratch)
            raise OSError("injected partial removal")
        return remove(path, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(shutil, "rmtree", partial)
        assert invoke(runtime, root, "--machine", "g2", "--replace-current") == 1
    capsys.readouterr()
    assert not scratch.exists()
    assert invoke(runtime, root, "--cancel-replacement") == 1
    capsys.readouterr()
    assert replacement(MachineRuntime(runtime.root), root)["status"] == "replacement_completed"


def interrupt_after_phase(runtime, root, monkeypatch, phase):
    from qqtools.plugins.qexp.agent import registration_replacement as transaction

    write = transaction._write_intent

    class Interrupted(BaseException):
        pass

    def interrupt(runtime, intent, next_phase):
        result = write(runtime, intent, next_phase)
        if next_phase == phase:
            raise Interrupted
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(transaction, "_write_intent", interrupt)
        with pytest.raises(Interrupted):
            replacement(runtime, root)
    return transaction.read_replacement_intent(runtime)


def test_expired_target_publication_renews_exact_generation(enrolled, monkeypatch):
    from qqtools.plugins.qexp.agent.registration import registration_state

    runtime, root, _ = enrolled
    intent = interrupt_after_phase(runtime, root, monkeypatch, "target_published")
    path = root / ".qexp/machines/g2/registration.json"
    value = json.loads(path.read_text())
    value["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00+00:00"
    atomic_replace(path, value)
    result = replacement(runtime, root)
    record = json.loads(path.read_text())["registration"]
    assert result["status"] == "replacement_completed"
    assert record["generation"] == intent["target_generation"]
    assert registration_state(record) == "eligible"


@pytest.mark.parametrize("phase", ["target_published", "binding_published", "completed"])
@pytest.mark.parametrize("damage", ["missing", "successor"])
def test_published_target_damage_preserves_intent(enrolled, monkeypatch, phase, damage):
    runtime, root, _ = enrolled
    interrupt_after_phase(runtime, root, monkeypatch, phase)
    path = root / ".qexp/machines/g2/registration.json"
    if damage == "missing":
        path.unlink()
    else:
        value = json.loads(path.read_text())
        value["registration"]["generation"] = "successor-generation"
        atomic_replace(path, value)
    with pytest.raises(RuntimeError, match="intent remains active"):
        replacement(runtime, root)
    assert (runtime.root / "project-registration-replacement.json").exists()
    if damage == "missing":
        assert not path.exists()
    else:
        assert json.loads(path.read_text())["registration"]["generation"] == "successor-generation"


def test_cancel_revalidates_shared_authority_at_restore(enrolled, monkeypatch):
    from qqtools.plugins.qexp.agent import registration_replacement as transaction

    runtime, root, old = enrolled
    hold_reservation(runtime, old)
    replacement(runtime, root)
    release = transaction._release_exact_reservation

    def takeover(*args, **kwargs):
        release(*args, **kwargs)
        path = root / ".qexp/machines/g3/registration.json"
        value = json.loads(path.read_text())
        value["registration"]["generation"] = "successor-generation"
        atomic_replace(path, value)

    monkeypatch.setattr(transaction, "_release_exact_reservation", takeover)
    with pytest.raises(RuntimeError, match="identity changed"):
        register_projects(runtime, [root], cancel_replacement=True)
    assert runtime.resolve_project_binding(root / ".qexp").enabled is False
    assert transaction.read_replacement_intent(runtime) is not None

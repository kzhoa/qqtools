"""Validate the failure boundary between authoritative writers and Idle Guard."""

import pytest

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.agent import idle_guard_worker
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.commands.group import create_group, group_control
from qqtools.plugins.qexp.runtime import project_activation
from qqtools.plugins.qexp.runtime.paths import group_path
from qqtools.plugins.qexp.runtime.store import read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("writer", ["group_resume", "submission"])
def test_commit_with_unfinished_checkpoint_cannot_reuse_idle_proof(tmp_path, monkeypatch, writer):
    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test", agent_mode="daemon")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "test")
    binding = runtime.add_binding(cfg.shared_root, "test")
    if writer == "group_resume":
        create_group(cfg, "jobs")
        group_control(cfg, "jobs", "pause")
    old, _ = idle_guard_worker._read_checkpoint(binding)
    replace = project_activation.atomic_replace
    reason = "group_dispatch_change" if writer == "group_resume" else "submission_commit"
    truth = []

    def fail_checkpoint(path, value):
        if (
            path == project_activation.activation_checkpoint_path(cfg.shared_root)
            and value["project_activation"]["reason"] == reason
        ):
            if writer == "group_resume":
                assert read_json(group_path(cfg.shared_root, "jobs"))["group"]["dispatch_state"] == "active"
            else:
                from qqtools.plugins.qexp.runtime.paths import shared_paths

                files = shared_paths(cfg.shared_root)["submissions"].glob("*.json")
                assert any(read_json(file)["submission"]["state"] == "committed" for file in files)
            with pytest.raises(RuntimeError, match="pending Project activation"):
                idle_guard_worker._read_checkpoint(binding)
            truth.append(True)
            raise OSError("checkpoint unavailable after business commit")
        return replace(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(project_activation, "atomic_replace", fail_checkpoint)
        if writer == "group_resume":
            with pytest.raises(OSError, match="checkpoint unavailable after business commit"):
                group_control(cfg, "jobs", "resume")
        else:
            # submit reconciles ambiguous success from committed business truth.
            submit(cfg, ["true"], working_dir=tmp_path)
    assert truth
    project_activation.recover_project_activation(cfg.shared_root)
    assert idle_guard_worker._read_checkpoint(binding)[0] != old


def test_explicit_writer_cutover_only_affects_selected_project(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import layout
    from qqtools.plugins.qexp.cli.entrypoint import main
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test")
    configs = []
    bindings = []
    floor = layout.IDLE_JOB_ACTIVATION_WRITERS_CAPABILITY
    for name in ("a", "b"):
        cfg = init_shared_root(tmp_path / name / ".qexp", "test")
        configs.append(cfg)
        bindings.append(runtime.add_binding(cfg.shared_root, "test"))
        assert layout.is_idle_job_activation_writers_root(cfg)
        schema = cfg.shared_root / "schema" / "version.json"
        old = read_json(schema)
        old["schema"]["required_capabilities"].remove(floor)
        atomic_replace(schema, old)
        with pytest.raises((ValueError, RuntimeError), match="writer"):
            idle_guard_worker._read_checkpoint(bindings[-1])
        assert submit(cfg, ["true"], working_dir=tmp_path).task_id

    cli = [
        "--machine-runtime-root",
        str(runtime.root),
        "admin",
        "upgrade",
        "advance",
        "--enable-idle-job-activation-writers",
    ]
    assert main(cli) != 0
    checkpoint = project_activation.read_project_activation(configs[0].shared_root)
    assert main([*cli, "--project", bindings[0].project_id]) == 0
    switched = project_activation.read_project_activation(configs[0].shared_root)
    assert switched != checkpoint
    assert switched["project_activation"]["reason"] == "idle_job_writer_cutover"
    assert main([*cli, "--project", bindings[0].project_id]) == 0
    assert project_activation.read_project_activation(configs[0].shared_root) == switched
    assert layout.is_idle_job_activation_writers_root(configs[0])
    assert not layout.is_idle_job_activation_writers_root(configs[1])
    idle_guard_worker._read_checkpoint(bindings[0])
    # Exercise the normal business entry point with the previous reader's
    # capability set. A process on that version cannot publish on the new root.
    monkeypatch.setattr(layout, "SUPPORTED_REQUIRED_CAPABILITIES", layout.SUPPORTED_REQUIRED_CAPABILITIES - {floor})
    with pytest.raises(RuntimeError, match="unsupported capabilities"):
        submit(configs[0], ["true"], working_dir=tmp_path)
    assert submit(configs[1], ["true"], working_dir=tmp_path).task_id


def test_cutover_rejects_replaced_project_identity(tmp_path):
    from qqtools.plugins.qexp import layout
    from qqtools.plugins.qexp.cli.entrypoint import main
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    initialize_machine(runtime, "test")
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "test")
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["required_capabilities"].remove(layout.IDLE_JOB_ACTIVATION_WRITERS_CAPABILITY)
    atomic_replace(schema_path, schema)
    identity_path = cfg.shared_root / "project" / "identity.json"
    identity = read_json(identity_path)
    identity["project"]["project_id"] = "deadcafe"
    atomic_replace(identity_path, identity)
    assert (
        main(
            [
                "--machine-runtime-root",
                str(runtime.root),
                "admin",
                "upgrade",
                "advance",
                "--project",
                binding.project_id,
                "--enable-idle-job-activation-writers",
            ]
        )
        != 0
    )
    assert read_json(schema_path) == schema

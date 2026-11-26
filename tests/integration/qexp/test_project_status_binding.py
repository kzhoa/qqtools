"""Exercise bounded status for the selected Project-local registration."""

import json

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.setup import (
    initialize_machine,
    initialize_project,
    register_projects,
    set_project_enablement,
)
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands import status
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json

pytestmark = pytest.mark.integration


@pytest.fixture
def projects(tmp_path):
    runtime = MachineRuntime(tmp_path / "machine runtime")
    initialize_machine(runtime, "machine-default")
    root, other = tmp_path / "project one", tmp_path / "project two"
    for project, machine in ((root, "g3"), (other, "g2")):
        initialize_project(project)
        assert register_projects(runtime, [project], machine_name=machine)["projects"][0]["status"] == "registered"
    return runtime, root, other


def status_args(runtime, root):
    return ["--machine-runtime-root", str(runtime.root), "--project", str(root), "status"]


def test_status_observes_exact_local_registration_without_enumerating_peers(projects, capsys, monkeypatch):
    runtime, root, other = projects
    registration = root / ".qexp" / "machines" / "g3" / "registration.json"
    binding = runtime.resolve_project_binding(root / ".qexp")
    before = (runtime.paths["registry"].read_bytes(), registration.read_bytes())
    original_read = status._ReadBudget.read_json
    paths = []

    def track(reader, path, **kwargs):
        paths.append(path)
        return original_read(reader, path, **kwargs)

    monkeypatch.setattr(status._ReadBudget, "read_json", track)
    assert main([*status_args(runtime, root), "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["local_participation"]["binding"] == {
        "machine_name": "g3",
        "registration_generation": binding.registration_generation,
        "enabled": True,
        "write_eligibility": "eligible",
    }
    assert result["local_participation"]["binding_reason"] is None
    assert result["budget"]["read_attempts"] <= result["budget"]["max_read_attempts"]
    assert registration in paths
    assert all(not path.is_relative_to(other) for path in paths)
    assert (runtime.paths["registry"].read_bytes(), registration.read_bytes()) == before

    assert main(status_args(runtime, root)) == 0
    output = capsys.readouterr().out
    assert "Local binding: g3" in output
    assert "Project takeover preview help" in output
    assert "--machine=g2" not in output
    assert "--machine g2" not in output


@pytest.mark.parametrize("scenario", ["expired", "wrong_generation", "wrong_runtime"])
def test_status_distinguishes_expiry_and_unverified_binding(projects, capsys, scenario):
    runtime, root, _ = projects
    path = root / ".qexp" / "machines" / "g3" / "registration.json"
    value = read_json(path)
    record = value["registration"]
    if scenario == "expired":
        record["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    elif scenario == "wrong_generation":
        record["generation"] = "another-generation"
    else:
        record["runtime_instance_id"] = "another-runtime"
    atomic_replace(path, value)
    before = path.read_bytes()

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)["local_participation"]
    assert result["state"] == "registered"
    if scenario == "expired":
        assert result["binding"]["machine_name"] == "g3"
        assert result["binding"]["write_eligibility"] == "expired"
    else:
        assert result["binding"] is None
        assert result["binding_reason"]
    assert path.read_bytes() == before


def test_status_uses_persisted_project_id(projects, capsys, tmp_path):
    runtime, _, _ = projects
    root = tmp_path / "project with a stable identity"
    initialize_project(root)
    identity_path = root / ".qexp" / "project" / "identity.json"
    identity = read_json(identity_path)
    identity["project"]["project_id"] = "retained project id"
    atomic_replace(identity_path, identity)
    assert register_projects(runtime, [root], machine_name="g4")["projects"][0]["status"] == "registered"

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["project"]["project_id"] == "retained project id"
    assert result["local_participation"]["binding"]["machine_name"] == "g4"
    assert result["scheduler_diagnostics"]["reason"] == "project_id_unavailable"
    assert all("--project-id" not in command for command in result["next_actions"])


def test_status_cannot_verify_registration_in_a_redirected_directory(projects, capsys, tmp_path):
    runtime, root, _ = projects
    machine = root / ".qexp" / "machines" / "g3"
    redirected = tmp_path / "redirected"
    machine.rename(redirected)
    machine.symlink_to(redirected, target_is_directory=True)

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)["local_participation"]
    assert result["binding"] is None
    assert result["binding_reason"]


@pytest.mark.parametrize("axis", ["project_id", "shared_root"])
def test_status_cannot_verify_an_ambiguous_registry_match(projects, capsys, axis):
    runtime, root, other = projects
    path = runtime.paths["registry"]
    roster = read_json(path)
    current, peer = (
        next(binding for binding in roster["registry"]["bindings"] if binding["shared_root"] == str(project / ".qexp"))
        for project in (root, other)
    )
    peer[axis] = current[axis]
    atomic_replace(path, roster)

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    participation = json.loads(capsys.readouterr().out)["local_participation"]
    assert participation["state"] == "registered"
    assert participation["binding"] is None


def test_status_honors_the_registration_input_size_limit(projects, capsys):
    runtime, root, _ = projects
    path = root / ".qexp" / "machines" / "g3" / "registration.json"
    path.write_bytes(path.read_bytes() + b" " * (64 * 1024))

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    assert json.loads(capsys.readouterr().out)["local_participation"]["binding"] is None


def test_status_reports_disabled_source_protocol_binding(projects, capsys):
    runtime, root, _ = projects
    assert set_project_enablement(runtime, root, False)["status"] == "committed"
    path = root / ".qexp" / "machines" / "g3" / "registration.json"
    document = read_json(path)
    record = document["registration"]
    record["version"] = 1
    record["protocol_version"] = 1
    record.pop("registration_protocol")
    record.pop("recovery_protocol")
    atomic_replace(path, document)
    before = path.read_bytes()

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    assert json.loads(capsys.readouterr().out)["local_participation"]["binding"]["enabled"] is False
    assert path.read_bytes() == before


def test_unbound_status_bypasses_execution_context_lookup(projects, capsys, monkeypatch, tmp_path):
    _, root, _ = projects
    runtime = MachineRuntime(tmp_path / "unbound")

    def forbid_lookup(*args, **kwargs):
        raise AssertionError("status attempted execution lookup and peer machine enumeration")

    monkeypatch.setattr(MachineRuntime, "verified_execution_context", forbid_lookup)

    assert main([*status_args(runtime, root), "--format=json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["local_participation"]["state"] == "unregistered"
    assert result["takeover_preview_help"]
    assert not runtime.root.exists()

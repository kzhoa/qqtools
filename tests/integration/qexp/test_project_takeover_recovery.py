"""Exercise bounded recovery-ledger observations through takeover preview."""

import json
import os

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.setup import initialize_machine, initialize_project, register_projects
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.responsibility import responsibility_root
from qqtools.plugins.qexp.runtime.responsibility_store import Ledger, identity_key
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = pytest.mark.integration


@pytest.fixture
def operation(tmp_path):
    runtime = MachineRuntime(tmp_path / "runtime")
    previous = MachineRuntime(tmp_path / "previous")
    initialize_machine(runtime, "g3")
    initialize_machine(previous, "g2")
    project = tmp_path / "project"
    initialize_project(project)
    for owner in (runtime, previous):
        assert register_projects(owner, [project])["projects"][0]["status"] == "registered"
    target = project / ".qexp" / "machines" / "g2" / "registration.json"
    document = json.loads(target.read_bytes())
    document["registration"]["eligibility_expires_at"] = "2000-01-01T00:00:00Z"
    atomic_replace(target, document)
    return runtime, project, runtime.resolve_project_binding(project / ".qexp")


def invoke(operation, capsys, cursor=None):
    runtime, project, _ = operation
    argv = [
        "--machine-runtime-root",
        str(runtime.root),
        "project",
        "takeover",
        str(project),
        "--machine",
        "g2",
        "--plan",
        "--format=json",
    ]
    if cursor is not None:
        argv.extend(("--recovery-cursor", cursor))
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def paths_and_ledger(operation):
    runtime, _, binding = operation
    paths = runtime.project_paths(binding.project_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    return paths, Ledger.create(responsibility_root(paths["root"]))


def snapshot(root):
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_departing_ledger_pages_are_read_only_and_keep_unidentified_records(operation, capsys):
    paths, ledger = paths_and_ledger(operation)
    attempt_names = {f"task-{index}-attempt-1" for index in range(29)}
    for attempt_id in sorted(attempt_names):
        ledger.publish(attempt_id, {"task_id": attempt_id.removesuffix("-attempt-1"), "attempt_number": 1})
    ledger.capture_local("control-plane", {"task_id": None, "attempt_number": None})
    ledger.capture_local("unidentified", {"task_id": None, "attempt_number": None})
    atomic_replace(
        paths["authority_diagnostics"] / "control-plane.json",
        {
            "authority_diagnostic": {
                "attempt_id": "control-plane",
                "reason": "cleanup_replay_unavailable",
                "at": "2026-01-01T00:00:00Z",
                "error_type": "RuntimeError",
                "error": "private diagnostic detail",
            }
        },
    )
    before = snapshot(paths["root"])

    first_code, first_result = invoke(operation, capsys)
    first = first_result["departing_recovery"]
    assert first_code == 0
    assert first["status"] == "observed" and first["count"] == 31
    assert len(first["records"]) == 25
    assert first["next_cursor"]

    second_code, second_result = invoke(operation, capsys, cursor=first["next_cursor"])
    second = second_result["departing_recovery"]
    assert second_code == 0
    assert second["count"] == first["count"]
    assert len(second["records"]) == 6
    assert second["next_cursor"] is None
    records = {record["identity"]: record for record in [*first["records"], *second["records"]]}
    assert len(records) == 31
    assert {identity for identity, record in records.items() if record["kind"] == "attempt"} == attempt_names
    assert records["control-plane"]["kind"] == "control_plane_diagnostic"
    assert records["control-plane"]["task_id"] is None
    assert records["unidentified"]["kind"] == "unidentified"
    assert "private diagnostic detail" not in json.dumps(records)
    assert snapshot(paths["root"]) == before


@pytest.mark.parametrize("is_initializing", [False, True])
def test_departing_ledger_with_pending_transaction_is_unavailable_without_replay(operation, capsys, is_initializing):
    paths, ledger = paths_and_ledger(operation)
    identity = "task-attempt-1"
    ledger.publish(identity, {"task_id": "task", "attempt_number": 1})
    bucket = int(identity_key(identity)[0], 16)
    pending = ledger.root / "initializing" if is_initializing else ledger.root / str(bucket) / "pending"
    pending.write_bytes(b"interrupted transaction")
    before = snapshot(paths["root"])

    code, result = invoke(operation, capsys)

    assert code == 1
    assert result["departing_recovery"]["status"] == "unavailable"
    assert result["departing_recovery"]["count"] is None
    assert "departing_recovery_unavailable" in {blocker["code"] for blocker in result["blockers"]}
    assert snapshot(paths["root"]) == before


@pytest.mark.parametrize("has_sibling", [False, True])
def test_control_plane_name_alone_does_not_classify_ambiguous_evidence(operation, capsys, has_sibling):
    paths, ledger = paths_and_ledger(operation)
    ledger.capture_local("control-plane", {"task_id": None, "attempt_number": None})
    diagnostic = {"attempt_id": "control-plane", "reason": "cleanup_replay_unavailable"}
    if has_sibling:
        diagnostic.update(at="2026-01-01T00:00:00Z", error_type="RuntimeError", error="unavailable")
        atomic_replace(
            paths["launch_intents"] / "control-plane.json", {"launch_intent": {"attempt_id": "control-plane"}}
        )
    else:
        diagnostic["pid"] = 1234
    atomic_replace(paths["authority_diagnostics"] / "control-plane.json", {"authority_diagnostic": diagnostic})

    code, result = invoke(operation, capsys)

    assert code == 0
    records = result["departing_recovery"]["records"]
    assert len(records) == 1
    assert records[0]["kind"] == "unidentified"


def test_departing_ledger_rejects_a_cursor_without_a_ledger(operation, capsys):
    code, result = invoke(operation, capsys, cursor="invalid")

    assert code == 2
    assert result["error"]["code"] == "invalid_argument"


@pytest.mark.parametrize("component", ["marker", "bucket", "lock", "header", "page", "entry"])
def test_departing_ledger_rejects_redirected_storage(operation, capsys, tmp_path, component):
    _, ledger = paths_and_ledger(operation)
    identity = "local-task-attempt-1"
    ledger.publish(identity, {"task_id": "local-task", "attempt_number": 1})
    key = identity_key(identity)
    bucket = ledger.root / str(int(key[0], 16))
    paths = {
        "marker": ledger.root / "marker",
        "bucket": bucket,
        "lock": bucket / "lock",
        "header": bucket / "header",
        "page": bucket / "p0",
        "entry": bucket / f"e{key}",
    }
    selected = paths[component]
    redirected = tmp_path / f"other-project-{component}"
    selected.rename(redirected)
    selected.symlink_to(redirected)
    before = snapshot(ledger.root)

    code, result = invoke(operation, capsys)

    assert code == 1
    recovery = result["departing_recovery"]
    assert recovery["status"] == "unavailable"
    assert recovery["records"] == []
    assert snapshot(ledger.root) == before


def test_departing_ledger_rejects_nonregular_member_without_opening_it_for_data(operation, capsys):
    _, ledger = paths_and_ledger(operation)
    identity = "local-task-attempt-1"
    ledger.publish(identity, {"task_id": "local-task", "attempt_number": 1})
    key = identity_key(identity)
    path = ledger.root / str(int(key[0], 16)) / f"e{key}"
    path.unlink()
    os.mkfifo(path)

    code, result = invoke(operation, capsys)

    assert code == 1
    assert result["departing_recovery"]["status"] == "unavailable"
    assert path.is_fifo()


def test_departing_ledger_carries_bucket_capture_uncertainty(operation, capsys):
    _, ledger = paths_and_ledger(operation)
    identity = "control-plane"
    ledger.capture_local(identity, {"task_id": None, "attempt_number": None})
    header_path = ledger.root / str(int(identity_key(identity)[0], 16)) / "header"
    header = json.loads(header_path.read_bytes())
    header["writer_capture_incomplete"] = True
    atomic_replace(header_path, header)

    code, result = invoke(operation, capsys)

    assert code == 1
    assert result["departing_recovery"]["status"] == "unavailable"
    assert header_path.read_bytes() and json.loads(header_path.read_bytes()) == header


def test_recovery_cursor_cannot_be_reused_after_binding_generation_changes(operation, capsys):
    runtime, project, binding = operation
    _, ledger = paths_and_ledger(operation)
    for index in range(26):
        ledger.publish(f"task-{index}-attempt-1", {"task_id": f"task-{index}", "attempt_number": 1})
    code, result = invoke(operation, capsys)
    assert code == 0
    cursor = result["departing_recovery"]["next_cursor"]
    assert cursor
    registry_path = runtime.paths["registry"]
    registry = json.loads(registry_path.read_bytes())
    registry["registry"]["bindings"][0]["registration_generation"] = "updated"
    atomic_replace(registry_path, registry)
    record_path = project / ".qexp" / "machines" / binding.machine_name / "registration.json"
    registration = json.loads(record_path.read_bytes())
    registration["registration"]["generation"] = "updated"
    atomic_replace(record_path, registration)

    code, result = invoke(operation, capsys, cursor=cursor)

    assert code == 2
    assert "stale" in result["error"]["message"]


@pytest.mark.parametrize("variant", ["extra-ledger-field", "missing-ledger-field", "non-utc-observation"])
def test_control_plane_diagnostic_requires_exact_producer_evidence(operation, capsys, variant):
    paths, ledger = paths_and_ledger(operation)
    locator = {"task_id": None, "attempt_number": None}
    diagnostic = {
        "attempt_id": "control-plane",
        "reason": "cleanup_replay_unavailable",
        "at": "2026-01-01T00:00:00Z",
        "error_type": "RuntimeError",
        "error": "unavailable",
    }
    if variant == "extra-ledger-field":
        locator["pid"] = 1234
    elif variant == "missing-ledger-field":
        locator.pop("attempt_number")
    else:
        diagnostic["at"] = "2026-01-01T05:00:00+05:00"
    ledger.publish("control-plane", locator)
    atomic_replace(paths["authority_diagnostics"] / "control-plane.json", {"authority_diagnostic": diagnostic})

    code, result = invoke(operation, capsys)

    assert code == 0
    assert result["departing_recovery"]["records"][0]["kind"] == "unidentified"

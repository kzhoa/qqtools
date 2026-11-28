"""Fixtures for the version-1 scheduler diagnostic publication limit."""

import time
from pathlib import Path

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.scheduler_diagnostics import MAX_ACTIVE_RECORDS, SchedulerDiagnosticStore
from qqtools.plugins.qexp.runtime.records import utc_now
from qqtools.plugins.qexp.runtime.store import atomic_replace, json_encoded_size, read_json


def overdue_identity(runtime: MachineRuntime, project_id: str, generation: str, request: int) -> dict[str, object]:
    return {
        "producer": "control_plane_health",
        "reason_code": "project_snapshot_overdue",
        "component": "control_plane",
        "stage": "snapshot_publication",
        "check": "progress",
        "scope_type": "project_binding",
        "runtime_id": runtime.instance_id,
        "project_id": project_id,
        "registration_generation": generation,
        "operation_kind": "machine_snapshot_publish",
        "target_id": "predecessor",
        "executor_epoch": "former-executor",
        "registry_revision": 1,
        "request_id": f"request-{request:04d}",
    }


def seed_over_budget_store(runtime: MachineRuntime, project_id: str, generation: str) -> None:
    store = SchedulerDiagnosticStore(runtime)
    observed_at = utc_now()
    identity = overdue_identity(runtime, "historical-project", "generation", 0)
    first = store.observe_finding(
        identity=identity, severity="fault", source_revision={"registry_revision": 1}, observed_at=observed_at
    )
    assert store.resolve_finding(
        identity=identity,
        episode=first["episode"],
        observation_revision=first["observation_revision"],
        source_revision=first["source_revision"],
        resolved_at=observed_at,
    )
    index_path = store.paths["scheduler_diagnostics_history_index"]
    index = read_json(index_path)
    history = index["diagnostics"]
    segments_dir = store.paths["scheduler_diagnostics_history_segments"]
    segment = read_json(segments_dir / "1-1.json")
    original_entry = segment["diagnostics"]["entries"][0]
    index_template = history["segments"][0]
    # Each historical resolution had its own immutable segment. Reproduce the
    # old layout in one fixture pass rather than running 500 transactions.
    for sequence in range(2, 502):
        name = f"{sequence}-{sequence}.json"
        envelope = {
            "diagnostics": {
                **segment["diagnostics"],
                "entries": [{**original_entry, "sequence": sequence, "episode": f"{sequence:032x}"}],
            }
        }
        path = segments_dir / name
        atomic_replace(path, envelope)
        size = json_encoded_size(envelope)
        assert size == path.stat().st_size
        history["segments"].append(
            {**index_template, "name": name, "first_sequence": sequence, "last_sequence": sequence, "byte_count": size}
        )
        history["byte_count"] += size
    history.update(max_sequence=501, retained_max_sequence=501, entry_count=501)
    atomic_replace(index_path, index)
    metadata_path = store.paths["scheduler_diagnostics_metadata"]
    metadata = read_json(metadata_path)
    metadata["diagnostics"].update(next_history_sequence=502, retained_max_sequence=501)
    atomic_replace(metadata_path, metadata)
    for request in range(MAX_ACTIVE_RECORDS):
        result = store.observe_finding(
            identity=overdue_identity(runtime, project_id, generation, request),
            severity="fault",
            source_revision={"registry_revision": 1},
            details={"progress": "started"},
            observed_at=observed_at,
        )
        assert result.get("episode"), result
    assert store.active_view()["total"] == MAX_ACTIVE_RECORDS


def diagnostic_tree(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def first_project_history(store: SchedulerDiagnosticStore, project_id: str) -> dict:
    cursor = None
    deadline = time.monotonic() + 5
    for _page in range(512):
        result = store.history_view(project_id=project_id, limit=1, cursor=cursor)
        while result["reason"] == "store_unavailable" and time.monotonic() < deadline:
            time.sleep(0.02)
            result = store.history_view(project_id=project_id, limit=1, cursor=cursor)
        if result["items"]:
            return result["items"][0]
        assert not result["end_of_capture"], result
        cursor = result["next_cursor"]
        assert cursor is not None, result
    raise AssertionError("retained history traversal exceeded the fixture's segment count")

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qqtools.plugins.qexp.agent import scheduler_diagnostics as diagnostic_module
from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.dispatch_loop import (
    PrimaryDemandProbe,
    _enablement_reconciliation_probe,
    _scheduler_diagnostic_probe,
    _termination_convergence_probe,
)
from qqtools.plugins.qexp.agent.inventory import InventoryReconciliation, ProjectInventoryEntry
from qqtools.plugins.qexp.agent.scheduler_diagnostics import SchedulerDiagnosticStore


@pytest.fixture(autouse=True)
def _deterministic_publication_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep store-contract tests independent of host filesystem latency."""
    monkeypatch.setattr(diagnostic_module, "time", SimpleNamespace(monotonic_ns=lambda: 0))


def _identity(index: int = 1, *, project_id: str = "project-a") -> dict[str, object]:
    return {
        "producer": "primary_probe",
        "reason_code": "ready_index_unreadable",
        "component": "scheduler",
        "stage": "admission",
        "check": "primary_demand",
        "scope_type": "project_route",
        "runtime_id": "a" * 64,
        "project_id": project_id,
        "registration_generation": f"registration-{index}",
        "resource_lane": "gpu",
        "route_scope": "shared",
    }


def _observe(store: SchedulerDiagnosticStore, index: int = 1, *, project_id: str = "project-a") -> dict:
    return store.observe_finding(
        identity=_identity(index, project_id=project_id),
        severity="fault",
        source_revision={"registry_revision": index},
        details={"exception_type": "OSError", "errno": 5},
        observed_at=f"2026-09-28T00:00:{index:02d}Z",
    )


def test_unresolved_primary_probe_records_borrow_blocking_decision_without_duplicating_finding(
    tmp_path: Path,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = ProjectBinding(
        project_id="project-a",
        shared_root=tmp_path / "project" / ".qexp",
        machine_name="gpu-1",
        registration_generation="registration-1",
        runtime_instance_id=runtime.instance_id,
        runtime_root=str(runtime.root),
    )
    probe = PrimaryDemandProbe(
        "unresolved",
        (
            {
                "reason": "ready_index_unreadable",
                "project_id": binding.project_id,
                "exception_type": "OSError",
            },
        ),
    )

    diagnostic = _scheduler_diagnostic_probe(
        runtime,
        lane="gpu",
        probe=probe,
        registry_revision=7,
        bindings={binding.project_id: binding},
        covered_project_ids={binding.project_id},
        available_capacity=2,
        borrow_denied=True,
    )

    assert diagnostic["identity"]["reason_code"] == "borrow_blocked_unresolved_primary"
    assert diagnostic["identity"]["resource_lane"] == "gpu"
    assert diagnostic["source_revision"] == {"registry_revision": 7}
    assert diagnostic["coverage"] == "incomplete"
    assert [item["identity"]["reason_code"] for item in diagnostic["findings"]] == ["ready_index_unreadable"]


def test_termination_convergence_finding_is_bounded_and_resolves(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = ProjectBinding(
        project_id="project-a",
        shared_root=tmp_path / "project" / ".qexp",
        machine_name="gpu-1",
        registration_generation="registration-1",
        runtime_instance_id=runtime.instance_id,
        runtime_root=str(runtime.root),
    )

    class Coordinator:
        findings: tuple[dict[str, object], ...] = (
            {
                "project_id": binding.project_id,
                "registration_generation": binding.registration_generation,
                "attempt_id": "task-a-attempt-1",
                "state": "invalid",
                "reason": "termination_exit_observation_mismatch",
            },
        )

        def read_termination_convergence_diagnostics(self, *_args, **_kwargs):
            return self.findings

    coordinator = Coordinator()
    blocked_probe = _termination_convergence_probe(runtime, coordinator, [binding], 9)
    finding = blocked_probe["findings"][0]

    assert finding["identity"] == {
        "producer": "termination_convergence",
        "reason_code": "termination_exit_observation_mismatch",
        "component": "scheduler",
        "stage": "reconciliation",
        "check": "termination_convergence",
        "scope_type": "project_attempt",
        "runtime_id": runtime.instance_id,
        "project_id": binding.project_id,
        "registration_generation": binding.registration_generation,
        "attempt_id": "task-a-attempt-1",
        "registry_revision": 9,
    }
    assert finding["details"] == {
        "diagnostic_code": "termination_exit_observation_mismatch",
        "progress": "invalid",
    }

    store = SchedulerDiagnosticStore(runtime)
    store.publish_cycle({}, {}, (blocked_probe,), {}, observed_at="2026-09-28T00:00:00Z")
    active = store.active_view(producer="termination_convergence")
    assert active["total"] == 1
    assert active["items"][0]["identity"]["attempt_id"] == "task-a-attempt-1"

    coordinator.findings = ()
    converged_probe = _termination_convergence_probe(runtime, coordinator, [binding], 9)
    assert store.reconcile_cycle((converged_probe,), resolved_at="2026-09-28T00:01:00Z")
    assert store.active_view(producer="termination_convergence")["items"] == []


def test_enablement_reconciliation_finding_coalesces_and_resolves_after_exact_convergence(
    tmp_path: Path,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    entry = ProjectInventoryEntry("project-a", tmp_path / "project" / ".qexp", enabled=True)
    binding = ProjectBinding(
        project_id=entry.project_id,
        shared_root=entry.shared_root,
        machine_name="gpu-1",
        enabled=False,
        registration_generation="registration-1",
        runtime_instance_id=runtime.instance_id,
        runtime_root=str(runtime.root),
    )
    incomplete = InventoryReconciliation(
        registry_revision=7,
        inventory_revision=3,
        entries=(entry,),
        bindings=(binding,),
        blockers=("inventory_mirror_incomplete",),
        converged=False,
        changed=False,
    )
    incomplete_probe = _enablement_reconciliation_probe(runtime, incomplete)
    store = SchedulerDiagnosticStore(runtime)

    store.publish_cycle({}, {}, (incomplete_probe,), {}, observed_at="2026-09-28T00:00:00Z")

    active = store.active_view(producer="enablement_reconciliation")
    assert active["total"] == 1
    assert active["items"][0]["identity"]["reason_code"] == "inventory_mirror_incomplete"
    assert active["items"][0]["details"]["progress"] == "mirror_write_pending"

    mirrored_entry = ProjectInventoryEntry(entry.project_id, entry.shared_root, enabled=False)
    converged = InventoryReconciliation(
        registry_revision=7,
        inventory_revision=4,
        entries=(mirrored_entry,),
        bindings=(binding,),
        blockers=(),
        converged=True,
        changed=True,
    )
    converged_probe = _enablement_reconciliation_probe(runtime, converged)
    store.publish_cycle({}, {}, (converged_probe,), {}, observed_at="2026-09-28T00:01:00Z")

    assert store.reconcile_cycle((converged_probe,), resolved_at="2026-09-28T00:01:01Z")
    assert store.active_view(producer="enablement_reconciliation")["items"] == []


def test_unknown_enablement_registry_is_machine_scoped_and_resolves_after_validated_read(
    tmp_path: Path,
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    unknown_probe = _enablement_reconciliation_probe(runtime, None)
    store = SchedulerDiagnosticStore(runtime)
    store.publish_cycle({}, {}, (unknown_probe,), {}, observed_at="2026-09-28T00:00:00Z")

    active = store.active_view(producer="enablement_reconciliation")
    assert active["total"] == 1
    assert active["items"][0]["identity"]["reason_code"] == "registry_enablement_unknown"

    converged = InventoryReconciliation(1, 1, (), (), (), True, False)
    converged_probe = _enablement_reconciliation_probe(runtime, converged)
    assert store.reconcile_cycle((converged_probe,), resolved_at="2026-09-28T00:01:00Z")


def test_missing_inventory_entry_is_a_blocked_binding_finding_not_convergence(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    binding = ProjectBinding(
        project_id="project-a",
        shared_root=tmp_path / "project" / ".qexp",
        machine_name="gpu-1",
        registration_generation="registration-1",
        runtime_instance_id=runtime.instance_id,
        runtime_root=str(runtime.root),
    )
    reconciliation = InventoryReconciliation(
        registry_revision=8,
        inventory_revision=4,
        entries=(),
        bindings=(binding,),
        blockers=("inventory_entry_missing",),
        converged=False,
        changed=False,
    )

    probe = _enablement_reconciliation_probe(runtime, reconciliation)

    assert probe["outcome"] == "blocked"
    assert probe["identity"]["reason_code"] == "enablement_mirror_diverged"
    assert len(probe["findings"]) == 1
    identity = probe["findings"][0]["identity"]
    assert identity["project_id"] == binding.project_id
    assert identity["registration_generation"] == binding.registration_generation
    assert probe["findings"][0]["details"]["progress"] == "blocked"


def test_missing_store_reads_are_unknown_and_do_not_create_runtime(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)

    active = store.active_view()
    history = store.history_view()
    summary = store.summary_view()

    assert active == {
        "schema_version": 1,
        "action": "diagnostics_active",
        "machine_runtime_root": str(runtime.root),
        "project_id": None,
        "coverage": "unknown",
        "observed_at": None,
        "snapshot_revision": None,
        "total": None,
        "aggregates": {},
        "items": [],
        "truncated": False,
        "reason": "store_not_initialized",
    }
    assert history["coverage"] == "unknown"
    assert history["reason"] == "store_not_initialized"
    assert history["items"] == []
    assert history["end_of_capture"] is False
    assert history["next_cursor"] is None
    assert summary["status"] == "unavailable"
    assert summary["coverage"] == "unknown"
    assert not runtime.root.exists()


def test_finding_sanitizer_never_persists_raw_exception_text(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    secret = "/private/token=super-secret"

    record = store.observe_finding(
        identity=_identity(),
        severity="fault",
        source_revision={"registry_revision": 1},
        details={
            "exception_type": "OSError",
            "errno": 13,
            "message": secret,
            "path": "/private",
            "command": "curl --token super-secret",
            "unknown": secret,
        },
        observed_at="2026-09-28T00:00:00Z",
    )

    encoded_store = b"".join(
        path.read_bytes() for path in (runtime.root / "diagnostics" / "scheduler-v1").rglob("*") if path.is_file()
    )
    assert secret.encode() not in encoded_store
    assert b"curl" not in encoded_store
    assert record["details"] == {"errno": 13, "exception_type": "OSError"}
    assert record["details_omitted"] is True


def test_repeated_finding_coalesces_and_stale_resolution_cannot_clear_it(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    first = _observe(store)
    second = _observe(store)

    assert second["episode"] == first["episode"]
    assert second["observation_revision"] == first["observation_revision"] + 1
    assert second["observation_count"] == 2
    assert second["first_observed_at"] == first["first_observed_at"]
    assert not store.resolve_finding(
        identity=_identity(),
        episode=first["episode"],
        observation_revision=first["observation_revision"],
        source_revision={"registry_revision": 1},
        resolved_at="2026-09-28T00:01:00Z",
    )
    assert store.active_view()["total"] == 1


def test_exact_resolution_moves_finding_to_immutable_history(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    finding = _observe(store)

    assert store.resolve_finding(
        identity=_identity(),
        episode=finding["episode"],
        observation_revision=finding["observation_revision"],
        source_revision={"registry_revision": 1},
        resolved_at="2026-09-28T00:01:00Z",
    )

    assert store.active_view()["items"] == []
    history = store.history_view()
    assert history["coverage"] == "complete"
    assert history["end_of_capture"] is True
    assert history["next_cursor"] is None
    assert len(history["items"]) == 1
    assert history["items"][0]["identity"] == _identity()
    assert history["items"][0]["episode"] == finding["episode"]
    segment = next((runtime.root / "diagnostics" / "scheduler-v1" / "history" / "segments").glob("*.json"))
    before = segment.read_bytes()
    _observe(store, 2)
    assert segment.read_bytes() == before


def test_history_cursor_captures_max_sequence_across_later_appends(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    for index in (1, 2):
        finding = _observe(store, index)
        assert store.resolve_finding(
            identity=_identity(index),
            episode=finding["episode"],
            observation_revision=finding["observation_revision"],
            source_revision={"registry_revision": index},
            resolved_at=f"2026-09-28T00:01:{index:02d}Z",
        )

    first_page = store.history_view(limit=1)
    assert first_page["end_of_capture"] is False
    assert first_page["next_cursor"]
    captured_max = first_page["captured_max_sequence"]

    later = _observe(store, 3)
    assert store.resolve_finding(
        identity=_identity(3),
        episode=later["episode"],
        observation_revision=later["observation_revision"],
        source_revision={"registry_revision": 3},
        resolved_at="2026-09-28T00:01:03Z",
    )

    second_page = store.history_view(cursor=first_page["next_cursor"], limit=32)
    assert second_page["captured_max_sequence"] == captured_max
    assert second_page["end_of_capture"] is True
    assert [item["sequence"] for item in first_page["items"] + second_page["items"]] == [1, 2]


def test_history_cursor_rejects_project_filter_mismatch_before_store_read(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    finding = _observe(store)
    assert store.resolve_finding(
        identity=_identity(),
        episode=finding["episode"],
        observation_revision=finding["observation_revision"],
        source_revision={"registry_revision": 1},
    )
    page = store.history_view(project_id="project-a", limit=1)
    if page["next_cursor"] is None:
        second = _observe(store, 2, project_id="project-a")
        assert store.resolve_finding(
            identity=_identity(2, project_id="project-a"),
            episode=second["episode"],
            observation_revision=second["observation_revision"],
            source_revision={"registry_revision": 2},
        )
        page = store.history_view(project_id="project-a", limit=1)

    with pytest.raises(ValueError, match="project"):
        store.history_view(project_id="project-b", cursor=page["next_cursor"])


def test_history_cursor_rejects_impossible_capture_claim(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    finding = _observe(store)
    assert store.resolve_finding(
        identity=_identity(),
        episode=finding["episode"],
        observation_revision=finding["observation_revision"],
        source_revision={"registry_revision": 1},
    )
    metadata = json.loads((store.runtime_root / "diagnostics" / "scheduler-v1" / "metadata.json").read_text())[
        "diagnostics"
    ]
    value = {
        "version": 1,
        "epoch": metadata["epoch"],
        "project_id": None,
        "captured_max_sequence": 99,
        "last_examined_sequence": 0,
    }
    cursor = (
        base64.urlsafe_b64encode(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode()
    )

    with pytest.raises(ValueError, match="captured sequence"):
        store.history_view(cursor=cursor)


def test_active_filters_totals_and_limit_are_exact_for_retained_records(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    _observe(store, 1, project_id="project-a")
    _observe(store, 2, project_id="project-b")

    page = store.active_view(project_id="project-a", reason="ready_index_unreadable", limit=1)

    assert page["coverage"] == "complete"
    assert page["total"] == 1
    assert len(page["items"]) == 1
    assert page["items"][0]["identity"]["project_id"] == "project-a"
    assert page["aggregates"]["reason"]["ready_index_unreadable"] == 1


def test_coalescing_finding_outside_summary_sample_does_not_inflate_project_count(
    tmp_path: Path,
) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    for index in range(1, 18):
        _observe(store, index, project_id="project-a")

    _observe(store, 17, project_id="project-a")

    assert store.active_view(project_id="project-a")["total"] == 17
    assert store.summary_view(project_id="project-a")["active_count"] == 17


@pytest.mark.parametrize("limit", [0, 33, True])
def test_query_limit_is_bounded(limit: object, tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))

    with pytest.raises((TypeError, ValueError), match="limit"):
        store.active_view(limit=limit)  # type: ignore[arg-type]


def test_encoded_history_page_stays_within_public_budget(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    for index in range(1, 33):
        finding = _observe(store, index, project_id="project-a")
        assert store.resolve_finding(
            identity=_identity(index, project_id="project-a"),
            episode=finding["episode"],
            observation_revision=finding["observation_revision"],
            source_revision={"registry_revision": index},
        )

    page = store.history_view(limit=32)

    assert len(json.dumps(page, ensure_ascii=False, sort_keys=True).encode("utf-8")) <= 128 * 1024


def test_cycle_publication_materializes_bounded_decision_and_finding(tmp_path: Path) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    finding_identity = _identity()
    store.publish_cycle(
        counters={"scheduler.cycles": 1},
        timings={"dispatch": 0.25},
        probes=(
            {
                "identity": {
                    "producer": "primary_probe",
                    "reason_code": "primary_demand",
                    "component": "scheduler",
                    "stage": "admission",
                    "check": "primary_demand",
                    "scope_type": "machine_lane",
                    "runtime_id": "a" * 64,
                    "resource_lane": "gpu",
                },
                "outcome": "unresolved",
                "coverage": "incomplete",
                "capacity_context": {"available": 2},
                "source_revision": {"registry_revision": 1},
                "findings": (
                    {
                        "identity": finding_identity,
                        "severity": "fault",
                        "source_revision": {"registry_revision": 1},
                        "details": {"exception_type": "OSError"},
                    },
                ),
            },
        ),
        working_set={"dormant_bindings": 2},
        observed_at="2026-09-28T00:00:00Z",
    )

    active = store.active_view()
    summary = store.summary_view()
    assert active["coverage"] == "incomplete"
    assert [item["identity"] for item in active["items"]] == [finding_identity]
    assert summary["counters"] == {"scheduler.cycles": 1}
    assert summary["timings"] == {"dispatch": 0.25}
    assert summary["working_set"] == {"dormant_bindings": 2}
    assert len(summary["decision_samples"]) == 1


def test_cycle_publication_keeps_selecting_after_a_failed_finding_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    first = {
        "identity": _identity(1, project_id="project-a"),
        "source_revision": {"registry_revision": 1},
    }
    second = {
        "identity": _identity(2, project_id="project-b"),
        "source_revision": {"registry_revision": 2},
    }
    probes = (
        {"coverage": "complete", "findings": (first,)},
        {"coverage": "complete", "findings": (second,)},
    )
    original_observe = store._observe_finding_locked
    calls = 0

    def reject_first_finding(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise diagnostic_module._StoreCapacityError("publication_operation_limit")
        return original_observe(**kwargs)

    monkeypatch.setattr(store, "_observe_finding_locked", reject_first_finding)

    store.publish_cycle({}, {}, probes, {}, observed_at="2026-09-28T00:00:00Z")

    active = store.active_view()
    summary = store.summary_view()
    assert [item["identity"]["project_id"] for item in active["items"]] == ["project-b"]
    assert summary["coverage"] == "incomplete"
    assert summary["reason"] == "publication_budget_exhausted"
    metadata = json.loads((store.runtime_root / "diagnostics" / "scheduler-v1" / "metadata.json").read_text())[
        "diagnostics"
    ]
    assert metadata["cycle_findings_written"] == 1


def test_cycle_publication_marks_malformed_finding_incomplete_without_dropping_later_valid_input(
    tmp_path: Path,
) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    probe = {
        "coverage": "complete",
        "findings": (
            {"identity": _identity(), "source_revision": None},
            {"identity": _identity(2), "source_revision": {"registry_revision": 2}},
        ),
    }

    store.publish_cycle({}, {}, (probe,), {}, observed_at="2026-09-28T00:00:00Z")

    active = store.active_view()
    assert active["coverage"] == "incomplete"
    assert active["reason"] == "identity_incomplete"
    assert [item["identity"] for item in active["items"]] == [_identity(2)]


def test_cycle_publication_checks_cap_before_ignoring_a_trailing_non_mapping_finding(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    findings = tuple(
        {"identity": _identity(index), "source_revision": {"registry_revision": index}} for index in range(1, 7)
    ) + ("ignored",)

    store.publish_cycle(
        {},
        {},
        ({"coverage": "complete", "findings": findings},),
        {},
        observed_at="2026-09-28T00:00:00Z",
    )

    summary = store.summary_view()
    assert summary["coverage"] == "incomplete"
    assert summary["reason"] == "publication_budget_exhausted"


def test_cycle_publication_stops_overflowed_probe_and_counts_later_persisted_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    retained = _observe(store)
    monkeypatch.setattr(diagnostic_module, "MAX_ACTIVE_RECORDS", 1)
    overflowing_probe = {
        "coverage": "complete",
        "findings": (
            {"identity": _identity(2), "source_revision": {"registry_revision": 2}},
            {"identity": _identity(3), "source_revision": {"registry_revision": 3}},
        ),
    }
    retained_probe = {
        "coverage": "complete",
        "findings": ({"identity": _identity(), "source_revision": {"registry_revision": 4}},),
    }

    store.publish_cycle(
        {},
        {},
        (overflowing_probe, retained_probe),
        {},
        observed_at="2026-09-28T00:00:00Z",
    )

    metadata = json.loads((store.runtime_root / "diagnostics" / "scheduler-v1" / "metadata.json").read_text())[
        "diagnostics"
    ]
    assert metadata["omitted_observations"] == 1
    assert metadata["cycle_findings_written"] == 1
    active = store.active_view()
    assert active["total"] == 1
    assert active["items"][0]["observation_revision"] == retained["observation_revision"] + 1


def test_decision_retention_evicts_at_sixty_five_identities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Retention is a record-count contract, not a host filesystem speed test.
    # Deadline enforcement has a separate deterministic boundary test below.
    monkeypatch.setattr(diagnostic_module, "time", SimpleNamespace(monotonic_ns=lambda: 0))
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    for index in range(65):
        identity = _identity(index)
        identity["reason_code"] = f"decision_{index}"
        store.record_decision(
            identity=identity,
            outcome="no_primary_demand",
            source_revision={"registry_revision": index},
            observed_at="2026-09-28T00:00:00Z",
        )

    decision_paths = list((runtime.root / "diagnostics" / "scheduler-v1" / "decisions").glob("*.json"))
    assert len(decision_paths) == 64
    assert store.summary_view()["status"] == "available"


def test_publication_deadline_rejects_late_work_without_spending_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    now_ns = [100]
    monkeypatch.setattr(diagnostic_module, "time", SimpleNamespace(monotonic_ns=lambda: now_ns[0]))
    budget = diagnostic_module._WriteBudget(0)
    now_ns[0] += diagnostic_module.MAX_PUBLICATION_ADMISSION_NS
    budget.admit(7)
    assert (budget.operations, budget.encoded_bytes) == (1, 7)
    now_ns[0] += 1
    with pytest.raises(OSError, match="publication_admission_deadline"):
        budget.admit(11)
    assert (budget.operations, budget.encoded_bytes) == (1, 7)
    budget.admit(11, final=True)
    assert (budget.operations, budget.encoded_bytes) == (2, 18)
    with pytest.raises(OSError, match="publication_byte_limit"):
        budget.admit(diagnostic_module.MAX_PUBLICATION_BYTES, final=True)
    assert (budget.operations, budget.encoded_bytes) == (2, 18)
    budget.operations = diagnostic_module.MAX_PUBLICATION_OPERATIONS
    with pytest.raises(OSError, match="publication_operation_limit"):
        budget.admit(0, final=True)


def test_pending_resolution_forces_incomplete_views_until_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    finding = _observe(store)
    original_write = store._write

    def crash_after_history_commit(path, value, budget, *, final=False):
        original_write(path, value, budget, final=final)
        if path == runtime.paths["scheduler_diagnostics_history_index"]:
            raise RuntimeError("simulated process crash")

    monkeypatch.setattr(store, "_write", crash_after_history_commit)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        store.resolve_finding(
            identity=_identity(),
            episode=finding["episode"],
            observation_revision=finding["observation_revision"],
            source_revision={"registry_revision": 1},
        )

    reopened = SchedulerDiagnosticStore(runtime)
    active = reopened.active_view()
    history = reopened.history_view()
    summary = reopened.summary_view()
    assert active["coverage"] == history["coverage"] == summary["coverage"] == "incomplete"
    assert active["reason"] == history["reason"] == summary["reason"] == "recovery_pending"
    assert len(active["items"]) == len(history["items"]) == 1

    assert reopened.recover() == {"recovered": 1, "discarded": 0}
    assert reopened.active_view()["items"] == []
    assert reopened.summary_view()["active_count"] == 0


def test_recovery_keeps_pending_until_all_orphan_segments_are_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    finding = _observe(store)
    original_write = store._write

    def crash_after_history_commit(path, value, budget, *, final=False):
        original_write(path, value, budget, final=final)
        if path == runtime.paths["scheduler_diagnostics_history_index"]:
            raise RuntimeError("simulated process crash")

    monkeypatch.setattr(store, "_write", crash_after_history_commit)
    with pytest.raises(RuntimeError, match="simulated process crash"):
        store.resolve_finding(
            identity=_identity(),
            episode=finding["episode"],
            observation_revision=finding["observation_revision"],
            source_revision={"registry_revision": 1},
        )

    segments = runtime.paths["scheduler_diagnostics_history_segments"]
    for sequence in range(100, 105):
        (segments / f"{sequence}-{sequence}.json").write_text("{}", encoding="utf-8")

    reopened = SchedulerDiagnosticStore(runtime)
    assert reopened.recover() == {"recovered": 1, "discarded": 4}
    assert len(list(runtime.paths["scheduler_diagnostics_pending"].glob("*.json"))) == 1
    assert reopened.recover() == {"recovered": 0, "discarded": 1}
    assert list(runtime.paths["scheduler_diagnostics_pending"].glob("*.json")) == []
    assert [path.name for path in segments.glob("*.json")] == ["1-1.json"]


@pytest.mark.parametrize("missing", ["metadata.json", "history/index.json"])
def test_orphan_summary_is_unavailable_when_required_store_record_is_missing(tmp_path: Path, missing: str) -> None:
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    store = SchedulerDiagnosticStore(runtime)
    _observe(store)
    (runtime.root / "diagnostics" / "scheduler-v1" / missing).unlink()

    summary = store.summary_view()

    assert summary["status"] == "unavailable"
    assert summary["coverage"] == "unknown"
    assert summary["reason"] == "store_corrupt"


def test_complete_cycle_resolves_only_the_same_binding_and_lane(tmp_path: Path) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    finding = _observe(store)
    complete_probe = {
        "identity": {
            "producer": "primary_probe",
            "reason_code": "primary_demand",
            "component": "scheduler",
            "stage": "admission",
            "check": "primary_demand",
            "scope_type": "machine_lane",
            "runtime_id": "a" * 64,
            "resource_lane": "gpu",
        },
        "outcome": "no_primary_demand",
        "coverage": "complete",
        "source_revision": {"registry_revision": 2},
        "covered_bindings": [
            {
                "project_id": "project-a",
                "registration_generation": "registration-1",
            }
        ],
        "findings": [],
    }

    assert store.reconcile_cycle((complete_probe,), resolved_at="2026-09-28T00:02:00Z")
    assert store.active_view()["items"] == []
    history = store.history_view()
    assert history["items"][0]["episode"] == finding["episode"]


def test_reconcile_cycle_rejects_candidate_updated_after_bounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SchedulerDiagnosticStore(MachineRuntime(tmp_path / "machine-runtime"))
    first = _observe(store)
    complete_probe = {
        "identity": {
            "producer": "primary_probe",
            "reason_code": "primary_demand",
            "component": "scheduler",
            "stage": "admission",
            "check": "primary_demand",
            "scope_type": "machine_lane",
            "runtime_id": "a" * 64,
            "resource_lane": "gpu",
        },
        "outcome": "no_primary_demand",
        "coverage": "complete",
        "source_revision": {"registry_revision": 2},
        "covered_bindings": [
            {
                "project_id": "project-a",
                "registration_generation": "registration-1",
            }
        ],
        "findings": [],
    }
    original_resolve = store.resolve_finding

    def update_then_resolve(**kwargs):
        newer = _observe(store)
        assert newer["episode"] == first["episode"]
        assert newer["observation_revision"] == first["observation_revision"] + 1
        return original_resolve(**kwargs)

    monkeypatch.setattr(store, "resolve_finding", update_then_resolve)

    assert not store.reconcile_cycle((complete_probe,), resolved_at="2026-09-28T00:02:00Z")
    active = store.active_view()
    assert active["total"] == 1
    assert active["items"][0]["observation_revision"] == first["observation_revision"] + 1

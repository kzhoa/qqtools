from __future__ import annotations

from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.task import submit as submit_task
from qqtools.plugins.qexp.layout import read_schema_version
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.runtime.locks import exclusive
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.upgrade import (
    DeterministicUpgradeError,
    MigrationPlugin,
    MigrationRegistry,
    MigrationSpec,
    PhaseResult,
    UpgradeCoordinator,
)
from qqtools.plugins.qexp.runtime.upgrade.machine import (
    MachineUpgradeBudget,
    advance_registered_upgrades,
    discover_registered_upgrades,
    inspect_registered_upgrades,
)


class _OnlinePlugin(MigrationPlugin):
    spec = MigrationSpec(
        name="test-online",
        source_protocol="schema-6",
        target_protocol="schema-6-online",
        compatible_readers=("schema-6", "schema-6-online"),
        compatible_writers=("schema-6", "schema-6-online"),
        phases=("expansion", "backfill", "audit", "activation"),
        work_budget=2,
    )

    def is_applicable(self, cfg) -> bool:
        del cfg
        return True

    def terminal_invariant_holds(self, cfg) -> bool:
        del cfg
        return True

    def backfill(self, context) -> PhaseResult:
        if context.journal["upgrade"]["migrations"][self.spec.name].get("cursor") is None:
            return PhaseResult("progressed", work_items=2, cursor="task-2")
        return PhaseResult("complete", work_items=1, cursor="task-2")

    def plan_repair(self, context, target: str):
        return {
            "target": target,
            "observed_revisions": {"projection": 1},
            "evidence": ["authoritative-test-record"],
            "intended_changes": ["rebuild projection"],
            "snapshot_ready": True,
        }

    def apply_repair(self, context, plan):
        return {"applied": True, "target": plan["target"]}

    def validate_repair(self, context, plan):
        return PhaseResult("complete", detail={"validated": plan["target"]})


def _config(tmp_path: Path):
    return init_shared_root(
        tmp_path / "project" / ".qexp",
        "gpu-1",
        runtime_root=tmp_path / "project-runtime",
    )


def test_online_migration_is_durable_and_phase_bounded(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    registry = MigrationRegistry([_OnlinePlugin()])
    coordinator = UpgradeCoordinator(cfg, registry=registry, holder_id="machine-a")

    assert coordinator.status()["state"] == "idle"
    assert coordinator.discover()["state"] == "runnable"
    assert coordinator.advance()["phase"] == "backfill"
    assert coordinator.advance()["phase"] == "backfill"
    assert coordinator.advance()["phase"] == "audit"
    assert coordinator.advance()["phase"] == "activation"
    assert coordinator.advance()["state"] == "completed"
    assert coordinator.status()["pending"] is False


def test_pause_repair_and_resume_are_explicit_and_restart_safe(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    registry = MigrationRegistry([_OnlinePlugin()])
    coordinator = UpgradeCoordinator(cfg, registry=registry, holder_id="operator")
    coordinator.discover()

    assert coordinator.request_pause("inspect evidence")["state"] == "paused"
    repair = coordinator.inspect_repair("test-online")
    assert repair["state"] == "planned"
    assert coordinator.apply_repair(repair["repair_id"])["state"] == "applied"
    assert coordinator.validate_repair(repair["repair_id"])["state"] == "validated"
    assert coordinator.resume()["state"] == "runnable"
    assert UpgradeCoordinator(cfg, registry=registry).status()["state"] == "runnable"


def test_default_production_migration_activates_additive_capability(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"].pop("protocol", None)
    atomic_replace(schema_path, schema)
    coordinator = UpgradeCoordinator(cfg)

    assert coordinator.discover()["state"] == "runnable"
    while coordinator.status()["pending"]:
        coordinator.advance()

    assert read_schema_version(cfg) == 6
    schema = read_json(schema_path)
    manifest = read_json(manifest_path)["upgrade_protocol_manifest"]
    assert "protocol" not in schema["schema"]
    assert manifest["protocol"] == "metadata:upgrade-journal-v1"
    assert manifest["schema_digest"]
    restarted = UpgradeCoordinator(cfg)
    assert restarted.status()["pending"] is False
    submitted = submit_task(cfg, ["echo", "upgrade-reload"])
    assert submitted.task_id


def test_machine_upgrade_discovery_is_cached_until_registry_changes(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    (cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json").unlink()
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)

    first = discover_registered_upgrades(runtime)
    assert first["pending_project_ids"]

    def should_not_probe(*args, **kwargs):
        raise AssertionError("waiting upgrade was probed from the agent loop")

    monkeypatch.setattr(UpgradeCoordinator, "status", should_not_probe)
    cached = discover_registered_upgrades(runtime)
    assert cached["pending_project_ids"] == first["pending_project_ids"]


def test_machine_budget_rotates_projects_without_losing_queued_work(tmp_path: Path) -> None:
    first_cfg = _config(tmp_path / "first")
    second_cfg = _config(tmp_path / "second")
    (first_cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json").unlink()
    (second_cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json").unlink()
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(first_cfg.shared_root, first_cfg.machine_name)
    runtime.add_binding(second_cfg.shared_root, second_cfg.machine_name)
    budget = MachineUpgradeBudget(
        max_projects=1,
        max_slices=1,
        max_records=1,
        max_metadata_ops=8,
        max_io_bytes=64 * 1024,
    )

    first = advance_registered_upgrades(runtime, budget=budget)
    second = advance_registered_upgrades(runtime, budget=budget)

    assert first["projects"][0]["project_id"] != second["projects"][0]["project_id"]
    assert first["budget_used"] == {
        "records": 1,
        "metadata_ops": 8,
        "io_bytes": 64 * 1024,
        "workers": 1,
        "queued_work": 1,
    }
    _revision, bindings = runtime.load_registry()
    assert set(first["pending_project_ids"]) == {binding.project_id for binding in bindings}


def test_inaccessible_discovery_is_retained_and_reprobed(tmp_path: Path, monkeypatch) -> None:
    cfg = _config(tmp_path)
    (cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json").unlink()
    runtime = MachineRuntime(tmp_path / "machine-runtime")
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    original_discover = UpgradeCoordinator.discover

    def inaccessible(*args, **kwargs):
        raise OSError("temporary shared storage outage")

    monkeypatch.setattr(UpgradeCoordinator, "discover", inaccessible)
    first = discover_registered_upgrades(runtime)
    assert first["pending_project_ids"]
    monkeypatch.setattr(UpgradeCoordinator, "discover", original_discover)
    runtime.upgrade_probe_deadlines[next(iter(runtime.upgrade_pending_projects))] = 0.0
    recovered = discover_registered_upgrades(runtime)
    assert recovered["pending_project_ids"]
    assert runtime.upgrade_runnable_projects


def test_pause_intent_survives_a_busy_coordinator(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    registry = MigrationRegistry([_OnlinePlugin()])
    coordinator = UpgradeCoordinator(cfg, registry=registry)
    coordinator.discover()
    lock_path = shared_paths(cfg.shared_root)["locks"] / "upgrade.lock"
    with exclusive(lock_path):
        paused = coordinator.request_pause("stop after this slice")
        assert paused["state"] == "pause_pending"
    restarted = UpgradeCoordinator(cfg, registry=registry)
    assert restarted.advance()["state"] == "paused"


def test_registry_sorts_dependencies_and_rejects_cycles() -> None:
    class Z(MigrationPlugin):
        spec = MigrationSpec("z", "schema-6", "z", ("schema-6",), ("schema-6",))

    class A(MigrationPlugin):
        spec = MigrationSpec("a", "schema-6", "a", ("schema-6",), ("schema-6",), dependencies=("z",))

    assert [plugin.spec.name for plugin in MigrationRegistry([A(), Z()]).values()] == ["z", "a"]
    class Loop(MigrationPlugin):
        spec = MigrationSpec("loop", "schema-6", "loop", ("schema-6",), ("schema-6",), dependencies=("loop",))

    with pytest.raises(ValueError, match="itself"):
        MigrationRegistry([Loop()])


def test_production_repair_plans_missing_manifest_and_rejects_schema_staleness(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest.unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    assert coordinator.request_pause("inspect absent manifest")["state"] == "paused"
    repair = coordinator.inspect_repair("upgrade-journal-v1")
    assert repair["plan"]["manifest_missing"] is True
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"]["created_at"] = "changed-after-repair-plan"
    atomic_replace(schema_path, schema)
    with pytest.raises(Exception, match="schema version record changed"):
        coordinator.apply_repair(repair["repair_id"])


def test_production_audit_rejects_manifest_when_schema_changes_after_expansion(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"].pop("protocol", None)
    atomic_replace(schema_path, schema)
    coordinator = UpgradeCoordinator(cfg)

    coordinator.discover()
    assert coordinator.advance()["phase"] == "audit"
    schema = read_json(schema_path)
    schema["schema"]["created_at"] = "changed-after-expansion"
    atomic_replace(schema_path, schema)

    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "schema digest is stale" in status["blockers"][0]


def test_production_activation_rejects_schema_changed_after_audit(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    schema_path = cfg.shared_root / "schema" / "version.json"
    schema = read_json(schema_path)
    schema["schema"].pop("protocol", None)
    atomic_replace(schema_path, schema)
    coordinator = UpgradeCoordinator(cfg)

    coordinator.discover()
    assert coordinator.advance()["phase"] == "audit"
    assert coordinator.advance()["phase"] == "activation"
    schema = read_json(schema_path)
    schema["schema"]["created_at"] = "changed-after-audit"
    atomic_replace(schema_path, schema)

    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "schema digest is stale" in status["blockers"][0]
    assert read_json(schema_path)["schema"].get("protocol") is None


def test_production_activation_keeps_schema_readable_by_strict_legacy_reader(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    coordinator = UpgradeCoordinator(cfg)

    coordinator.discover()
    while coordinator.status()["pending"]:
        coordinator.advance()

    schema = read_json(cfg.shared_root / "schema" / "version.json")["schema"]
    assert set(schema) == {
        "name",
        "version",
        "minimum_reader_version",
        "created_at",
        "required_capabilities",
    }
    assert read_schema_version(cfg) == 6


def test_repair_validation_rejects_manifest_corrupted_after_apply(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    assert coordinator.request_pause("repair manifest")["state"] == "paused"
    repair = coordinator.inspect_repair("upgrade-journal-v1")
    coordinator.apply_repair(repair["repair_id"])
    atomic_replace(manifest_path, {"upgrade_protocol_manifest": {"capability": "corrupt"}})

    validated = coordinator.validate_repair(repair["repair_id"])
    assert validated["state"] == "validation_failed"
    journal = read_json(shared_paths(cfg.shared_root)["upgrade_journal"])
    assert "repair_terminal_invariant" in (
        journal["upgrade"]["migrations"]["upgrade-journal-v1"]["error"]
    )


def test_repair_resume_rechecks_manifest_after_successful_validation(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    assert coordinator.request_pause("repair manifest")["state"] == "paused"
    repair = coordinator.inspect_repair("upgrade-journal-v1")
    coordinator.apply_repair(repair["repair_id"])
    assert coordinator.validate_repair(repair["repair_id"])["state"] == "validated"
    atomic_replace(manifest_path, {"upgrade_protocol_manifest": {"capability": "corrupt"}})

    with pytest.raises(DeterministicUpgradeError, match="repair_terminal_invariant"):
        coordinator.resume()
    assert coordinator.status()["state"] == "paused"
    assert coordinator.status()["pending"] is True


def test_repair_snapshot_write_cannot_exceed_migration_io_budget(tmp_path: Path) -> None:
    apply_calls: list[str] = []

    class OversizedRepairSnapshot(MigrationPlugin):
        spec = MigrationSpec(
            "oversized-repair-snapshot",
            "schema-6",
            "oversized-repair-snapshot",
            ("schema-6",),
            ("schema-6",),
            max_io_bytes_per_slice=32,
        )

        def is_applicable(self, cfg) -> bool:
            del cfg
            return True

        def plan_repair(self, context, target: str):
            del context
            return {
                "target": target,
                "evidence": ["test-record"],
                "intended_changes": ["test-repair"],
                "snapshot_ready": True,
            }

        def repair_snapshot(self, context):
            del context
            return {"payload": "x" * 4096}

        def apply_repair(self, context, plan):
            del context, plan
            apply_calls.append("called")
            return {"applied": True}

    cfg = _config(tmp_path)
    coordinator = UpgradeCoordinator(
        cfg, registry=MigrationRegistry([OversizedRepairSnapshot()])
    )
    coordinator.discover()
    coordinator.request_pause("test oversized repair snapshot")
    repair = coordinator.inspect_repair("oversized-repair-snapshot")

    with pytest.raises(DeterministicUpgradeError, match="I/O bytes budget"):
        coordinator.apply_repair(repair["repair_id"])

    snapshot_path = (
        shared_paths(cfg.shared_root)["upgrade_repairs"] / f"{repair['repair_id']}.snapshot.json"
    )
    assert not snapshot_path.exists()
    assert apply_calls == []
    assert coordinator.status()["repair"]["state"] == "planned"


def test_completed_production_migration_detects_missing_terminal_manifest(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    manifest_path = cfg.shared_root / "operations" / "upgrades" / "protocol-manifest.json"
    manifest_path.unlink()
    coordinator = UpgradeCoordinator(cfg)
    coordinator.discover()
    while coordinator.status()["pending"]:
        coordinator.advance()

    manifest_path.unlink()
    restarted = UpgradeCoordinator(cfg)
    assert restarted.status()["state"] == "repair_required"
    discovered = restarted.discover()
    assert discovered["state"] == "repair_required"
    assert discovered["pending"] is True
    assert discovered["migrations"][0]["state"] == "repair_required"


def test_upgrade_storage_rejects_actual_json_write_over_io_budget(tmp_path: Path) -> None:
    class OverIoBudget(MigrationPlugin):
        spec = MigrationSpec(
            "over-io-budget",
            "schema-6",
            "over-io-budget",
            ("schema-6",),
            ("schema-6",),
            max_io_bytes_per_slice=32,
        )

        def is_applicable(self, cfg) -> bool:
            del cfg
            return True

        def expansion(self, context) -> PhaseResult:
            context.storage.atomic_replace(context.cfg.shared_root / "operations" / "too-large.json", {"value": "x" * 64})
            return PhaseResult("complete")

    coordinator = UpgradeCoordinator(_config(tmp_path), registry=MigrationRegistry([OverIoBudget()]))
    coordinator.discover()
    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "I/O bytes budget" in status["blockers"][0]


def test_phase_cannot_bypass_storage_budget_with_global_json_store(tmp_path: Path) -> None:
    class BypassStorage(MigrationPlugin):
        spec = MigrationSpec(
            "bypass-storage",
            "schema-6",
            "bypass-storage",
            ("schema-6",),
            ("schema-6",),
            max_io_bytes_per_slice=32,
        )

        def is_applicable(self, cfg) -> bool:
            del cfg
            return True

        def expansion(self, context) -> PhaseResult:
            atomic_replace(context.cfg.shared_root / "operations" / "bypass.json", {"value": "x" * 4096})
            return PhaseResult("complete")

    coordinator = UpgradeCoordinator(_config(tmp_path), registry=MigrationRegistry([BypassStorage()]))
    coordinator.discover()
    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "must use UpgradeContext.storage" in status["blockers"][0]


def test_activation_predicate_cannot_bypass_storage_budget(tmp_path: Path) -> None:
    class PredicateBypass(MigrationPlugin):
        spec = MigrationSpec(
            "predicate-bypass",
            "schema-6",
            "predicate-bypass",
            ("schema-6",),
            ("schema-6",),
            phases=("audit", "activation"),
        )

        def is_applicable(self, cfg) -> bool:
            del cfg
            return True

        def activation_ready(self, context) -> bool:
            atomic_replace(
                context.cfg.shared_root / "operations" / "predicate-bypass.json",
                {"value": "unmetered"},
            )
            return True

    coordinator = UpgradeCoordinator(
        _config(tmp_path), registry=MigrationRegistry([PredicateBypass()])
    )
    coordinator.discover()
    assert coordinator.advance()["phase"] == "activation"

    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "migration callbacks must use UpgradeContext.storage" in status["blockers"][0]


def test_phase_result_cannot_exceed_the_declared_record_budget(tmp_path: Path) -> None:
    class OverBudget(MigrationPlugin):
        spec = MigrationSpec(
            "over-budget",
            "schema-6",
            "over-budget",
            ("schema-6",),
            ("schema-6",),
            work_budget=1,
            max_records_per_slice=1,
        )

        def is_applicable(self, cfg) -> bool:
            del cfg
            return True

        def expansion(self, context) -> PhaseResult:
            del context
            return PhaseResult("complete", work_items=2)

    coordinator = UpgradeCoordinator(_config(tmp_path), registry=MigrationRegistry([OverBudget()]))
    coordinator.discover()
    status = coordinator.advance()
    assert status["state"] == "repair_required"
    assert "record budget" in status["blockers"][0]

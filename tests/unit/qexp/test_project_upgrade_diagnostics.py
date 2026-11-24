"""Bounded selected-Project repair evidence and actionable status output."""

from __future__ import annotations

import json
import shlex
from copy import deepcopy

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.bindings import ProjectBinding
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.setup import initialize_machine
from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.commands import status as module
from qqtools.plugins.qexp.layout import project_id
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.store import atomic_replace


def _project(tmp_path, *, registrations=1):
    cfg = init_shared_root(tmp_path / "selected project" / ".qexp", "gpu-1", runtime_root=tmp_path / "partition")
    runtime = MachineRuntime(tmp_path / "machine runtime's root")
    initialize_machine(runtime, "gpu-1", agent_mode="daemon")
    selected = project_id(cfg.shared_root)
    bindings = [ProjectBinding(selected, cfg.shared_root, "gpu-1").to_dict()] if registrations else []
    bindings += [
        ProjectBinding(f"sibling-{i}", tmp_path / f"sibling-{i}", "gpu-1").to_dict()
        for i in range(max(0, registrations - 1))
    ]
    atomic_replace(runtime.paths["registry"], {"registry": {"version": 2, "revision": 0, "bindings": bindings}})
    return cfg, runtime, selected


def _journal(cfg, selected, *entries):
    value = {
        "upgrade": {
            "version": 1,
            "project_id": selected,
            "shared_root": str(cfg.shared_root),
            "pause": {},
            "updated_at": "2000-01-01T00:00:00Z",
            "migrations": {item["name"]: item for item in entries},
        }
    }
    path = shared_paths(cfg.shared_root)["upgrade_journal"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _entry(name="task-dependencies-v2", state="repair_required", **changes):
    return {"name": name, "state": state, "phase": "backfill", **changes}


def _status(cfg, runtime):
    return module.project_status(cfg, selection_source="explicit", machine_runtime=runtime)


def _human(result):
    return render(CliOutput(OutputKind.STATUS, result), "human")


@pytest.mark.parametrize(
    ("flag", "sentence"),
    [
        (True, "This migration blocks new training."),
        (False, "This migration does not block new training."),
        (None, "Effect on new training is unknown."),
        ("false", "Effect on new training is unknown."),
    ],
)
def test_repair_fields_warning_and_scoped_action_agree(tmp_path, flag, sentence):
    cfg, runtime, selected = _project(tmp_path)
    changes = {} if flag is None else {"admission_blocked": flag}
    _journal(cfg, selected, _entry(**changes))
    result = _status(cfg, runtime)
    assert result["upgrade"] == {
        "state": "repair_required",
        "migration": "task-dependencies-v2",
        "phase": "backfill",
        "admission_blocked": flag if type(flag) is bool else None,
        "reason": None,
    }
    repair_warnings = [w for w in result["warnings"] if "needs upgrade repair" in w]
    assert len(repair_warnings) == 1
    assert sentence in repair_warnings[0]
    human = _human(result)
    assert sentence in human
    command = shlex.join(
        ["qexp", "--machine-runtime-root", str(runtime.root), "admin", "upgrade", "status", "--project", selected]
    )
    assert result["next_actions"][0] == command
    assert result["next_actions"].count(command) == 1
    assert human.count(command) == 1


def test_first_repair_in_journal_order_not_first_active_or_later_admission_flag(tmp_path):
    cfg, runtime, selected = _project(tmp_path)
    _journal(
        cfg,
        selected,
        _entry("first-wait", "waiting", admission_blocked=True),
        _entry("repair-z", admission_blocked=False),
        _entry("repair-a", admission_blocked=True, phase="audit"),
    )
    result = _status(cfg, runtime)
    assert result["upgrade"]["migration"] == "repair-z"
    assert result["upgrade"]["phase"] == "backfill"
    assert result["upgrade"]["admission_blocked"] is False
    assert result["upgrade"]["additional_repairs"] == 1
    assert "Additional migrations need repair" in _human(result)
    assert len([w for w in result["warnings"] if "needs upgrade repair" in w]) == 1


def test_unregistered_repair_keeps_enrollment_guidance(tmp_path):
    cfg, runtime, selected = _project(tmp_path, registrations=0)
    _journal(cfg, selected, _entry())
    result = _status(cfg, runtime)
    assert result["upgrade"]["state"] == "repair_required"
    assert shlex.split(result["next_actions"][0])[3:5] == ["project", "register"]
    assert not any("admin upgrade status" in c for c in result["next_actions"])


@pytest.mark.parametrize(
    "observation",
    [
        "absent",
        "healthy",
        "wrong_id",
        "wrong_root",
        "version",
        "malformed",
        "oversized",
        "bad_entry",
        "bad_later_entry",
    ],
)
def test_optional_journal_failure_and_healthy_contract(tmp_path, observation):
    cfg, runtime, selected = _project(tmp_path)
    path = shared_paths(cfg.shared_root)["upgrade_journal"]
    value = _journal(cfg, selected, _entry(state="completed"))
    if observation == "absent":
        path.unlink()
    elif observation == "wrong_id":
        value["upgrade"]["project_id"] = "someone-else"
    elif observation == "wrong_root":
        value["upgrade"]["shared_root"] = str(tmp_path / "other")
    elif observation == "version":
        value["upgrade"]["version"] = 99
    elif observation == "bad_entry":
        value["upgrade"]["migrations"]["task-dependencies-v2"]["phase"] = "invalid"
    elif observation == "bad_later_entry":
        value["upgrade"]["migrations"]["task-dependencies-v2"]["state"] = "repair_required"
        value["upgrade"]["migrations"]["later"] = []
    if observation not in {"absent", "healthy"}:
        atomic_replace(path, value)
    if observation == "malformed":
        path.write_text("{")
    elif observation == "oversized":
        path.write_bytes(b" " * (module.MAX_RECORD_BYTES + 1))
    result = _status(cfg, runtime)
    upgrade = result["upgrade"]
    assert upgrade["state"] == ("no_repair_recorded" if observation == "healthy" else "unknown")
    assert all(upgrade[k] is None for k in ("migration", "phase", "admission_blocked"))
    if observation == "absent":
        assert upgrade["reason"] == "journal_missing"
    upgrade_warnings = [w for w in result["warnings"] if "upgrade" in w.lower()]
    assert bool(upgrade_warnings) == (observation not in {"absent", "healthy"})


@pytest.mark.parametrize("registrations", [1, 70])
def test_exactly_one_selected_journal_read_with_shared_budget(tmp_path, monkeypatch, registrations):
    cfg, runtime, selected = _project(tmp_path, registrations=registrations)
    _journal(cfg, selected, _entry())
    paths = []
    original = module._ReadBudget.read_json

    def read(reader, path):
        paths.append(path)
        return original(reader, path)

    monkeypatch.setattr(module._ReadBudget, "read_json", read)
    result = _status(cfg, runtime)
    assert paths.count(shared_paths(cfg.shared_root)["upgrade_journal"]) == 1
    assert not any("sibling-" in str(p) for p in paths)
    assert result["budget"]["read_attempts"] <= 32
    assert result["budget"]["bytes_read"] <= 2 * 1024 * 1024
    assert result["budget"]["bytes_read"] == sum(p.stat().st_size for p in paths if p.exists())


def test_exhausted_budget_cannot_read_upgrade_outside_reader(tmp_path, monkeypatch):
    cfg, runtime, selected = _project(tmp_path)
    _journal(cfg, selected, _entry())
    original = module._ReadBudget.read_json

    def read(reader, path):
        if path == shared_paths(cfg.shared_root)["upgrade_journal"]:
            reader.read_attempts = reader.max_read_attempts
        return original(reader, path)

    monkeypatch.setattr(module._ReadBudget, "read_json", read)
    result = _status(cfg, runtime)
    assert result["upgrade"]["state"] == "unknown"
    assert result["upgrade"]["reason"] == "read_budget_exhausted"
    assert result["budget"]["read_attempts"] == 32


def test_healthy_sibling_does_not_inherit_repair(tmp_path):
    cfg, runtime, selected = _project(tmp_path)
    _journal(cfg, selected, _entry())
    sibling = init_shared_root(tmp_path / "healthy" / ".qexp", "gpu-1", runtime_root=tmp_path / "sibling-runtime")
    _journal(sibling, project_id(sibling.shared_root), _entry(state="completed"))
    assert _status(cfg, runtime)["upgrade"]["state"] == "repair_required"
    result = _status(sibling, runtime)
    assert result["upgrade"]["state"] == "no_repair_recorded"
    assert not any("upgrade" in w.lower() for w in result["warnings"])


def test_shared_read_budget_caps_io_when_file_grows_after_stat(tmp_path, monkeypatch):
    path = tmp_path / "growing.json"
    path.write_text('{"value":"' + "x" * 5000 + '"}')
    from pathlib import Path
    from types import SimpleNamespace

    original = Path.lstat

    def stale_stat(candidate):
        if candidate == path:
            return SimpleNamespace(st_mode=original(candidate).st_mode, st_size=10)
        return original(candidate)

    monkeypatch.setattr(Path, "lstat", stale_stat)
    budget = module._ReadBudget(max_record_bytes=8192, max_total_bytes=100)
    with pytest.raises(module._BoundedReadFailure):
        budget.read_json(path)
    assert budget.bytes_read == 100


def test_total_byte_exhaustion_returns_unknown_without_private_journal_read(tmp_path, monkeypatch):
    cfg, runtime, selected = _project(tmp_path)
    _journal(cfg, selected, _entry())
    original = module._ReadBudget.read_json

    def read(reader, path):
        if path == shared_paths(cfg.shared_root)["upgrade_journal"]:
            reader.bytes_read = reader.max_total_bytes
        return original(reader, path)

    monkeypatch.setattr(module._ReadBudget, "read_json", read)
    result = _status(cfg, runtime)
    assert result["upgrade"]["state"] == "unknown"
    assert result["upgrade"]["reason"] == "total_read_budget_exhausted"
    assert result["budget"]["bytes_read"] == 2 * 1024 * 1024


@pytest.mark.parametrize("invalid", ["pause_missing", "pause_list", "empty_name", "slash_name", "backslash_name"])
def test_coordinator_envelope_and_migration_name_invariants(tmp_path, invalid):
    cfg, runtime, selected = _project(tmp_path)
    value = _journal(cfg, selected, _entry())
    upgrade = value["upgrade"]
    if invalid == "pause_missing":
        del upgrade["pause"]
    elif invalid == "pause_list":
        upgrade["pause"] = []
        upgrade["migrations"] = {}
    else:
        name = {"empty_name": "", "slash_name": "invalid/name", "backslash_name": "invalid\\name"}[invalid]
        upgrade["migrations"] = {name: _entry(name)}
    atomic_replace(shared_paths(cfg.shared_root)["upgrade_journal"], value)
    result = _status(cfg, runtime)
    assert result["upgrade"]["state"] == "unknown"
    assert result["upgrade"]["reason"] == "journal_malformed"
    assert any("Upgrade journal unavailable" in w for w in result["warnings"])


def test_budget_boundary_cannot_accept_valid_prefix_of_a_larger_record(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    path = tmp_path / "record.json"
    path.write_text("{}x")
    original = Path.lstat
    monkeypatch.setattr(
        Path, "lstat", lambda candidate: SimpleNamespace(st_mode=original(candidate).st_mode, st_size=2)
    )
    reader = module._ReadBudget(max_total_bytes=2)
    with pytest.raises(module._BoundedReadFailure, match="total_read_budget_exhausted"):
        reader.read_json(path)
    assert reader.bytes_read == 2

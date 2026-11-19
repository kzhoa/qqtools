"""Typed, process-restart-safe Group-service transactions."""

from __future__ import annotations

from dataclasses import replace

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import activation, locator
from qqtools.plugins.qexp.runtime.group_discovery.advance import (
    advance_group_service,
    initial_control_continuation,
    initial_discovery_continuation,
)
from qqtools.plugins.qexp.runtime.group_discovery.coverage import GroupCoverage
from qqtools.plugins.qexp.runtime.group_discovery.probe import initial_group_service_probe_state
from qqtools.plugins.qexp.runtime.locks import group_writer_lock
from qqtools.plugins.qexp.runtime.paths import submission_path
from tests.helpers.qexp_discovery import isolated_group, source_file

pytestmark = pytest.mark.integration


def _activate(cfg):
    for _ in range(20_000):
        record = activation.advance_group_service_activation(cfg)
        if record["state"] == "active":
            return
    raise AssertionError("Group-service activation did not converge")


def _advance_until_terminal(cfg, candidate, continuation, *, limit=2_000):
    for _ in range(limit):
        result = advance_group_service(cfg, candidate, continuation, initial_group_service_probe_state())
        continuation = result["continuation"]
        if result["state"] in {"quiescent", "stale", "blocked"}:
            return result
    raise AssertionError(f"Group-service transaction did not converge: {result}")


@pytest.mark.parametrize("slice_steps", [2, 3])
def test_legacy_discovery_continuation_converges_across_fresh_owners(tmp_path, monkeypatch, slice_steps):
    cfg = isolated_group(tmp_path, tail=1)
    cfg = replace(cfg, runtime_root=tmp_path / "runtime")
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")

    candidate = {"group": "experiment", "lane": "legacy", "generation": None}
    with monkeypatch.context() as bounded:
        bounded.setattr(
            "qqtools.plugins.qexp.runtime.group_discovery.advance._DISCOVERY_ADVANCES_PER_SLICE", slice_steps
        )
        first = advance_group_service(
            cfg, candidate, initial_discovery_continuation(), initial_group_service_probe_state()
        )
    assert first["state"] == "progress"
    assert first["continuation"]["bootstrap_revision"] is not None
    result = _advance_until_terminal(cfg, candidate, first["continuation"])

    assert result["state"] == "quiescent"
    assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete


@pytest.mark.parametrize("lane", ["legacy", "membership"])
def test_resume_after_failed_active_source_write(tmp_path, monkeypatch, lane):
    from qqtools.plugins.qexp.runtime.group_discovery import discovery

    cfg = replace(isolated_group(tmp_path, tail=1), runtime_root=tmp_path / "runtime")
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")
    candidate = {"group": "experiment", "lane": "legacy", "generation": None}
    if lane == "membership":
        _activate(cfg)
        discovery.publish_submission_debt(cfg.shared_root, "experiment", "batch")
        with group_writer_lock(cfg, "experiment"):
            record = locator.publish_group_locator_locked(cfg, "experiment", lane, "submission_commit")
        candidate.update(lane=lane, generation=record["generation"])
    original = discovery.atomic_replace

    def fail_active(path, value):
        if path.name == "active-source.json":
            raise OSError("active source unavailable")
        return original(path, value)

    with monkeypatch.context() as fault:
        fault.setattr(discovery, "atomic_replace", fail_active)
        failed = advance_group_service(
            cfg, candidate, initial_discovery_continuation(), initial_group_service_probe_state()
        )
    assert failed["state"] == "blocked"
    assert failed["continuation"]["debt_offset"] == 0
    result = _advance_until_terminal(cfg, candidate, failed["continuation"])
    assert result["state"] == "quiescent"
    assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete


def test_pre_activation_candidate_is_stale_after_cutover(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    _activate(cfg)
    candidate = {"group": "experiment", "lane": "legacy", "generation": None}
    result = advance_group_service(
        cfg, candidate, initial_discovery_continuation(), initial_group_service_probe_state()
    )
    assert result == {"state": "stale", "continuation": None, "probe": None}


def test_membership_transaction_retires_exact_locator_generation(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    cfg = replace(cfg, runtime_root=tmp_path / "runtime")
    _activate(cfg)
    with group_writer_lock(cfg, "experiment"):
        record = locator.publish_group_locator_locked(cfg, "experiment", "membership", "submission_commit")

    result = _advance_until_terminal(
        cfg,
        {"group": "experiment", "lane": "membership", "generation": record["generation"]},
        initial_discovery_continuation(),
    )

    assert result["state"] == "quiescent"
    assert locator.read_group_locator(cfg.shared_root, "experiment", "membership") is None


def test_control_transaction_carries_cursor_until_exact_retirement(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    cfg = replace(cfg, runtime_root=tmp_path / "runtime")
    _activate(cfg)
    with group_writer_lock(cfg, "experiment"):
        record = locator.publish_group_locator_locked(cfg, "experiment", "control", "task_change")

    result = _advance_until_terminal(
        cfg,
        {"group": "experiment", "lane": "control", "generation": record["generation"]},
        initial_control_continuation(),
    )

    assert result["state"] == "quiescent"
    assert locator.read_group_locator(cfg.shared_root, "experiment", "control") is None

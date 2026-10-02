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


def test_legacy_discovery_continuation_converges_across_fresh_owners(tmp_path):
    cfg = isolated_group(tmp_path, tail=1)
    cfg = replace(cfg, runtime_root=tmp_path / "runtime")
    source_file(submission_path(cfg.shared_root, "batch"), operation="batch")

    result = _advance_until_terminal(
        cfg,
        {"group": "experiment", "lane": "legacy", "generation": None},
        initial_discovery_continuation(),
    )

    assert result["state"] == "quiescent"
    assert GroupCoverage(cfg.shared_root, "experiment").status().is_complete


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

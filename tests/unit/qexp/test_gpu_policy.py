from __future__ import annotations

import json

import pytest


def test_gpu_id_parser_is_strict_and_normalizes_order() -> None:
    from qqtools.plugins.qexp.gpu_policy import parse_gpu_id_list

    assert parse_gpu_id_list("12,3,45,6,78") == (3, 6, 12, 45, 78)
    for value in ("", "0,", ",0", "0,,1", "0,0", "-1", "+1", "1-2", " 1", "1 ", "１"):
        with pytest.raises(ValueError):
            parse_gpu_id_list(value)


def test_persisted_policy_revisions_and_compare_and_swap(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import reset_gpu_policy, set_gpu_policy
    from qqtools.plugins.qexp.runtime.store import read_json

    runtime = MachineRuntime(tmp_path / "machine")
    first = set_gpu_policy(runtime, (2, 0), expected_revision=0)
    assert first["previous_revision"] == 0
    assert first["current_revision"] == 1
    assert first["configured_gpu_ids"] == [0, 2]

    with pytest.raises(ValueError, match=r"current 1"):
        set_gpu_policy(runtime, (1,), expected_revision=0)

    reset = reset_gpu_policy(runtime, expected_revision=1)
    assert reset["previous_revision"] == 1
    assert reset["current_revision"] == 2
    policy = read_json(runtime.paths["gpu_policy"])["gpu_policy"]
    assert policy["schema_version"] == 1
    assert policy["revision"] == 2
    assert policy["mode"] == "auto"
    assert policy["configured_gpu_ids"] is None


def test_malformed_policy_is_not_overwritten_and_fails_closed(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, resolve_gpu_policy, set_gpu_policy
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime = MachineRuntime(tmp_path / "machine")
    runtime.ensure_layout()
    malformed = {
        "gpu_policy": {
            "schema_version": 1,
            "revision": 4,
            "mode": "explicit",
            "configured_gpu_ids": [0, 0],
            "updated_at": "2026-09-21T00:00:00Z",
        }
    }
    atomic_replace(runtime.paths["gpu_policy"], malformed)

    view = resolve_gpu_policy(
        runtime.root,
        discovery=GpuDiscovery((0, 1), "available", None),
        environment_gpu_ids=None,
        environment_status="absent",
    )
    assert view.visible_gpu_ids is None
    assert view.visible_status == "unavailable"
    assert view.source == "persisted"
    assert any(item["reason"] == "gpu_policy_invalid" for item in view.warnings)

    with pytest.raises(ValueError, match="GPU policy"):
        set_gpu_policy(runtime, (1,))
    assert json.loads(runtime.paths["gpu_policy"].read_text(encoding="utf-8")) == malformed


def test_policy_precedence_intersection_and_missing_ids(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, resolve_gpu_policy, set_gpu_policy

    runtime = MachineRuntime(tmp_path / "machine")
    discovery = GpuDiscovery((0, 1, 2, 3), "available", None)

    environment = resolve_gpu_policy(
        runtime.root,
        discovery=discovery,
        environment_gpu_ids=(1, 3),
        environment_status="valid",
    )
    assert environment.source == "environment"
    assert environment.visible_gpu_ids == (1, 3)

    set_gpu_policy(runtime, (12, 3, 45, 6, 78))
    persisted = resolve_gpu_policy(
        runtime.root,
        discovery=discovery,
        environment_gpu_ids=(1, 3),
        environment_status="valid",
    )
    assert persisted.source == "persisted"
    assert persisted.visible_gpu_ids == (3,)
    assert persisted.undiscovered_configured_gpu_ids == (6, 12, 45, 78)
    warning = next(item for item in persisted.warnings if item["reason"] == "configured_gpu_ids_not_discovered")
    assert warning["configured_gpu_ids"] == [3, 6, 12, 45, 78]
    assert warning["visible_gpu_ids"] == [3]
    assert "--visible 3" in "\n".join(warning["repair_commands"])


def test_discovery_failure_does_not_classify_configured_ids_as_missing(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, resolve_gpu_policy, set_gpu_policy

    runtime = MachineRuntime(tmp_path / "machine")
    set_gpu_policy(runtime, (4, 5))
    view = resolve_gpu_policy(
        runtime.root,
        discovery=GpuDiscovery(None, "unavailable", "inventory_unavailable"),
        environment_gpu_ids=None,
        environment_status="absent",
    )
    assert view.visible_gpu_ids is None
    assert view.undiscovered_configured_gpu_ids is None
    assert not any(item["reason"] == "configured_gpu_ids_not_discovered" for item in view.warnings)
    assert any(item["reason"] == "gpu_discovery_unavailable" for item in view.warnings)


def test_explicit_none_is_known_empty_when_discovery_is_unavailable(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, resolve_gpu_policy, set_gpu_policy

    runtime = MachineRuntime(tmp_path / "machine")
    set_gpu_policy(runtime, ())
    view = resolve_gpu_policy(
        runtime.root,
        discovery=GpuDiscovery(None, "unavailable", "inventory_unavailable"),
        environment_gpu_ids=None,
        environment_status="absent",
    )
    assert view.visible_gpu_ids == ()
    assert view.visible_status == "empty"
    assert view.undiscovered_configured_gpu_ids is None
    assert view.warnings == ()


def test_locked_reservation_rechecks_policy_after_candidate_selection(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy, set_gpu_policy
    from qqtools.plugins.qexp.runtime.resources.reservations import reserve
    from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

    runtime = MachineRuntime(tmp_path / "machine")
    set_gpu_policy(runtime, (0, 1))
    candidate_context = GpuReservationPolicy(
        discovery=GpuDiscovery((0, 1), "available", None),
        environment_gpu_ids=None,
        environment_status="absent",
    )

    # Candidate selection observed GPU 1, then the policy mutation linearized first.
    set_gpu_policy(runtime, (0,))
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics):
        with pytest.raises(ValueError, match="not visible"):
            reserve(runtime.root, "task-stale", [1], gpu_policy=candidate_context)
    assert diagnostics.counters["scheduler.gpu_policy_rejections.not_visible"] == 1

    accepted = reserve(runtime.root, "task-current", [0], gpu_policy=candidate_context)
    assert accepted["reservation"]["gpu_ids"] == [0]


def test_invalid_environment_policy_fails_closed_for_gpu_reservation(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import GpuDiscovery, GpuReservationPolicy
    from qqtools.plugins.qexp.runtime.resources.reservations import reserve

    runtime = MachineRuntime(tmp_path / "machine")
    context = GpuReservationPolicy(
        discovery=GpuDiscovery((0,), "available", None),
        environment_gpu_ids=None,
        environment_status="invalid",
    )
    with pytest.raises(ValueError, match="environment"):
        reserve(runtime.root, "task-invalid-environment", [0], gpu_policy=context)


def test_pending_fallback_does_not_invent_draining_state(tmp_path) -> None:
    from qqtools.plugins.qexp.agent.context import MachineRuntime
    from qqtools.plugins.qexp.gpu_policy import show_gpu_policy
    from qqtools.plugins.qexp.runtime.resources.reservations import reserve

    runtime = MachineRuntime(tmp_path / "machine")
    reserve(runtime.root, "task-running", [2])

    shown = show_gpu_policy(runtime)

    assert shown["source"] == "pending"
    assert shown["visible_gpu_ids"] is None
    assert shown["unreserved_gpu_ids"] is None
    assert shown["reserved_gpu_ids"] == [2]
    assert shown["draining_gpu_ids"] == []

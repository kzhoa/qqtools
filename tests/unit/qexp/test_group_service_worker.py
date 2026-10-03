from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent import group_service_worker
from qqtools.plugins.qexp.agent.group_service_transport import (
    group_service_advance_request,
    group_service_probe_request,
    initial_group_service_continuation,
)
from qqtools.plugins.qexp.agent.project_io_protocol import PROJECT_IO_PROTOCOL_VERSION, ProjectIORequest
from qqtools.plugins.qexp.config_types import RootConfig
from qqtools.plugins.qexp.runtime.group_discovery.probe import initial_group_service_probe_state
from qqtools.plugins.qexp.runtime.store import check_mutation_fence


def _request(operation_kind: str, parameters) -> ProjectIORequest:
    return ProjectIORequest(
        protocol_version=PROJECT_IO_PROTOCOL_VERSION,
        runtime_id="a" * 64,
        executor_epoch="b" * 32,
        request_id="c" * 32,
        operation_kind=operation_kind,
        project_id="project",
        canonical_shared_root="/project/.qexp",
        registration_generation="generation",
        registry_revision=1,
        source_revisions={},
        provisional_offer_id=None,
        prepared_at="2026-10-03T00:00:00Z",
        parameters=parameters,
    )


def _config() -> RootConfig:
    return RootConfig.from_canonical_paths(
        Path("/project/.qexp"),
        Path("/project"),
        "gpu-1",
        Path("/runtime/old"),
    )


def test_probe_handler_uses_the_closed_read_only_route(monkeypatch):
    probe_state = initial_group_service_probe_state()
    spec = group_service_probe_request("gpu-1", probe_state)
    request = _request(spec.operation_kind, spec.parameters)
    events = []
    monkeypatch.setattr(
        group_service_worker,
        "load_root_config",
        lambda root, machine: events.append(("load", root, machine)) or _config(),
    )
    monkeypatch.setattr(
        group_service_worker,
        "probe_group_service",
        lambda root, state: events.append(("probe", root, state)) or {"state": "pending"},
    )

    result = group_service_worker.handle_group_service_request(request, Path("/runtime"), None)

    assert result == {"state": "pending"}
    assert events == [
        ("load", Path("/project/.qexp"), "gpu-1"),
        ("probe", Path("/project/.qexp"), probe_state),
    ]


def test_advance_handler_applies_exact_config_fence_immediately_before_transaction(monkeypatch):
    probe_state = initial_group_service_probe_state()
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    continuation = initial_group_service_continuation(candidate)
    spec = group_service_advance_request("gpu-1", candidate, continuation, probe_state)
    request = _request(spec.operation_kind, spec.parameters)
    events = []
    monkeypatch.setattr(group_service_worker, "load_root_config", lambda _root, _machine: _config())

    @contextmanager
    def fenced(shared_root, before_shared_mutation):
        events.append(("fence", shared_root))
        before_shared_mutation()
        yield

    def before_shared_mutation(cfg):
        events.append(("before", cfg.shared_root, cfg.runtime_root))

    def advance(cfg, observed_candidate, observed_continuation, observed_probe_state):
        events.append(("advance", cfg.shared_root, cfg.runtime_root))
        assert observed_candidate == candidate
        assert observed_continuation == continuation
        assert observed_probe_state == probe_state
        return {"state": "progress"}

    monkeypatch.setattr(group_service_worker, "fenced_mutations", fenced)
    monkeypatch.setattr(group_service_worker, "advance_group_service", advance)

    result = group_service_worker.handle_group_service_request(
        request,
        Path("/runtime"),
        before_shared_mutation,
    )

    assert result == {"state": "progress"}
    assert events == [
        ("fence", Path("/project/.qexp")),
        ("before", Path("/project/.qexp"), Path("/runtime/projects/project")),
        ("advance", Path("/project/.qexp"), Path("/runtime/projects/project")),
    ]


def test_advance_handler_rechecks_the_exact_config_for_every_shared_mutation(monkeypatch):
    probe_state = initial_group_service_probe_state()
    candidate = {"group": "experiment", "lane": "membership", "generation": 3}
    spec = group_service_advance_request(
        "gpu-1",
        candidate,
        initial_group_service_continuation(candidate),
        probe_state,
    )
    request = _request(spec.operation_kind, spec.parameters)
    loaded = _config()
    observed_configs = []
    advance_configs = []
    monkeypatch.setattr(group_service_worker, "load_root_config", lambda _root, _machine: loaded)

    def advance(cfg, _candidate, _continuation, _probe_state):
        advance_configs.append(cfg)
        check_mutation_fence(cfg.shared_root / "first.json")
        check_mutation_fence(cfg.shared_root / "second.json")
        return {"state": "progress"}

    monkeypatch.setattr(group_service_worker, "advance_group_service", advance)

    result = group_service_worker.handle_group_service_request(
        request,
        Path("/runtime"),
        observed_configs.append,
    )

    assert result == {"state": "progress"}
    assert len(observed_configs) == 2
    assert observed_configs[0] is observed_configs[1]
    assert observed_configs[0] is advance_configs[0]
    assert observed_configs[0] == replace(loaded, runtime_root=Path("/runtime/projects/project"))


def test_group_worker_handler_has_no_worker_entrypoint_dependency():
    source = inspect.getsource(group_service_worker)
    assert "project_io_worker" not in source


def test_group_worker_handler_rejects_other_operations():
    probe = group_service_probe_request("gpu-1", initial_group_service_probe_state())
    request = _request(probe.operation_kind, probe.parameters)
    object.__setattr__(request, "operation_kind", "scheduler_claim")

    with pytest.raises(ValueError, match="unsupported Group-service operation"):
        group_service_worker.handle_group_service_request(request, Path("/runtime"), None)

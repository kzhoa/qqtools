"""Opt-in retained history fixtures for the real machine workload qualification.

Use with authority_measurement and --settled-history-count. Fixture construction
is outside latency measurement and is not a lifecycle or durable-write benchmark.
The measured workload retains its real runners, fsyncs, and 15-second deadlines.
"""

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.agent.recovery_capture import RecoveryCapture
from qqtools.plugins.qexp.runtime.paths import idempotency_path, submission_path, task_path
from qqtools.plugins.qexp.runtime.recovery_admission import fence_recovery_admission
from qqtools.plugins.qexp.runtime.store import read_json
from qqtools.plugins.qexp.runtime.submission import normalize_submission_request, semantic_digest
from qqtools.plugins.qexp.runtime.submission_control import publish_submission
from qqtools.plugins.qexp.scheduler import cancel_task
from qqtools.plugins.qexp.task_observation import build_task_observation
from tests.helpers.qexp import observation_scale
from tests.helpers.qexp.observation_scale import build_observation


def pytest_addoption(parser):
    group = parser.getgroup("settled history qualification")
    group.addoption("--settled-history-count", type=int, default=0)
    group.addoption("--settled-history-submissions", choices=("bulk", "single"), default="bulk")


@pytest.fixture(autouse=True)
def retained_workload_history(request, monkeypatch, tmp_path):
    count = request.config.getoption("--settled-history-count")
    assert count in {0, 1000, 10_000, 100_000}
    if request.node.name != "test_authority_workload":
        yield
        return
    fixture_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(observation_scale.__file__))
    }
    module = request.module
    original = module.init_shared_root
    mode = request.config.getoption("--settled-history-submissions")

    overrides = json.loads(request.config.getoption("--authority-workload-profile", default="{}"))
    binding_count = overrides.get("bindings", 4)
    assert type(binding_count) is int and 1 <= binding_count <= 16
    prepared = {}
    # Build every retained dataset before any measured runtime binding exists;
    # large later fixtures must not age the first binding's eligibility lease.
    for index in range(binding_count):
        root = tmp_path / f"project-{index}"
        cfg = original(root / ".qexp", "gpu-1", agent_mode="daemon", runtime_root=root / "legacy")
        if count:
            seed_settled_history(cfg, count, mode)
        prepared[cfg.shared_root] = cfg

    def initialize(root, *args, **kwargs):
        return prepared[Path(root).resolve()]

    original_binding = MachineRuntime.ensure_binding

    def qualified_binding(runtime, *args, **kwargs):
        result = original_binding(runtime, *args, **kwargs)
        binding = result[0]
        with runtime.scheduler_authority() as owns:
            assert owns
            assert runtime.prepare_recovery_registration(binding)
            assert fence_recovery_admission(runtime, binding).is_fenced
            capture = RecoveryCapture(runtime, binding)
            deadline = time.monotonic() + 30
            try:
                while not capture.advance():
                    assert time.monotonic() < deadline, "fixture recovery qualification did not converge"
                assert capture.activate_group_authority()
                assert capture.release_source()
            finally:
                capture.close()
        return result

    monkeypatch.setattr(MachineRuntime, "ensure_binding", qualified_binding)
    monkeypatch.setattr(module, "init_shared_root", initialize)
    output = request.config.getoption("--authority-workload-output", default=None)
    assert output, "history qualification requires a retained workload report"
    yield
    report = json.loads(Path(output).read_text())
    assert report["outcome"] == "passed"
    for counters in report["agent"]["operations"].values():
        assert counters.get("history.file_open", {}).get("calls", 0) == 0, counters
    projects = report["control_plane_snapshot"]["authority_control_plane"]["projects"]
    assert len(projects) == report["profile"]["bindings"]
    assert all(project.get("work", {}).get("discovery_mode") == "primary" for project in projects), projects
    report["settled_history"] = {
        "per_project": count,
        "submission_shape": mode,
        "agent_history_file_opens": 0,
        "fixture_sha256": fixture_hashes,
    }
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")


def seed_settled_history(cfg, count, mode):
    """Build quiescent Task, Submission ownership and observation truth together."""
    assert count > 0 and mode in {"bulk", "single"}
    task = submit(cfg, ["true"], task_id="settled-history-000000")
    cancel_task(cfg, task.task_id)
    template = read_json(task_path(cfg.shared_root, task.task_id))
    old_source = submission_path(cfg.shared_root, task.submission_operation_id)
    source = read_json(old_source)
    identifiers = [f"settled-history-{number:06d}" for number in range(count)]
    first_key = source["submission"]["idempotency_key"]
    for index, identifier in enumerate(identifiers):
        operation_id = "settled-history-bulk" if mode == "bulk" else identifier
        record = deepcopy(template)
        record["task"]["task_id"] = identifier
        record["task"]["submission_operation_id"] = operation_id
        task_path(cfg.shared_root, identifier).write_text(json.dumps(record), encoding="utf-8")
        if mode == "single":
            operation = deepcopy(source)
            key = first_key if index == 0 else f"settled-history-key-{index}"
            _write_operation(cfg, operation, operation_id, key, [identifier])
    if mode == "bulk":
        _write_operation(cfg, source, "settled-history-bulk", first_key, identifiers)
    old_source.unlink()
    build_observation(cfg, {identifier: ("cancelled", None) for identifier in identifiers})


def request_worker_set(context):
    """Reconstruct whether the persisted request declared a Worker Set."""
    additions = context["worker_set_additions"]
    is_declared = context.get("worker_set_declared")
    if is_declared is None:
        is_declared = bool(additions)
    return additions if is_declared else None


def _write_operation(cfg, operation, operation_id, key, identifiers):
    submission = operation["submission"]
    context = submission["resolved_context"]
    template = context["task_specs"][0]
    context["task_ids"] = identifiers
    context["task_specs"] = [dict(template, task_id=identifier) for identifier in identifiers]
    submission["operation_id"] = operation_id
    submission["idempotency_key"] = key
    submission["kind"] = "batch" if len(identifiers) > 1 else "single"
    request = normalize_submission_request(
        context["task_specs"],
        group_name=submission["target_group"],
        kind=submission["kind"],
        worker_set=request_worker_set(context),
    )
    submission["raw_request_digest"] = request.raw_request_digest
    submission["resolved_context_digest"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
    submission["staged_task_count"] = len(identifiers)
    operation["task_observation"] = build_task_observation(identifiers, [None] * len(identifiers))
    if len(identifiers) > 1:
        publish_submission(cfg, operation)
    else:
        submission_path(cfg.shared_root, operation_id).write_text(json.dumps(operation), encoding="utf-8")
    mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
    mapping.write_text(json.dumps({"operation_id": operation_id}), encoding="utf-8")

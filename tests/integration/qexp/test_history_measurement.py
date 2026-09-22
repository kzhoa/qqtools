"""Validate that scale instrumentation observes both runtime JSON read paths."""

import hashlib
import json
import subprocess
import sys

import pytest

pytestmark = pytest.mark.integration


def test_history_measurement_detects_scalar_and_json_reads(tmp_path, checkout_subprocess_env):
    source = tmp_path / "settled-history-positive-control.json"
    source.write_text('{"state":"committed"}')
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,os,sys\n"
            "from pathlib import Path\n"
            "from tests.helpers.qexp.authority_measurement import AgentMeasurements\n"
            "from qqtools.plugins.qexp.runtime.store import read_json\n"
            "measurements=AgentMeasurements()\n"
            "measurements.install()\n"
            "path=Path(sys.argv[1])\n"
            "read_json(path)\n"
            "fd=os.open(path,os.O_RDONLY)\n"
            "try: os.read(fd,8192)\n"
            "finally: os.close(fd)\n"
            "print(json.dumps(dict(measurements.operations)))\n",
            str(source),
        ],
        env=checkout_subprocess_env,
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    assert json.loads(result.stdout)["MainThread"]["history.file_open"]["calls"] == 2


@pytest.mark.parametrize("shape", ["bulk", "single"])
def test_retained_fixture_preserves_idempotency_and_complete_observation(tmp_path, shape):
    from qqtools.plugins.qexp import init_shared_root, observer
    from qqtools.plugins.qexp.runtime.paths import idempotency_path
    from qqtools.plugins.qexp.runtime.store import read_json
    from qqtools.plugins.qexp.runtime.submission import decode_submission_plan, semantic_digest, submit_specs
    from tests.helpers.qexp.history_qualification import request_worker_set, seed_settled_history

    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "runtime")
    seed_settled_history(cfg, 3, shape)
    expected = {f"settled-history-{index:06d}" for index in range(3)}
    page = observer.list_tasks_page(cfg, page_size=10)
    assert {item["task_id"] for item in page["items"]} == expected
    assert page["next_cursor"] is None
    observed = set()
    sources = list((cfg.shared_root / "operations/submissions").glob("*.json"))
    assert len(sources) == (1 if shape == "bulk" else 3)
    for source in sources:
        operation_record = read_json(source)
        plan = decode_submission_plan(operation_record)
        operation = operation_record["submission"]
        context = operation["resolved_context"]
        assert list(plan.task_ids) == context["task_ids"]
        assert list(plan.tmux_overrides) == [None] * len(context["task_ids"])
        assert (
            operation["resolved_context_digest"]
            == hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        )
        assert operation["staged_task_count"] == len(context["task_ids"]) == len(context["task_specs"])
        key = operation["idempotency_key"]
        mapping = idempotency_path(cfg.shared_root, semantic_digest({"project": str(cfg.shared_root), "key": key}))
        assert read_json(mapping)["operation_id"] == operation["operation_id"]
        replay = submit_specs(
            cfg,
            operation["resolved_context"]["task_specs"],
            group_name=operation["target_group"],
            idempotency_key=key,
            kind=operation["kind"],
            worker_set=request_worker_set(context),
        )
        for task in replay:
            assert task.state["projection"] == "cancelled"
            assert task.submission_operation_id == operation["operation_id"]
            assert task.task_id not in observed
            observed.add(task.task_id)
    assert observed == expected


@pytest.mark.parametrize(
    ("context", "expected"),
    [
        ({"worker_set_additions": {}, "worker_set_declared": False}, None),
        ({"worker_set_additions": {}, "worker_set_declared": True}, {}),
        ({"worker_set_additions": {"worker": {}}}, {"worker": {}}),
    ],
)
def test_retained_fixture_reconstructs_worker_set_declaration(context, expected):
    from tests.helpers.qexp.history_qualification import request_worker_set

    assert request_worker_set(context) == expected


def test_retained_fixture_accepts_new_task_transitions_without_rebuild(tmp_path, monkeypatch):
    from qqtools.plugins.qexp import init_shared_root, scheduler, submit
    from qqtools.plugins.qexp.runtime.observation import projection
    from tests.helpers.qexp.history_qualification import seed_settled_history

    cfg = init_shared_root(tmp_path / "project/.qexp", "worker", runtime_root=tmp_path / "runtime")
    seed_settled_history(cfg, 1000, "bulk")
    original = projection.sync_task
    failures = []

    def checked(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            failures.append(repr(exc))
            raise

    monkeypatch.setattr(projection, "sync_task", checked)
    task = submit(cfg, ["true"], task_id="new-arrival")
    assert not failures
    attempt = scheduler.claim_task(cfg, task.task_id, [0])
    assert attempt is not None
    assert not failures
    scheduler.cancel_task(cfg, task.task_id)
    assert not failures
    assert projection.inspect_observation(cfg)["state"] == "active"

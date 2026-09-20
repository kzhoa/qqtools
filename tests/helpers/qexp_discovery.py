"""Real filesystem builders for Group discovery integration checks."""

import json

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.group_discovery.slice_io import SliceIO
from qqtools.plugins.qexp.runtime.group_namespace import activate_group_authority_locked
from qqtools.plugins.qexp.runtime.locks import schema_lock
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json


def source_file(path, *, operation="op-1", group="experiment", tasks=None, sequences=None, state="committed"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": {"schema_version": 6},
                "submission": {
                    "operation_id": operation,
                    "target_group": group,
                    "state": state,
                    "resolved_context": {"task_ids": ["task-a"] if tasks is None else tasks},
                    "commit_plan": {"group_membership_sequences": [1] if sequences is None else sequences},
                },
            }
        )
    )
    return path


def finish_source(session, *, io_bytes=262144, operations=32):
    for _ in range(20000):
        io = SliceIO(max_io_bytes=io_bytes, max_operations=operations)
        session.advance(io)
        assert io.io_bytes_used <= io_bytes
        assert io.operations_used <= operations
        if session.is_complete:
            return session.result
    raise AssertionError("source confirmation did not converge")


def close_source(session):
    session.request_close()
    for _ in range(10000):
        if session.is_closed:
            return
        session.advance(SliceIO())
    raise AssertionError("source confirmation retained resources")


def isolated_group(tmp_path, *, tail=1):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "g1", runtime_root=tmp_path / "runtime")
    create_group(cfg, "experiment")
    with schema_lock(cfg.shared_root):
        schema_path = cfg.shared_root / "schema/version.json"
        schema = read_json(schema_path)
        capabilities = schema["schema"]["required_capabilities"]
        if "local-recovery-v1" not in capabilities:
            capabilities.append("local-recovery-v1")
        atomic_replace(schema_path, schema)
        assert activate_group_authority_locked(cfg)
    set_tail(cfg, tail)
    return cfg


def set_tail(cfg, tail, *, pending=None):
    path = group_path(cfg.shared_root, "experiment")
    record = read_json(path)
    record["group"]["next_membership_sequence"] = tail + 1
    record["group"]["pending_submission_commit"] = pending
    atomic_replace(path, record)


def confirmed_source(cfg, coverage, operation, tasks, sequences):
    from qqtools.plugins.qexp.runtime.group_discovery.recovery import RecoverableSource

    path = source_file(
        submission_path(cfg.shared_root, operation), operation=operation, tasks=tasks, sequences=sequences
    )
    session = RecoverableSource(path, coverage.source_scratch(operation), operation, "experiment")
    try:
        return finish_source(session)
    finally:
        close_source(session)


def discover_group(cfg, group="experiment"):
    """Run the real background membership service to a certified checkpoint."""
    from qqtools.plugins.qexp.runtime.group_discovery.service import GroupDiscoveryService

    service = GroupDiscoveryService(cfg.shared_root, group)
    try:
        for _ in range(20000):
            if service.advance()["state"] == "complete":
                return
        raise AssertionError("membership discovery did not converge")
    finally:
        service.request_close()
        for _ in range(10000):
            if service.is_closed:
                break
            service.advance()
        assert service.is_closed

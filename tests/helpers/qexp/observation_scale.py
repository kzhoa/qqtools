"""Valid retained-Task fixtures for explicit observation scale qualification.

Fixture generation is not a writer-throughput benchmark. It writes terminal
Task records and constructs the index using the production radix-node builder;
all timed query calls subsequently use normal production I/O and validation.
"""

import hashlib
import json
from collections import defaultdict
from copy import deepcopy

from qqtools.plugins.qexp import init_shared_root, submit
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.observation import projection
from qqtools.plugins.qexp.runtime.observation.tree import IndexTree, partition_key
from qqtools.plugins.qexp.runtime.paths import group_path, submission_path, task_path
from qqtools.plugins.qexp.runtime.store import atomic_replace, read_json
from qqtools.plugins.qexp.runtime.submission import semantic_digest
from qqtools.plugins.qexp.runtime.submission_control import publish_submission


def retained_tasks(root, history_count):
    cfg = init_shared_root(root / "project/.qexp", "worker", runtime_root=root / "runtime")
    templates = {}
    operations = {}
    members = defaultdict(list)
    expected = {}
    for group in ("dense", "sparse"):
        create_group(cfg, group)
        task = submit(cfg, ["true"], task_id=f"a-{group}", group=group)
        templates[group] = task.to_dict()
        operations[group] = read_json(submission_path(cfg.shared_root, task.submission_operation_id))
        members[group].append(task.task_id)
        expected[task.task_id] = ("queued", group)
    for index in range(history_count):
        group = "sparse" if index % 997 == 0 else "dense"
        record = deepcopy(templates[group])
        task = record["task"]
        identifier = f"history-{index:06d}"
        task["task_id"] = identifier
        task["state"] = {"projection": "cancelled", "reason": "scale_fixture_completed_cancellation"}
        task["group_membership_sequence"] = len(members[group]) + 1
        task_path(cfg.shared_root, identifier).write_text(json.dumps(record), encoding="utf-8")
        members[group].append(identifier)
        expected[identifier] = ("cancelled", group)
    for group, identifiers in members.items():
        operation = operations[group]
        submission = operation["submission"]
        context = submission["resolved_context"]
        template = context["task_specs"][0]
        context["task_ids"] = identifiers
        context["task_specs"] = [dict(template, task_id=identifier) for identifier in identifiers]
        submission["resolved_context_digest"] = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()
        submission["staged_task_count"] = len(identifiers)
        submission["kind"] = "batch" if len(identifiers) > 1 else "single"
        submission["raw_request_digest"] = semantic_digest(
            {"group": group, "tasks": context["task_specs"], "worker_set": {}}
        )
        submission["commit_plan"]["group_membership_sequences"] = list(range(1, len(identifiers) + 1))
        publish_submission(cfg, operation)
        path = group_path(cfg.shared_root, group)
        group_record = read_json(path)
        group_record["group"]["next_membership_sequence"] = len(identifiers) + 1
        atomic_replace(path, group_record)
    build_observation(cfg, expected)
    return cfg, expected


def build_observation(cfg, expected):
    """Construct all historical partitions for a quiescent fixture truth set."""
    routes = defaultdict(list)
    for identifier, (phase, group) in expected.items():
        for route in {(None, None), (phase, None), (None, group), (phase, group)}:
            routes[route].append(identifier)
    state = projection.new_state(cfg, state="active")
    base = projection.initialize_generation(cfg, state)
    catalog = IndexTree(base / "catalog")
    for (phase, group), identifiers in routes.items():
        tree = IndexTree(base, phase=phase, group=group)
        tree.ensure_root()
        if len(identifiers) <= 128:
            for identifier in sorted(identifiers):
                tree.add(identifier)
        else:
            tree._persist_built(tree._build_split("", tuple(sorted(identifiers))))
        catalog.add(partition_key(phase, group))
    projection.write_state(cfg, state)

"""Coordinated conversion from historical Group records to the working set."""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Literal

from ...config_types import RootConfig
from .. import group_activity
from ..directory_capture import read_directory_entry
from ..group_discovery.activation_submission import advance_activation_submission_source
from ..group_namespace import group_directory
from ..group_working_set import enroll_group, initialize_index, is_index_initialized
from ..group_working_set_migration import (
    begin_audit,
    group_stream_fence,
    index_active,
    prepare_activation,
    publish_activation,
    read_schema,
    schema_capabilities,
    schema_path,
    sealed_proof_path,
)
from ..locks import schema_lock
from ..paths import shared_paths, submission_path
from ..protocol_compatibility import GROUP_WORKING_SET_CAPABILITY
from ..records import validate_group_name, validate_identifier
from ..store import authorized_migration_json_io
from .contracts import (
    DeterministicUpgradeError,
    MigrationPlugin,
    MigrationSpec,
    PhaseResult,
    UpgradeContext,
    UpgradeStorage,
)
from .group_record_stream import advance_group_source, publish_group_copy

_PAGE_SIZE = 32
_PAGE_BYTES = 4096
_GROUP_BYTES = 8 * 1024 * 1024
# One pending IndexTree transaction can replay before a new enrollment. Across
# both updates there are at most seven tree operations. Each descends at most
# 86 nodes and can split one leaf into at most 67 nodes of up to 64 KiB each.
_ENROLL_INDEX_METADATA_OPS = 1200
_ENROLL_INDEX_IO_BYTES = 7 * (86 + 67) * 64 * 1024 + 4 * 4096
_REPAIR_QUEUE_IO_BYTES = 3 * 66 * 65536


def _work_path(root: Path) -> Path:
    return shared_paths(root)["upgrade"] / GROUP_WORKING_SET_CAPABILITY


def _page_path(root: Path, page: int) -> Path:
    return _work_path(root) / f"groups-{page}.json"


def _directory_revision(path: Path, storage: UpgradeStorage) -> list[str | int]:
    storage.account_metadata_ops(1)
    value = path.lstat()
    if not stat.S_ISDIR(value.st_mode):
        raise DeterministicUpgradeError(f"Group namespace is not a directory: {path}")
    # Shared-root mounts can use different local device numbers.
    return [path.name, value.st_ino, value.st_mtime_ns, value.st_ctime_ns]


def _group_directory(root: Path, storage: UpgradeStorage) -> Path:
    storage.account_metadata_ops(3)
    return group_directory(root, storage=storage)


def _enroll(context: UpgradeContext, name: str) -> None:
    root = context.cfg.shared_root
    clock = _work_path(root) / "enrollment-clock.json"
    context.storage.atomic_replace(clock, {"group": name})
    context.storage.account_metadata_ops(_ENROLL_INDEX_METADATA_OPS + 1)
    context.slice_budget.consume_io_bytes(_ENROLL_INDEX_IO_BYTES)
    now = clock.stat().st_ctime_ns / 1_000_000_000
    with authorized_migration_json_io():
        enroll_group(root, name, now=now)


class GroupWorkingSetMigration(MigrationPlugin):
    """Fence released writers and certify the full published Group index."""

    spec = MigrationSpec(
        name=GROUP_WORKING_SET_CAPABILITY,
        source_protocol="schema-6",
        target_protocol="metadata:group-working-set-v1",
        compatible_readers=("schema-6",),
        compatible_writers=("metadata:group-working-set-v1",),
        phases=("expansion", "audit", "activation"),
        work_budget=_PAGE_SIZE,
        max_records_per_slice=_PAGE_SIZE,
        max_metadata_ops_per_slice=2048,
        max_io_bytes_per_slice=96 * 1024 * 1024,
        normal_level="L2",
        recovery_level="L2",
    )

    def is_applicable(self, cfg: RootConfig) -> bool:
        return not index_active(cfg.shared_root)

    def is_applicable_with_storage(self, cfg: RootConfig, storage: UpgradeStorage) -> bool:
        return not index_active(cfg.shared_root, storage=storage)

    def is_applicable_for_status(self, cfg: RootConfig, storage: UpgradeStorage) -> bool:
        return self.is_applicable_with_storage(cfg, storage)

    def validate_terminal_invariant(self, context: UpgradeContext) -> bool:
        root = context.cfg.shared_root
        if not index_active(root, storage=context.storage):
            return False
        context.storage.account_metadata_ops(2)
        context.slice_budget.consume_io_bytes(128)
        with authorized_migration_json_io():
            return is_index_initialized(root)

    def _cursor(self, context: UpgradeContext) -> dict[str, Any] | None:
        raw = context.journal["upgrade"]["migrations"][self.spec.name]["cursor"]
        if raw is None:
            return None
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {
            "directory",
            "offset",
            "pages",
            "max_pages",
            "next_page",
            "next_group",
            "cleanup",
        }:
            raise DeterministicUpgradeError("Group working-set migration cursor is malformed")
        generation = value["directory"]
        counters = [value[key] for key in ("offset", "pages", "max_pages", "next_page", "next_group", "cleanup")]
        if (
            not isinstance(generation, list)
            or len(generation) != 4
            or generation[0] not in {"groups", "groups-v2"}
            or any(type(item) is not int for item in generation[1:])
            or any(type(item) is not int or item < 0 for item in counters)
            or value["offset"] >= 1 << 63
            or not value["next_page"] <= value["pages"] <= value["max_pages"]
            or value["next_group"] >= _PAGE_SIZE
            or value["cleanup"] > value["max_pages"] + 1
        ):
            raise DeterministicUpgradeError("Group working-set migration cursor is invalid")
        return value

    def _result(
        self,
        context: UpgradeContext,
        state: Literal["progressed", "waiting", "complete"],
        cursor: dict[str, Any] | None,
        *,
        blocker: str | None = None,
    ) -> PhaseResult:
        usage = context.slice_budget.used(
            max_records=self.spec.max_records_per_slice,
            max_metadata_ops=self.spec.max_metadata_ops_per_slice,
            max_io_bytes=self.spec.max_io_bytes_per_slice,
        )
        return PhaseResult(
            state,
            work_items=usage["records"],
            metadata_ops=usage["metadata_ops"],
            io_bytes=usage["io_bytes"],
            cursor=json.dumps(cursor) if cursor is not None else None,
            blocker=blocker,
        )

    def expansion(self, context: UpgradeContext) -> PhaseResult:
        root = context.cfg.shared_root
        cursor = self._cursor(context)
        with schema_lock(root, blocking=False) as acquired:
            if not acquired:
                return self._result(context, "waiting", cursor, blocker="schema_lock_busy")
            return self._inventory(context, cursor)

    def _inventory(self, context: UpgradeContext, cursor: dict[str, Any] | None) -> PhaseResult:
        root, storage = context.cfg.shared_root, context.storage
        directory = _group_directory(root, storage)
        revision = _directory_revision(directory, storage)
        if cursor is None:
            schema = read_schema(root, storage=storage)
            capabilities = schema_capabilities(schema)
            prepare_activation(root, storage=storage)
            if GROUP_WORKING_SET_CAPABILITY not in capabilities:
                # QQTOOLS-COMPAT-0025: released writers reject the unknown
                # capability under their schema fence before mutating Group truth.
                schema["schema"]["required_capabilities"] = [*capabilities, GROUP_WORKING_SET_CAPABILITY]
                storage.atomic_replace(schema_path(root), schema)
            storage.account_metadata_ops(_ENROLL_INDEX_METADATA_OPS)
            context.slice_budget.consume_io_bytes(_ENROLL_INDEX_IO_BYTES)
            with authorized_migration_json_io():
                initialize_index(root)
            cursor = {
                "directory": revision,
                "offset": 0,
                "pages": 0,
                "max_pages": 0,
                "next_page": 0,
                "next_group": 0,
                "cleanup": 0,
            }
            return self._result(context, "progressed", cursor)
        if cursor["directory"] != revision:
            # A directory cookie is valid only for the captured generation.
            # Reuse numbered scratch pages and keep their cleanup high-water mark.
            cursor.update(directory=revision, offset=0, pages=0)

        names: list[str] = []
        complete = False
        for _ in range(context.work_budget):
            storage.account_metadata_ops(6)
            entry, cursor["offset"] = read_directory_entry(directory, cursor["offset"])
            if entry is None:
                complete = True
                break
            context.slice_budget.consume_records()
            if entry.endswith(".json"):
                name = entry.removesuffix(".json")
                validate_group_name(name)
                names.append(name)
        if _directory_revision(directory, storage) != revision:
            cursor.update(offset=0, pages=0)
            return self._result(context, "progressed", cursor)
        if names:
            storage.atomic_replace(_page_path(root, cursor["pages"]), {"groups": names})
            cursor["pages"] += 1
            cursor["max_pages"] = max(cursor["max_pages"], cursor["pages"])
        return self._result(context, "complete" if complete else "progressed", cursor)

    def audit(self, context: UpgradeContext) -> PhaseResult:
        root = context.cfg.shared_root
        cursor = self._cursor(context)
        if cursor is None:
            raise DeterministicUpgradeError("Group working-set inventory cannot be verified")
        with schema_lock(root, blocking=False) as acquired:
            if not acquired:
                return self._result(context, "waiting", cursor, blocker="schema_lock_busy")
            begin_audit(root, context.storage)
            if cursor["next_page"] == cursor["pages"]:
                return self._result(context, "complete", cursor)
            return self._audit_group(context, cursor)

    def _audit_group(self, context: UpgradeContext, cursor: dict[str, Any]) -> PhaseResult:
        root, storage = context.cfg.shared_root, context.storage
        if GROUP_WORKING_SET_CAPABILITY not in schema_capabilities(read_schema(root, storage=storage)):
            raise DeterministicUpgradeError("Group working-set writer fence disappeared")
        position = cursor["next_group"]
        page = storage.read_json_limited(_page_path(root, cursor["next_page"]), max_bytes=_PAGE_BYTES)
        names = page.get("groups")
        if not isinstance(names, list) or not position < len(names) <= _PAGE_SIZE:
            raise DeterministicUpgradeError("Group working-set inventory page is malformed")
        name = validate_group_name(names[position])
        if name is None:
            raise DeterministicUpgradeError("Group working-set inventory contains an empty name")
        context.slice_budget.consume_records()
        group_path = _group_directory(root, storage) / f"{name}.json"
        try:
            storage.account_metadata_ops(1)
            large = group_path.lstat().st_size > _GROUP_BYTES
            checkpoint = _work_path(root) / "group-checkpoint.json"
            if large:
                fence = group_stream_fence(root)
                if not storage.exists(fence):
                    storage.atomic_replace(fence, {"group": name})
                elif storage.read_json_limited(fence, max_bytes=4096) != {"group": name}:
                    raise DeterministicUpgradeError("Group stream fence does not match inventory position")
                record = advance_group_source(context, group_path, checkpoint)
                if record is None:
                    return self._result(context, "progressed", cursor)
            else:
                record = storage.read_json_limited(group_path, max_bytes=_GROUP_BYTES)
        except FileNotFoundError:
            if storage.exists(group_path):
                # A scratch file can disappear independently of canonical truth.
                return self._result(context, "progressed", cursor)
            record = None
        if record is not None:
            group = record.get("group")
            if not isinstance(group, dict) or group.get("name") != name:
                raise DeterministicUpgradeError(f"Group {name!r} has invalid canonical truth")
            creation_id = group.get("creation_operation_id")
            is_published = creation_id is None
            if creation_id is not None:
                operation_id = validate_identifier(creation_id, "creation_operation_id")
                step = advance_activation_submission_source(
                    storage,
                    submission_path(root, operation_id),
                    operation_id,
                    _work_path(root) / "submission-checkpoint.json",
                    "submissions",
                    mount_independent=True,
                )
                if step.state == "progressed":
                    return self._result(context, "progressed", cursor)
                if step.target_group != name:
                    raise DeterministicUpgradeError(f"Group {name!r} has a different Submission target")
                is_published = step.submission_state == "committed"
            if is_published:
                _enroll(context, name)
            if "admission_state" in group:
                if group["admission_state"] == "sealed":
                    storage.atomic_replace(sealed_proof_path(root, name), {"sealed": True})
                del group["admission_state"]
                if large:
                    try:
                        publish_group_copy(context, group_path, checkpoint)
                    except FileNotFoundError:
                        if not storage.exists(group_path):
                            raise
                        # Enrollment and the sealed proof are idempotent. Rebuild
                        # a lost stage from the still-fenced canonical record.
                        storage.unlink(checkpoint)
                        return self._result(context, "progressed", cursor)
                else:
                    storage.atomic_replace(group_path, record)
        storage.unlink(group_stream_fence(root))
        cursor["next_group"] += 1
        if cursor["next_group"] == len(names):
            cursor.update(next_page=cursor["next_page"] + 1, next_group=0)
        return self._result(context, "complete" if cursor["next_page"] == cursor["pages"] else "progressed", cursor)

    def _repair_activity(self, context: UpgradeContext, cursor: dict[str, Any]) -> PhaseResult | None:
        root, storage = context.cfg.shared_root, context.storage
        storage.account_metadata_ops(85)
        context.slice_budget.consume_io_bytes(66 * 65536)
        try:
            with authorized_migration_json_io():
                name = group_activity.next_bootstrap_repair(root)
        except BlockingIOError:
            return self._result(context, "waiting", cursor, blocker="activity_queue_busy")
        if name is None:
            return None
        validate_group_name(name)
        context.slice_budget.consume_records()
        path = _group_directory(root, storage) / f"{name}.json"
        try:
            storage.account_metadata_ops(1)
            if path.lstat().st_size > _GROUP_BYTES:
                fence = group_stream_fence(root)
                if not storage.exists(fence):
                    storage.atomic_replace(fence, {"group": name})
                record = advance_group_source(
                    context, path, _work_path(root) / "activity-group.json", copy_record=False
                )
                if record is None:
                    return self._result(context, "progressed", cursor)
            else:
                record = storage.read_json_limited(path, max_bytes=_GROUP_BYTES)
        except FileNotFoundError:
            if storage.exists(path):
                # A scratch file can disappear independently of canonical truth.
                return self._result(context, "progressed", cursor)
            record = None
        if record is not None:
            group = record["group"]
            if group.get("name") != name:
                raise DeterministicUpgradeError("Group creation proof has a different name")
            published = True
            if (creation := group.get("creation_operation_id")) is not None:
                creation = validate_identifier(creation, "creation_operation_id")
                step = advance_activation_submission_source(
                    storage,
                    submission_path(root, creation),
                    creation,
                    _work_path(root) / "activity-checkpoint.json",
                    "submissions",
                    mount_independent=True,
                )
                if step.state == "progressed":
                    return self._result(context, "progressed", cursor)
                if step.target_group != name:
                    raise DeterministicUpgradeError("Group creation proof has a different Submission target")
                published = step.submission_state == "committed"
            if published:
                _enroll(context, name)
        storage.account_metadata_ops(192)
        context.slice_budget.consume_io_bytes(_REPAIR_QUEUE_IO_BYTES)
        with authorized_migration_json_io():
            group_activity.finish_bootstrap_repair(root, name)
        storage.unlink(group_stream_fence(root))
        return self._result(context, "progressed", cursor)

    def activation(self, context: UpgradeContext) -> PhaseResult:
        # The coordinator holds schema and upgrade exclusion for activation.
        root, storage = context.cfg.shared_root, context.storage
        cursor = self._cursor(context)
        if cursor is None or cursor["next_page"] != cursor["pages"]:
            raise DeterministicUpgradeError("Group working-set inventory has not been verified")
        if GROUP_WORKING_SET_CAPABILITY not in schema_capabilities(read_schema(root, storage=storage)):
            raise DeterministicUpgradeError("Group working-set writer fence disappeared")
        if not index_active(root, storage=storage):
            if (repair := self._repair_activity(context, cursor)) is not None:
                return repair
            publish_activation(root, storage=storage)
        # An interrupted inventory page write can leave one page beyond the
        # journal's last acknowledged high-water mark.
        page_limit = cursor["max_pages"] + 1
        stop = min(page_limit, cursor["cleanup"] + 1)
        for page in range(cursor["cleanup"], stop):
            path = _page_path(root, page)
            try:
                names = storage.read_json_limited(path, max_bytes=_PAGE_BYTES)["groups"]
            except FileNotFoundError:
                names = []
            for name in names:
                storage.unlink(sealed_proof_path(root, validate_group_name(name)))
            storage.unlink(path)
            context.slice_budget.consume_records()
        cursor["cleanup"] = stop
        if stop < page_limit:
            return self._result(context, "progressed", cursor)
        storage.unlink(_work_path(root) / "submission-checkpoint.json")
        storage.unlink(_work_path(root) / "activity-checkpoint.json")
        for filename in (
            "group-checkpoint.json",
            "group-checkpoint.stage",
            "activity-group.json",
            "enrollment-clock.json",
        ):
            storage.unlink(_work_path(root) / filename)
        return self._result(context, "complete", None)

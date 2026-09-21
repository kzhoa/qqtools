"""Compatibility interpretation for existing qexp authority protocols.

This module owns policy only: it performs no storage mutation or lock acquisition.
Callers read durable evidence under their existing fences. Projection state cannot
lower the floor derived from schema and namespace activation records.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CPU_LANE_CAPABILITY = "cpu-lane-v1"
TASK_DEPENDENCIES_CAPABILITY = "task-dependencies-v1"
GROUP_READY_MEMBERS_CAPABILITY = "group-ready-members-v1"
LOCAL_RECOVERY_CAPABILITY = "local-recovery-v1"
GROUP_AUTHORITY_CAPABILITY = "group-authority-v2"
READY_WRITER_CAPABILITY = "ready-v1"
CURRENT_READY_WRITER_CAPABILITY = "ready-v2"
OBSERVATION_CAPABILITY = "task-observation-v1"
# QQTOOLS-COMPAT-0015: old readers must be fenced before Submission-owned
# provisional Groups can carry creation provenance.
SUBMISSION_GROUP_PUBLICATION_CAPABILITY = "submission-group-publication-v1"
SUPPORTED_READY_WRITERS = frozenset({READY_WRITER_CAPABILITY, CURRENT_READY_WRITER_CAPABILITY})
SUPPORTED_REQUIRED_CAPABILITIES = frozenset(
    {
        CPU_LANE_CAPABILITY,
        TASK_DEPENDENCIES_CAPABILITY,
        GROUP_READY_MEMBERS_CAPABILITY,
        LOCAL_RECOVERY_CAPABILITY,
        GROUP_AUTHORITY_CAPABILITY,
        OBSERVATION_CAPABILITY,
        SUBMISSION_GROUP_PUBLICATION_CAPABILITY,
    }
)


def minimum_ready_writer(schema: dict[str, Any], *, has_group_cutover: bool) -> str:
    """Resolve the monotonic writer floor from authoritative activation evidence."""
    if (
        CURRENT_READY_WRITER_CAPABILITY in schema.get("writer_capabilities", [])
        or GROUP_AUTHORITY_CAPABILITY in schema.get("required_capabilities", [])
        or has_group_cutover
    ):
        return CURRENT_READY_WRITER_CAPABILITY
    return READY_WRITER_CAPABILITY


def snapshot_digest(value: object) -> str:
    """Fingerprint an exact observed record for audit and repair freshness."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def manifest_schema_digest(schema: dict[str, Any]) -> str:
    """Fingerprint base schema without independently fenced recovery extensions.

    Audit evidence and repair plans still fingerprint the full version record.
    The terminal metadata manifest accepts only these named later capabilities.
    """
    record = schema.get("schema")
    if not isinstance(record, dict):
        return snapshot_digest(schema)
    record = dict(record)
    capabilities = record.get("required_capabilities")
    if isinstance(capabilities, list):
        record["required_capabilities"] = [
            item
            for item in capabilities
            if item
            not in {
                LOCAL_RECOVERY_CAPABILITY,
                GROUP_AUTHORITY_CAPABILITY,
                OBSERVATION_CAPABILITY,
                SUBMISSION_GROUP_PUBLICATION_CAPABILITY,
            }
        ]
    writers = record.get("writer_capabilities")
    if isinstance(writers, list):
        record["writer_capabilities"] = [item for item in writers if item != CURRENT_READY_WRITER_CAPABILITY]
    return snapshot_digest({**schema, "schema": record})


def matches_terminal_manifest_schema(schema: dict[str, Any], fingerprint: object) -> bool:
    """Recognize current metadata and the exact released bootstrap ordering."""
    return fingerprint == manifest_schema_digest(schema) or matches_released_bootstrap_schema(schema, fingerprint)


def matches_released_bootstrap_schema(schema: dict[str, Any], fingerprint: object) -> bool:
    """Recognize the released initializer before its later ready-v1 bootstrap."""
    record = schema.get("schema")
    if not isinstance(record, dict):
        return False
    writers = record.get("writer_capabilities")
    if not isinstance(writers, list) or [item for item in writers if item != CURRENT_READY_WRITER_CAPABILITY] != [
        READY_WRITER_CAPABILITY
    ]:
        return False
    original = {key: value for key, value in record.items() if key != "writer_capabilities"}
    return fingerprint == manifest_schema_digest({**schema, "schema": original})

"""Closed worker handler for isolated Group-service requests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config_types import RootConfig
from ..layout import load_root_config
from ..runtime.group_discovery.advance import advance_group_service
from ..runtime.group_discovery.probe import probe_group_service
from ..runtime.paths import machine_project_paths
from ..runtime.store import fenced_mutations
from .project_io_protocol import ProjectIORequest

_GROUP_SERVICE_OPERATION_KINDS = frozenset({"group_service_probe", "group_service_advance"})


def handle_group_service_request(
    request: ProjectIORequest,
    runtime_root: Path,
    before_shared_mutation: Callable[[RootConfig], None] | None,
) -> dict[str, Any]:
    """Execute one closed Group probe or fenced advance transaction."""
    if not isinstance(request, ProjectIORequest):
        raise ValueError("Group-service handling requires a ProjectIORequest.")
    if request.operation_kind not in _GROUP_SERVICE_OPERATION_KINDS:
        raise ValueError(f"unsupported Group-service operation: {request.operation_kind!r}")

    parameters = request.parameters
    shared_root = Path(request.canonical_shared_root)
    machine_name = parameters["machine_name"]
    if request.operation_kind == "group_service_probe":
        load_root_config(shared_root, machine_name)
        return probe_group_service(shared_root, parameters["probe_state"])

    if before_shared_mutation is None:
        raise ValueError("Group-service advance requires a shared mutation fence.")
    cfg = load_root_config(shared_root, machine_name)
    cfg = replace(cfg, runtime_root=machine_project_paths(Path(runtime_root), request.project_id)["root"])

    def apply_mutation_fence() -> None:
        before_shared_mutation(cfg)

    with fenced_mutations(cfg.shared_root, apply_mutation_fence):
        return advance_group_service(
            cfg,
            parameters["candidate"],
            parameters["continuation"],
            parameters["probe_state"],
        )


__all__ = ["handle_group_service_request"]

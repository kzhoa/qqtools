"""Isolation primitives shared by qexp tests.

These helpers deliberately live in the test tree: production qexp must not expose a
configuration switch that weakens its machine-authority semantics.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


def _safe_node_name(nodeid: str) -> str:
    """Return a filesystem-safe, bounded test identifier."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", nodeid).strip("._")
    return (value or "qexp-test")[:48]


@dataclass(frozen=True, slots=True)
class TestResourceScope:
    """Per-test qexp resource namespace with no production configuration surface."""

    __test__ = False

    run_id: str
    root: Path
    runtime_root: Path
    shared_root: Path
    local_temp_root: Path
    authority_root: Path
    home_root: Path
    xdg_root: Path
    tmux_socket: Path
    resource_ledger_root: Path
    diagnostics_root: Path

    @classmethod
    def create(cls, base_root: Path, nodeid: str) -> "TestResourceScope":
        run_id = f"{_safe_node_name(nodeid)}-{uuid.uuid4().hex[:12]}"
        root = base_root / run_id
        runtime_root = root / "runtime"
        shared_root = root / "shared"
        local_temp_root = root / "local-tmp"
        authority_root = local_temp_root / "authority"
        home_root = root / "home"
        xdg_root = root / "xdg"
        tmux_socket = root / "tmux" / "server.sock"
        resource_ledger_root = root / "resource-ledger"
        diagnostics_root = root / "diagnostics"
        for path in (
            runtime_root,
            shared_root,
            local_temp_root,
            authority_root,
            home_root,
            xdg_root,
            tmux_socket.parent,
            resource_ledger_root,
            diagnostics_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            root=root,
            runtime_root=runtime_root,
            shared_root=shared_root,
            local_temp_root=local_temp_root,
            authority_root=authority_root,
            home_root=home_root,
            xdg_root=xdg_root,
            tmux_socket=tmux_socket,
            resource_ledger_root=resource_ledger_root,
            diagnostics_root=diagnostics_root,
        )

    def child_environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Build an environment frozen before a participant imports qexp."""
        environment = dict(os.environ if base is None else base)
        environment.update(
            {
                "TMPDIR": str(self.local_temp_root),
                "TMP": str(self.local_temp_root),
                "TEMP": str(self.local_temp_root),
                "HOME": str(self.home_root),
                "XDG_CACHE_HOME": str(self.xdg_root / "cache"),
                "XDG_CONFIG_HOME": str(self.xdg_root / "config"),
                "XDG_DATA_HOME": str(self.xdg_root / "data"),
                "TMUX_TMPDIR": str(self.tmux_socket.parent),
                "QEXP_MACHINE_RUNTIME_ROOT": str(self.runtime_root),
            }
        )
        return environment

    def record_resource(self, kind: str, identity: dict[str, object]) -> Path:
        """Persist a test-owned external resource before cleanup can be attempted."""
        if not kind or any(character in kind for character in "/\\"):
            raise ValueError("resource kind must be a non-empty path component")
        path = self.resource_ledger_root / f"{uuid.uuid4().hex[:12]}-{kind}.json"
        path.write_text(
            json.dumps({"resource": {"kind": kind, "identity": identity}}, sort_keys=True),
            encoding="utf-8",
        )
        return path

    def record_cleanup_diagnostic(self, kind: str, details: dict[str, object]) -> Path:
        """Persist cleanup diagnostics without touching non-test resources."""
        if not kind or any(character in kind for character in "/\\"):
            raise ValueError("diagnostic kind must be a non-empty path component")
        path = self.diagnostics_root / f"{uuid.uuid4().hex[:12]}-{kind}.json"
        path.write_text(json.dumps({"diagnostic": details}, sort_keys=True), encoding="utf-8")
        return path

"""Read-only validation of machine identity and fresh-runtime prerequisites."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..runtime.store import read_json

_DIAGNOSIS_GUIDANCE = " Run 'qexp admin repair identity --dry-run' for a read-only diagnosis."


class MachineRuntimeIdentityError(RuntimeError):
    """Existing runtime state cannot establish a usable machine identity."""


class MachineRuntimeUninitializedError(MachineRuntimeIdentityError):
    """A machine-scoped command requires an initialized runtime identity."""


def load_identity_record(path: Path) -> dict[str, Any] | None:
    """Read a validated identity; only a missing file returns None."""
    try:
        value = read_json(path)
    except FileNotFoundError:
        # A dangling link is damaged state, not permission to create an identity.
        if path.is_symlink():
            raise MachineRuntimeIdentityError(
                f"machine runtime identity is inaccessible at {path}; check its target and mount." + _DIAGNOSIS_GUIDANCE
            ) from None
        return None
    except OSError as exc:
        raise MachineRuntimeIdentityError(
            f"machine runtime identity is unreadable at {path}; "
            "check QEXP_MACHINE_RUNTIME_ROOT, mount availability and access permissions." + _DIAGNOSIS_GUIDANCE
        ) from exc
    except ValueError as exc:
        raise MachineRuntimeIdentityError(
            f"machine runtime identity is malformed at {path}; "
            "preserve runtime data and recover the original identity from a verified backup." + _DIAGNOSIS_GUIDANCE
        ) from exc
    record = value.get("machine_runtime")
    if isinstance(record, dict):
        seed = record.get("instance_id")
        effective = record.get("runtime_id")
        if (
            isinstance(seed, str)
            and bool(seed)
            and (
                effective is None
                or (
                    isinstance(effective, str)
                    and len(effective) == 64
                    and all(char in "0123456789abcdef" for char in effective)
                )
            )
        ):
            return record
    raise MachineRuntimeIdentityError(
        f"machine runtime identity is malformed at {path}; "
        "preserve runtime data and recover the original identity from a verified backup." + _DIAGNOSIS_GUIDANCE
    )


def require_fresh_runtime(root: Path) -> None:
    """Reject missing identity beside durable data; empty layout and locks are harmless."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = directory / entry.name
                    if path == root / "locks" and entry.is_dir(follow_symlinks=False):
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(path)
                        continue
                    raise MachineRuntimeIdentityError(
                        f"machine runtime identity is missing at {root / 'identity.json'}, "
                        f"but existing runtime data remains at {path}; "
                        "preserve runtime data and recover the original identity before continuing."
                        + _DIAGNOSIS_GUIDANCE
                    )
        except FileNotFoundError:
            # Only an absent root establishes a fresh runtime. A disappearing
            # descendant cannot certify that an existing runtime has no data.
            if directory == root and not root.is_symlink():
                return
            raise MachineRuntimeIdentityError(
                f"cannot inspect machine runtime data at {directory}; check mount availability and retry."
                + _DIAGNOSIS_GUIDANCE
            ) from None
        except OSError as exc:
            raise MachineRuntimeIdentityError(
                f"cannot inspect machine runtime data at {directory}; "
                "check QEXP_MACHINE_RUNTIME_ROOT, mount availability and access permissions." + _DIAGNOSIS_GUIDANCE
            ) from exc

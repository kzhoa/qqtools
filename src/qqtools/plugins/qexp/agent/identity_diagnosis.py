"""Read-only diagnosis of MachineRuntime identity evidence."""

from __future__ import annotations

import os
import shlex
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..infrastructure.host import host_instance_id
from ..runtime.store import read_json
from .bindings import ProjectBinding
from .context import MACHINE_RUNTIME_ENV, MachineRuntime, default_machine_runtime_root
from .identity import MachineRuntimeIdentityError, load_identity_record, require_fresh_runtime
from .setup import SetupOperationalError, pending_machine_replacement

_RUNTIME_ID_LENGTH = 64


def diagnose_machine_identity(
    runtime_root: str | Path | None,
    *,
    selection_source: str,
) -> dict[str, object]:
    """Inspect identity, generation, registry, replacement and host evidence without writes."""
    reported_root = _reported_runtime_root(runtime_root)
    try:
        runtime = MachineRuntime(runtime_root)
    except (OSError, RuntimeError, ValueError):
        return _payload(
            outcome="failed",
            reason="access_failed",
            runtime_root=reported_root,
            selection_source=selection_source,
            affected_file=str(Path(reported_root) / "identity.json"),
            evidence_sources=[],
            checks=[_check("runtime_root", "failed", "The selected machine runtime path is unavailable or invalid.")],
            next_action="Check QEXP_MACHINE_RUNTIME_ROOT and filesystem access, then rerun the dry-run diagnosis.",
        )

    runtime_root_value = str(runtime.root)
    identity_path = runtime.paths["identity"]
    generation_path = runtime.paths["current_generation"]
    registry_path = runtime.paths["registry"]
    replacement_path = runtime.paths["replacement_transaction"]
    evidence_sources = [str(replacement_path)]

    if replacement_path.is_symlink() and not replacement_path.exists():
        return _failed_access(
            runtime_root_value,
            selection_source,
            str(replacement_path),
            evidence_sources,
            "replacement",
        )

    try:
        replacement = pending_machine_replacement(runtime)
    except SetupOperationalError as exc:
        return _payload(
            outcome="blocked",
            reason="replacement_pending",
            runtime_root=runtime_root_value,
            selection_source=selection_source,
            affected_file=str(replacement_path),
            evidence_sources=evidence_sources,
            checks=[_check("replacement", "blocked", "The replacement transaction is malformed or unsupported.")],
            next_action=(
                "Preserve replacement-transaction.json and have the machine owner validate its phase and archive "
                "target before resuming."
            ),
            replacement_error=str(exc),
        )
    except (TypeError, ValueError, KeyError) as exc:
        return _payload(
            outcome="blocked",
            reason="replacement_pending",
            runtime_root=runtime_root_value,
            selection_source=selection_source,
            affected_file=str(replacement_path),
            evidence_sources=evidence_sources,
            checks=[_check("replacement", "blocked", "The replacement transaction is malformed.")],
            next_action=(
                "Preserve replacement-transaction.json and have the machine owner validate its phase and "
                "archive target before resuming."
            ),
            replacement_error=str(exc),
        )
    except OSError:
        return _failed_access(
            runtime_root_value,
            selection_source,
            str(replacement_path),
            evidence_sources,
            "replacement",
        )

    if replacement is not None:
        target = replacement.get("target_name")
        phase = replacement.get("phase")
        return _payload(
            outcome="blocked",
            reason="replacement_pending",
            runtime_root=runtime_root_value,
            selection_source=selection_source,
            affected_file=str(replacement_path),
            evidence_sources=evidence_sources,
            checks=[
                _check(
                    "replacement",
                    "blocked",
                    "A validated identity replacement is pending and takes precedence over identity diagnosis.",
                )
            ],
            next_action=_replacement_resume_action(replacement),
            replacement_phase=phase,
            replacement_target=target,
        )

    evidence_sources.append(str(identity_path))
    try:
        identity = load_identity_record(identity_path)
    except MachineRuntimeIdentityError as exc:
        message = str(exc).lower()
        if "unreadable" in message or "inaccessible" in message:
            return _failed_access(
                runtime_root_value,
                selection_source,
                str(identity_path),
                evidence_sources,
                "identity_record",
            )
        return _payload(
            outcome="blocked",
            reason="insufficient_evidence",
            runtime_root=runtime_root_value,
            selection_source=selection_source,
            affected_file=str(identity_path),
            evidence_sources=evidence_sources,
            checks=[_check("identity_record", "blocked", "The machine identity record is malformed.")],
            next_action=(
                "Preserve runtime data and recover the original MachineRuntime identity from a verified backup; "
                "do not initialize this root."
            ),
        )
    except OSError:
        return _failed_access(
            runtime_root_value,
            selection_source,
            str(identity_path),
            evidence_sources,
            "identity_record",
        )

    if identity is None:
        try:
            require_fresh_runtime(runtime.root)
        except MachineRuntimeIdentityError as exc:
            message = str(exc).lower()
            if "cannot inspect" in message or "check mount availability" in message:
                return _failed_access(
                    runtime_root_value,
                    selection_source,
                    str(identity_path),
                    evidence_sources,
                    "runtime_freshness",
                )
            return _payload(
                outcome="blocked",
                reason="insufficient_evidence",
                runtime_root=runtime_root_value,
                selection_source=selection_source,
                affected_file=str(identity_path),
                evidence_sources=evidence_sources,
                checks=[
                    _check(
                        "identity_record",
                        "blocked",
                        "Identity is absent while other machine runtime data remains.",
                    )
                ],
                next_action=(
                    "Preserve runtime data and recover the original MachineRuntime identity before continuing; "
                    "do not initialize this root."
                ),
            )
        except OSError:
            return _failed_access(
                runtime_root_value,
                selection_source,
                str(identity_path),
                evidence_sources,
                "runtime_freshness",
            )
        return _payload(
            outcome="blocked",
            reason="uninitialized",
            runtime_root=runtime_root_value,
            selection_source=selection_source,
            affected_file=str(identity_path),
            evidence_sources=evidence_sources,
            checks=[_check("identity_record", "blocked", "No identity or other durable runtime data was found.")],
            next_action="Run qexp init --machine NAME to initialize this genuinely fresh machine runtime.",
        )

    identity_snapshot = dict(identity)
    seed = identity.get("instance_id")
    explicit_runtime_id = identity.get("runtime_id")
    checks = [
        _check("replacement", "passed", "No pending identity replacement was found."),
        _check("identity_record", "passed", "The machine identity record has a supported basic shape."),
    ]

    evidence_sources.extend((str(generation_path), str(registry_path), "host_instance_id()"))
    generation_status, generation_doc, generation_error = _read_optional_json(generation_path)
    registry_status, registry_doc, registry_error = _read_optional_json(registry_path)
    if generation_status == "error" or registry_status == "error":
        failed_file = generation_path if generation_status == "error" else registry_path
        failed_name = "generation" if generation_status == "error" else "registry"
        return _failed_access(
            runtime_root_value,
            selection_source,
            str(failed_file),
            evidence_sources,
            failed_name,
        )

    generation, generation_problem = _parse_generation(generation_status, generation_doc)
    registry, registry_problem = _parse_registry(registry_status, registry_doc)

    try:
        host_token = host_instance_id()
    except (OSError, RuntimeError, ValueError):
        return _failed_access(
            runtime_root_value,
            selection_source,
            str(identity_path),
            evidence_sources,
            "host_continuity",
        )
    computed_runtime_id = sha256(f"{seed}\0{host_token}".encode()).hexdigest()
    effective_runtime_id = explicit_runtime_id if explicit_runtime_id is not None else computed_runtime_id

    if generation_problem is not None:
        if generation_problem == "missing" and explicit_runtime_id is None:
            checks.append(
                _check(
                    "generation",
                    "blocked",
                    "This legacy identity has no current-generation record, so generation consistency is unavailable.",
                )
            )
        else:
            checks.append(_check("generation", "blocked", "Current-generation evidence is absent or malformed."))
    elif generation is not None and generation["runtime_id"] == effective_runtime_id:
        checks.append(_check("generation", "passed", "Current-generation version and runtime ID are consistent."))
    elif explicit_runtime_id is None:
        checks.append(
            _check("generation", "blocked", "The host-derived identity differs from current-generation evidence.")
        )
    else:
        checks.append(_check("generation", "blocked", "The stored identity and current generation disagree."))

    mismatched_bindings: list[ProjectBinding] = []
    incomplete_bindings: list[ProjectBinding] = []
    foreign_root_bindings: list[ProjectBinding] = []
    if registry_problem == "missing":
        checks.append(_check("binding_consistency", "blocked", "Required machine registry evidence is absent."))
    elif registry_problem is not None:
        checks.append(_check("binding_consistency", "blocked", "Machine registry shape or bindings are invalid."))
    else:
        mismatched_bindings = [
            binding
            for binding in registry or []
            if binding.runtime_instance_id is not None and binding.runtime_instance_id != effective_runtime_id
        ]
        incomplete_bindings = [
            binding
            for binding in registry or []
            if binding.runtime_instance_id is None
            or binding.registration_generation is None
            or binding.runtime_root is None
        ]
        foreign_root_bindings = [
            binding
            for binding in registry or []
            if binding.runtime_root is not None and binding.runtime_root != runtime_root_value
        ]
        if mismatched_bindings:
            checks.append(
                _check(
                    "binding_consistency",
                    "blocked",
                    f"{len(mismatched_bindings)} registry binding(s) reference a different runtime identity.",
                )
            )
        elif foreign_root_bindings:
            checks.append(
                _check(
                    "binding_consistency",
                    "blocked",
                    f"{len(foreign_root_bindings)} registry binding(s) reference another machine runtime root.",
                )
            )
        elif incomplete_bindings:
            checks.append(
                _check(
                    "binding_consistency",
                    "blocked",
                    f"{len(incomplete_bindings)} registry binding(s) lack runtime, root or registration-generation evidence.",
                )
            )
        else:
            checks.append(
                _check(
                    "binding_consistency",
                    "passed",
                    f"All non-null binding runtime IDs match across {len(registry or [])} binding(s).",
                )
            )

    generation_matches = generation is not None and generation["runtime_id"] == effective_runtime_id
    seed_host_mismatch = (
        generation is not None and explicit_runtime_id is None and generation["runtime_id"] != computed_runtime_id
    )
    if generation_problem == "missing" and explicit_runtime_id is None:
        host_status = "blocked"
        host_detail = "A legacy identity without current-generation evidence cannot prove host continuity."
    elif generation_matches and explicit_runtime_id is None:
        host_status = "passed"
        host_detail = "The host-derived identity matches the published current generation."
    elif generation_matches and explicit_runtime_id == computed_runtime_id:
        host_status = "passed"
        host_detail = "The explicit runtime ID matches both this host and the published current generation."
    elif generation_matches:
        host_status = "blocked"
        host_detail = "The explicit runtime ID is valid, but current-host continuity is not independently proven."
    elif seed_host_mismatch:
        host_status = "blocked"
        host_detail = "The host-derived identity differs from the published current generation."
    else:
        host_status = "not_checked"
        host_detail = "Generation evidence did not agree with the identity, so host continuity cannot be established."
    checks.append(_check("host_continuity", host_status, host_detail))

    if generation_problem == "invalid":
        reason = "inconsistent_generation"
        outcome = "blocked"
        affected_file = str(generation_path)
        next_action = "Preserve state and repair the current-generation evidence at machine level."
    elif seed_host_mismatch:
        reason = "host_mismatch"
        outcome = "blocked"
        affected_file = str(generation_path)
        next_action = "Use the original host runtime or recover at machine level; do not replace identity on this host."
    elif registry_problem == "missing":
        reason = "insufficient_evidence"
        outcome = "blocked"
        affected_file = str(registry_path)
        next_action = "Restore registry.json from a verified source before relying on this machine runtime."
    elif registry_problem is not None or mismatched_bindings or foreign_root_bindings:
        reason = "inconsistent_generation"
        outcome = "blocked"
        affected_file = str(registry_path)
        next_action = "Preserve state and reconcile registry.json against identity and current-generation evidence."
    elif incomplete_bindings:
        reason = "insufficient_evidence"
        outcome = "blocked"
        affected_file = str(registry_path)
        next_action = (
            "Preserve state and verify the runtime root, registration generation and runtime ID for each binding."
        )
    elif generation_problem is not None and not (generation_problem == "missing" and explicit_runtime_id is None):
        reason = "inconsistent_generation"
        outcome = "blocked"
        affected_file = str(generation_path)
        next_action = "Preserve state and reconcile current-generation.json with the stored identity and registry."
    elif generation is not None and explicit_runtime_id is not None and not generation_matches:
        reason = "inconsistent_generation"
        outcome = "blocked"
        affected_file = str(generation_path)
        next_action = "Preserve state and reconcile current-generation.json with the stored identity and registry."
    elif host_status == "blocked":
        reason = "host_continuity_unverified"
        outcome = "blocked"
        affected_file = str(generation_path)
        next_action = "Verify this runtime on its original host or obtain independent host-continuity evidence."
    else:
        reason = "identity_consistent"
        outcome = "healthy"
        affected_file = str(identity_path)
        next_action = "Continue normal qexp machine operations; no identity change is indicated."

    changed, access_failure = _evidence_changed(
        runtime,
        identity_path,
        generation_path,
        registry_path,
        identity_snapshot,
        generation_status,
        generation_doc,
        registry_status,
        registry_doc,
    )
    if access_failure is not None:
        return _failed_access(
            runtime_root_value,
            selection_source,
            access_failure,
            evidence_sources,
            "consistent_observation",
        )
    if changed:
        checks.append(
            _check(
                "consistent_observation",
                "blocked",
                "Identity, generation or registry evidence changed during diagnosis.",
            )
        )
        return _payload(
            outcome="blocked",
            reason="inconsistent_generation",
            runtime_root=runtime_root_value,
            selection_source=selection_source,
            affected_file=str(identity_path),
            evidence_sources=evidence_sources,
            checks=checks,
            next_action="Stop concurrent identity or registry changes and rerun the dry-run diagnosis.",
        )
    checks.append(_check("consistent_observation", "passed", "Identity, generation and registry stayed stable."))
    if generation_problem is not None and generation_status == "invalid":
        affected_file = str(generation_path)
    if generation_status == "invalid":
        reason = "inconsistent_generation"
        outcome = "blocked"
        next_action = "Preserve state and repair the current-generation evidence at machine level."
    if registry_status == "invalid":
        reason = "inconsistent_generation"
        outcome = "blocked"
        affected_file = str(registry_path)
        next_action = "Preserve state and repair registry.json at machine level."
    result = _payload(
        outcome=outcome,
        reason=reason,
        runtime_root=runtime_root_value,
        selection_source=selection_source,
        affected_file=affected_file,
        evidence_sources=evidence_sources,
        checks=checks,
        next_action=next_action,
    )
    if generation_problem is not None:
        result["generation_error"] = generation_error or generation_problem
    if registry_problem is not None:
        result["registry_error"] = registry_error or registry_problem
    return result


def _read_optional_json(path: Path) -> tuple[str, dict[str, Any] | None, str | None]:
    try:
        return "present", read_json(path), None
    except FileNotFoundError:
        if path.is_symlink():
            return "error", None, "dangling symbolic link"
        return "missing", None, None
    except OSError:
        return "error", None, "unreadable"
    except ValueError:
        return "invalid", None, "malformed JSON or non-object document"


def _parse_generation(
    status: str,
    document: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    if status == "missing":
        return None, "missing"
    if status != "present" or document is None:
        return None, "invalid"
    current = document.get("current_generation")
    if not isinstance(current, dict):
        return None, "invalid"
    runtime_id = current.get("runtime_id")
    if type(current.get("version")) is not int or current["version"] != 1 or not _is_runtime_id(runtime_id):
        return None, "invalid"
    return current, None


def _parse_registry(
    status: str,
    document: dict[str, Any] | None,
) -> tuple[list[ProjectBinding] | None, str | None]:
    if status == "missing":
        return None, "missing"
    if status != "present" or document is None:
        return None, "invalid"
    registry = document.get("registry")
    if not isinstance(registry, dict):
        return None, "invalid"
    revision = registry.get("revision")
    raw_bindings = registry.get("bindings")
    if (
        type(registry.get("version")) is not int
        or registry["version"] != 1
        or type(revision) is not int
        or revision < 0
        or not isinstance(raw_bindings, list)
    ):
        return None, "invalid"
    try:
        bindings = [ProjectBinding.from_dict(item) for item in raw_bindings]
    except (KeyError, TypeError, ValueError):
        return None, "invalid"
    return bindings, None


def _evidence_changed(
    runtime: MachineRuntime,
    identity_path: Path,
    generation_path: Path,
    registry_path: Path,
    identity_snapshot: dict[str, Any],
    generation_status: str,
    generation_doc: dict[str, Any] | None,
    registry_status: str,
    registry_doc: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    replacement_path = runtime.paths["replacement_transaction"]
    if replacement_path.is_symlink() and not replacement_path.exists():
        return False, str(replacement_path)
    try:
        if pending_machine_replacement(runtime) is not None:
            return True, None
    except (SetupOperationalError, ValueError, TypeError, KeyError):
        return True, None
    except OSError:
        return False, str(replacement_path)
    try:
        current_identity = load_identity_record(identity_path)
    except MachineRuntimeIdentityError as exc:
        if "unreadable" in str(exc).lower() or "inaccessible" in str(exc).lower():
            return False, str(identity_path)
        return True, None
    except OSError:
        return False, str(identity_path)
    if current_identity != identity_snapshot:
        return True, None
    current_generation_status, current_generation_doc, _ = _read_optional_json(generation_path)
    if current_generation_status == "error":
        return False, str(generation_path)
    current_registry_status, current_registry_doc, _ = _read_optional_json(registry_path)
    if current_registry_status == "error":
        return False, str(registry_path)
    if current_generation_status != generation_status or current_generation_doc != generation_doc:
        return True, None
    if current_registry_status != registry_status or current_registry_doc != registry_doc:
        return True, None
    return False, None


def _failed_access(
    runtime_root: str,
    selection_source: str,
    affected_file: str,
    evidence_sources: list[str],
    check_name: str,
) -> dict[str, object]:
    return _payload(
        outcome="failed",
        reason="access_failed",
        runtime_root=runtime_root,
        selection_source=selection_source,
        affected_file=affected_file,
        evidence_sources=evidence_sources,
        checks=[_check(check_name, "failed", "Required machine identity evidence could not be read safely.")],
        next_action="Restore access to the affected runtime evidence, then rerun qexp admin repair identity --dry-run.",
    )


def _replacement_resume_action(transaction: dict[str, Any]) -> str:
    target = shlex.quote(str(transaction.get("target_name", "NAME")))
    command = ["qexp", "init", "--machine", target]
    policy = transaction.get("policy")
    if isinstance(policy, str) and policy:
        command.extend(("--agent-mode", shlex.quote(policy)))
    if transaction.get("detach_old_runtime") is True:
        command.append("--detach-old-runtime")
    command.append("--yes")
    return f"Resume the validated replacement with {' '.join(command)} using its recorded options."


def _reported_runtime_root(runtime_root: str | Path | None) -> str:
    configured = runtime_root if runtime_root is not None else os.environ.get(MACHINE_RUNTIME_ENV)
    if configured:
        path = Path(configured).expanduser()
        try:
            return str(path.resolve())
        except (OSError, RuntimeError):
            return str(path)
    return str(default_machine_runtime_root())


def _is_runtime_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _RUNTIME_ID_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _payload(
    *,
    outcome: str,
    reason: str,
    runtime_root: str,
    selection_source: str,
    affected_file: str,
    evidence_sources: list[str],
    checks: list[dict[str, str]],
    next_action: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "outcome": outcome,
        "reason": reason,
        "runtime_root": runtime_root,
        "selection_source": selection_source,
        "affected_file": affected_file,
        "evidence_sources": evidence_sources,
        "checks": checks,
        "planned_changes": [],
        "next_action": next_action,
        "configuration_scope": "not_checked",
        **extra,
    }


__all__ = ["diagnose_machine_identity"]

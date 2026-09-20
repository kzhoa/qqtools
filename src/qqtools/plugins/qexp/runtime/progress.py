"""Attempt-bound application observations, isolated from execution authority.

The passive runner provisions only machine-local channel identity. The agent's
separate observation thread is the sole shared progress writer. Nothing in this
module renews an Attempt lease, changes Task/Attempt truth, sends a signal, or
releases a resource.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qqtools.qexp._progress_protocol import (
    MAX_PAYLOAD_BYTES,
    identifier,
    read_advisory_snapshot,
    replace_advisory_snapshot,
    semantic_key,
    validate_payload,
)

from .locks import exclusive
from .progress_types import ProgressObservation

_LOCAL_DIRS = ("progress", "progress-contexts", "progress-observed", "progress-diagnostics")
_IDENTITY = (
    "task_id",
    "attempt_id",
    "attempt_number",
    "machine_name",
    "launch_id",
    "wrapper_pid",
    "wrapper_start_time_ticks",
)
_TERMINAL = frozenset(("succeeded", "failed", "cancelled"))
_DEFAULT_PROGRESS_INTERVAL_SECONDS = 30
_POLICY_VERSION = 1


@dataclass(frozen=True, slots=True)
class ProgressChannel:
    """Frozen local channel returned after one successful context provision."""

    path: str
    interval_seconds: int | float | None

    def __str__(self) -> str:
        return self.path

    def __fspath__(self) -> str:
        return self.path


def _valid_interval(value: Any) -> int | float:
    if type(value) not in (int, float) or value < 1:
        raise ValueError("progress interval_seconds must be a finite number of at least 1 second")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("progress interval_seconds must be a finite number of at least 1 second")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_progress_path(runtime_root: Path, attempt_id: str) -> Path:
    """Return the producer mailbox inside an Attempt-owned disposable directory."""
    return Path(runtime_root) / "progress" / identifier(attempt_id) / "latest.json"


def shared_progress_path(shared_root: Path, task_id: str, attempt_id: str) -> Path:
    return Path(shared_root) / "progress" / identifier(task_id) / f"{identifier(attempt_id)}.json"


def _progress_lock_path(shared_root: Path, task_id: str) -> Path:
    return Path(shared_root) / "locks" / "progress" / f"{identifier(task_id)}.lock"


def _local_path(cfg: Any, directory: str, attempt_id: str) -> Path:
    return Path(cfg.runtime_root) / directory / f"{identifier(attempt_id)}.json"


def record_progress_diagnostic(cfg: Any, attempt_id: str, reason: str) -> None:
    """Best-effort bounded policy/launch diagnostic after context creation."""
    try:
        context_path = _local_path(cfg, "progress-contexts", attempt_id)
        if not context_path.exists():
            return
        path = _local_path(cfg, "progress-diagnostics", attempt_id)
        try:
            previous = read_advisory_snapshot(path)
            elapsed = (
                datetime.now(timezone.utc) - datetime.fromisoformat(previous["at"].replace("Z", "+00:00"))
            ).total_seconds()
            if elapsed < 60:
                return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        replace_advisory_snapshot(
            path,
            {"attempt_id": attempt_id, "reason": str(reason)[:160], "at": _now()},
        )
    except Exception:
        pass


def has_local_progress_mailbox(runtime_root: Path) -> bool:
    """Check local producer evidence without touching the shared filesystem."""
    root = Path(runtime_root) / "progress"
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    if (Path(entry.path) / "latest.json").is_file():
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def prepare_progress_channel(
    cfg: Any,
    task: Any,
    attempt: Any,
    *,
    wrapper_start_time_ticks: int | None,
    interval_seconds: int | float = _DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> ProgressChannel | None:
    """Provision only machine-local advisory state for one authorized launch.

    This function deliberately performs no shared-root I/O. It is called after
    the runner releases launch authority locks. The context is identity evidence,
    not authorization, and every shared projection is independently revalidated.
    """
    try:
        if type(wrapper_start_time_ticks) is not int:
            return None
        frozen_interval = _valid_interval(interval_seconds)
        context = {
            "protocol_version": 1,
            "reporting_policy_version": _POLICY_VERSION,
            "interval_seconds": frozen_interval,
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "machine_name": cfg.machine_name,
            "launch_id": attempt.authorization["launch_id"],
            "wrapper_pid": os.getpid(),
            "wrapper_start_time_ticks": wrapper_start_time_ticks,
        }
        path = local_progress_path(cfg.runtime_root, attempt.attempt_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        for directory in ("progress-contexts", "progress-observed", "progress-diagnostics"):
            (Path(cfg.runtime_root) / directory).mkdir(parents=True, exist_ok=True)
        context_path = _local_path(cfg, "progress-contexts", attempt.attempt_id)
        if context_path.exists():
            existing = read_advisory_snapshot(context_path)
            try:
                existing_interval = _valid_interval(existing.get("interval_seconds"))
            except ValueError:
                return None
            if (
                existing.get("protocol_version") != 1
                or existing.get("reporting_policy_version") != _POLICY_VERSION
                or any(existing.get(key) != context.get(key) for key in _IDENTITY)
            ):
                # Existing malformed or mismatched context is evidence that
                # this launch cannot safely reuse the channel.  Never replace
                # runner-owned identity evidence in place.
                return None
            return ProgressChannel(str(path), existing_interval)
        replace_advisory_snapshot(context_path, context)
        return ProgressChannel(str(path), frozen_interval)
    except Exception:
        return None


def resolve_progress_binding(cfg: Any, context: dict[str, Any]) -> dict[str, Any] | None:
    """Bind runner identity to current Attempt truth without mutating authority.

    ``None`` means superseded/cleaned. Transient incomplete authority raises so
    the local context survives for a later observation. Terminal publication
    clears ``current_attempt_id`` by qexp contract, therefore terminal history is
    joined by the preserved current Attempt number.
    """
    from .paths import attempt_path
    from .records import AttemptRecord
    from .store import read_json
    from .tasks import load_task

    task_id = identifier(context.get("task_id"))
    attempt_id = identifier(context.get("attempt_id"))
    try:
        task = load_task(cfg, task_id)
    except FileNotFoundError:
        return None
    if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
        return None
    number = task.attempt_control.get("current_attempt_number")
    if type(number) is not int or number != context.get("attempt_number"):
        return None
    task_terminal = task.state["projection"] in _TERMINAL
    current_attempt_id = task.attempt_control.get("current_attempt_id")
    if task_terminal:
        if current_attempt_id not in {None, attempt_id}:
            return None
    elif current_attempt_id != attempt_id:
        return None
    attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, number)))
    if attempt.attempt_id != attempt_id or attempt.task_id != task_id or attempt.machine_name != cfg.machine_name:
        return None
    if context.get("machine_name") != attempt.machine_name or context.get("launch_id") != attempt.authorization.get(
        "launch_id"
    ):
        return None
    for key in ("wrapper_pid", "wrapper_start_time_ticks"):
        if type(context.get(key)) is not int or attempt.process.get(key) != context[key]:
            raise ValueError("progress process identity not yet verified")
    terminal = task_terminal and attempt.phase in _TERMINAL
    if task_terminal and not terminal:
        raise ValueError("terminal progress transition incomplete")
    if not terminal:
        claim = task.claim_control.get("active_claim") or {}
        if (
            task.state["projection"] != "running"
            or attempt.phase not in {"starting", "running"}
            or claim.get("attempt_id") != attempt_id
            or claim.get("fencing_token") != attempt.current_fencing_token
            or claim.get("machine_name") != cfg.machine_name
        ):
            raise ValueError("progress authority currently unavailable")
    return {
        **{key: context[key] for key in _IDENTITY},
        "fencing_token": attempt.current_fencing_token,
        "terminal": terminal,
    }


def _validate_projection(
    value: Any,
    identity: dict[str, Any],
    *,
    require_token: bool = False,
    require_generation: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("protocol_version")) is not int or value["protocol_version"] != 1:
        raise ValueError("unsupported progress snapshot")
    expected = {
        "protocol_version",
        *_IDENTITY,
        "registration_generation",
        "fencing_token",
        "source_update_id",
        "sequence",
        "reported_at",
        "advanced_at",
        "progress",
    }
    if set(value) != expected or not isinstance(value.get("progress"), dict):
        raise ValueError("invalid progress snapshot fields")
    if any(value.get(key) != identity.get(key) for key in _IDENTITY):
        raise ValueError("progress snapshot identity mismatch")
    identifier(value.get("registration_generation"))
    if require_generation and value["registration_generation"] != identity.get("registration_generation"):
        raise ValueError("superseded progress registration generation")
    if type(value.get("fencing_token")) is not int:
        raise ValueError("invalid progress fencing token")
    if require_token and value["fencing_token"] != identity.get("fencing_token"):
        raise ValueError("superseded progress snapshot")
    if type(value.get("sequence")) is not int or value["sequence"] < 1:
        raise ValueError("invalid progress sequence")
    for key in ("reported_at", "advanced_at"):
        if not isinstance(value[key], str):
            raise ValueError("invalid progress timestamp")
        stamp = datetime.fromisoformat(value[key].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("progress timestamp must have a timezone")
    normalized = validate_payload({**value["progress"], "protocol_version": 1, "update_id": value["source_update_id"]})
    if any(key in value["progress"] for key in ("protocol_version", "update_id")):
        raise ValueError("invalid nested progress payload")
    result = dict(value)
    result["progress"] = {key: item for key, item in normalized.items() if key not in {"protocol_version", "update_id"}}
    return result


def _signature(value: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        value["source_update_id"],
        value["sequence"],
        value["fencing_token"],
        value["registration_generation"],
    )


def _payload_observation_key(value: dict[str, Any]) -> tuple[Any, ...]:
    payload = value.get("progress", value)
    return tuple(payload.get(name) for name in ("stage", "current", "total", "unit", "message"))


def _mailbox_signature(path: Path) -> tuple[int, int, int, int] | None:
    """Return a cheap regular-file witness; ``None`` means metadata is ambiguous."""
    try:
        info = path.lstat()
    except OSError:
        return None
    import stat

    if not stat.S_ISREG(info.st_mode):
        return None
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _attempt_offset(attempt_id: str, interval: float) -> float:
    import hashlib

    digest = hashlib.sha256(attempt_id.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return fraction * interval


def _initial_deadline(attempt_id: str, now: float, interval: float) -> float:
    """Defer at least one interval and spread subsequent ordinary work."""
    return now + interval + _attempt_offset(attempt_id, interval)


def _next_deadline(deadline: float, now: float, interval: float) -> float:
    """Advance on the existing phase while skipping every missed slot."""
    if not math.isfinite(deadline):
        return float("inf")
    target = now + interval
    if not math.isfinite(target):
        return float("inf")
    ratio = (target - deadline) / interval
    rounded = round(ratio)
    steps = max(
        1,
        rounded if math.isclose(ratio, rounded, rel_tol=1e-12, abs_tol=1e-12) else math.ceil(ratio),
    )
    candidate = deadline + steps * interval
    if not math.isfinite(candidate):
        return float("inf")
    return max(candidate, target)


def _context_interval(context: dict[str, Any]) -> float | None:
    fields = ("reporting_policy_version", "interval_seconds")
    if not any(field in context for field in fields):
        return None
    if context.get("reporting_policy_version") != _POLICY_VERSION:
        raise ValueError("unsupported progress reporting policy version")
    value = _valid_interval(context.get("interval_seconds"))
    try:
        return float(value)
    except OverflowError:
        return float("inf")


class ProgressProjector:
    """Bounded latest-only ingestion owned by one machine registration generation."""

    def __init__(
        self,
        cfg: Any,
        *,
        registration_generation: str,
        clock=time.monotonic,
        wall_clock=_now,
        resolver=resolve_progress_binding,
    ) -> None:
        self.cfg = cfg
        self._registration_generation = identifier(registration_generation)
        self._clock = clock
        self._wall_clock = wall_clock
        self._resolve = resolver
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._scan = None

    def close(self) -> None:
        if self._scan is not None:
            self._scan.close()
            self._scan = None

    def _binding(self, context: dict[str, Any]) -> dict[str, Any] | None:
        binding = self._resolve(self.cfg, context)
        if binding is None:
            return None
        return {**binding, "registration_generation": self._registration_generation}

    def tick(self, *, budget: int = 64) -> None:
        """Observe at most ``budget`` contexts; malformed producers never abort a sweep."""
        try:
            if self._scan is None:
                self._scan = os.scandir(Path(self.cfg.runtime_root) / "progress-contexts")
            for _ in range(max(1, min(budget, 256))):
                try:
                    entry = next(self._scan)
                except StopIteration:
                    self.close()
                    break
                if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                    continue
                attempt_id = entry.name[:-5]
                try:
                    identifier(attempt_id)
                    self.observe(attempt_id)
                except Exception:
                    self._diagnostic(attempt_id, "invalid_or_unavailable_progress")
        except Exception:
            self.close()

    def _write_local(self, directory: str, attempt_id: str, value: dict[str, Any]) -> None:
        context_path = _local_path(self.cfg, "progress-contexts", attempt_id)
        if not context_path.exists():
            return
        path = _local_path(self.cfg, directory, attempt_id)
        replace_advisory_snapshot(path, value)
        if not context_path.exists():
            path.unlink(missing_ok=True)

    def _diagnostic(self, attempt_id: str, reason: str, *, interval: float | None = None) -> None:
        try:
            context_path = _local_path(self.cfg, "progress-contexts", attempt_id)
            if not context_path.exists():
                return
            path = _local_path(self.cfg, "progress-diagnostics", attempt_id)
            try:
                previous = read_advisory_snapshot(path)
                elapsed = (
                    datetime.now(timezone.utc) - datetime.fromisoformat(previous["at"].replace("Z", "+00:00"))
                ).total_seconds()
                spacing = max(60.0, interval or 0.0)
                if elapsed < spacing:
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass
            self._write_local(
                "progress-diagnostics",
                attempt_id,
                {"attempt_id": attempt_id, "reason": reason, "at": _now()},
            )
        except Exception:
            pass

    def _restore(self, attempt_id: str, binding: dict[str, Any], *, interval: float | None = None) -> dict[str, Any]:
        """Restore freshness locally across agent-generation changes without trusting stale shared writers."""
        shared_path = shared_progress_path(self.cfg.shared_root, binding["task_id"], attempt_id)
        local_latest = None
        shared_latest = None
        try:
            # The local accepted observation is bound to runner process identity,
            # so a new machine registration generation may reuse its timestamps.
            local_latest = _validate_projection(
                read_advisory_snapshot(_local_path(self.cfg, "progress-observed", attempt_id)),
                binding,
            )
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            # A shared snapshot from another registration generation is not used
            # as recovery evidence by a new projector.
            shared_latest = _validate_projection(
                read_advisory_snapshot(shared_path),
                binding,
                require_generation=True,
            )
        except (OSError, ValueError, KeyError, TypeError):
            pass
        values = [value for value in (local_latest, shared_latest) if value is not None]
        latest = max(values, key=lambda item: item["sequence"]) if values else None
        if interval is None:
            return {"latest": latest, "published": shared_latest, "last_write": float("-inf")}
        restored = latest is not None
        now = self._clock()
        deferred = _initial_deadline(attempt_id, now, interval) if restored else float("-inf")
        return {
            "latest": latest,
            "published": shared_latest,
            "candidate": None,
            "mailbox_signature": None,
            "cache_next_due": deferred,
            "shared_next_due": deferred,
            "cache_initial_available": not restored,
            "shared_initial_available": not restored,
            "cache_final_available": True,
            "shared_final_available": True,
            "interval": interval,
            "last_write": float("-inf"),
        }

    def _retire(self, attempt_id: str) -> None:
        self._entries.pop(attempt_id, None)
        try:
            _local_path(self.cfg, "progress-contexts", attempt_id).unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(local_progress_path(self.cfg.runtime_root, attempt_id).parent, ignore_errors=True)
        for directory in ("progress-observed", "progress-diagnostics"):
            try:
                _local_path(self.cfg, directory, attempt_id).unlink(missing_ok=True)
            except OSError:
                pass

    def _projection_for_payload(
        self,
        payload: dict[str, Any],
        latest: dict[str, Any] | None,
        binding: dict[str, Any],
    ) -> dict[str, Any]:
        now = self._wall_clock()
        advanced = latest is None or semantic_key(payload) != semantic_key(latest["progress"])
        return {
            "protocol_version": 1,
            **{key: binding[key] for key in _IDENTITY},
            "registration_generation": binding["registration_generation"],
            "fencing_token": binding["fencing_token"],
            "source_update_id": payload["update_id"],
            "sequence": 1 if latest is None else latest["sequence"] + 1,
            "reported_at": now,
            "advanced_at": now if advanced else latest["advanced_at"],
            "progress": {key: item for key, item in payload.items() if key not in {"protocol_version", "update_id"}},
        }

    def _observe_policy(
        self,
        attempt_id: str,
        context_path: Path,
        context: dict[str, Any],
        binding: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        """Observe and publish a policy-context Attempt with independent cadences."""
        interval = state["interval"]
        mailbox = local_progress_path(self.cfg.runtime_root, attempt_id)
        signature = _mailbox_signature(mailbox)
        signature_changed = signature is None or signature != state.get("mailbox_signature")
        if signature_changed:
            try:
                payload = validate_payload(read_advisory_snapshot(mailbox, max_bytes=MAX_PAYLOAD_BYTES))
            except FileNotFoundError:
                payload = None
            except (OSError, ValueError, TypeError, RecursionError):
                state["mailbox_signature"] = None
                self._diagnostic(attempt_id, "invalid_payload", interval=interval)
                payload = None
            else:
                state["mailbox_signature"] = signature
            if payload is not None:
                latest = state["latest"]
                if latest is None:
                    state["candidate"] = payload
                elif payload["update_id"] == latest["source_update_id"] or _payload_observation_key(
                    payload
                ) == _payload_observation_key(latest):
                    state["candidate"] = None
                else:
                    state["candidate"] = payload

        candidate = state.get("candidate")
        latest = state["latest"]
        if candidate is not None and latest is not None:
            if candidate["update_id"] == latest["source_update_id"] or _payload_observation_key(
                candidate
            ) == _payload_observation_key(latest):
                state["candidate"] = None
                candidate = None

        now = self._clock()
        if candidate is not None:
            if binding["terminal"]:
                due = state["cache_final_available"]
            else:
                due = state["cache_initial_available"] or now >= state["cache_next_due"]
            if due:
                accepted = self._projection_for_payload(candidate, latest, binding)
                try:
                    self._write_local("progress-observed", attempt_id, accepted)
                except OSError:
                    self._diagnostic(attempt_id, "observation_cache_unavailable", interval=interval)
                    return
                state["latest"] = accepted
                state["candidate"] = None
                state["cache_initial_available"] = False
                if binding["terminal"]:
                    state["cache_final_available"] = False
                succeeded_at = self._clock()
                if math.isfinite(state["cache_next_due"]):
                    state["cache_next_due"] = _next_deadline(state["cache_next_due"], succeeded_at, interval)
                else:
                    state["cache_next_due"] = _initial_deadline(attempt_id, succeeded_at, interval)
                latest = accepted

        if latest is None:
            if binding["terminal"]:
                self._retire(attempt_id)
            return

        # Generation/fencing transitions alter provenance but never fabricate
        # reported_at or advanced_at.  A deferred republish preserves evidence.
        latest = {
            **latest,
            "fencing_token": binding["fencing_token"],
            "registration_generation": self._registration_generation,
        }
        state["latest"] = latest
        published = state["published"]
        if _signature(published) == _signature(latest):
            if binding["terminal"]:
                self._retire(attempt_id)
            return
        now = self._clock()
        if binding["terminal"]:
            due = state["shared_final_available"]
        else:
            due = state["shared_initial_available"] or now >= state["shared_next_due"]
        if not due:
            return

        # This advisory lock is deliberately outside qexp's Task/Attempt
        # authority lock order. Cleanup uses the same lock after marking the
        # Task as cleaning, so stale projection writes cannot recreate it.
        with exclusive(_progress_lock_path(self.cfg.shared_root, binding["task_id"]), blocking=False) as acquired:
            if not acquired:
                return
            current_binding = self._binding(context)
            if current_binding is None or not context_path.exists():
                self._retire(attempt_id)
                return
            if (
                current_binding["fencing_token"] != binding["fencing_token"]
                or current_binding["registration_generation"] != self._registration_generation
            ):
                return
            shared_path = shared_progress_path(self.cfg.shared_root, binding["task_id"], attempt_id)
            try:
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                replace_advisory_snapshot(shared_path, latest)
            except OSError:
                self._diagnostic(attempt_id, "projection_unavailable", interval=interval)
                return
        state["published"] = latest
        state["shared_initial_available"] = False
        if binding["terminal"]:
            state["shared_final_available"] = False
        succeeded_at = self._clock()
        if math.isfinite(state["shared_next_due"]):
            state["shared_next_due"] = _next_deadline(state["shared_next_due"], succeeded_at, interval)
        else:
            state["shared_next_due"] = _initial_deadline(attempt_id, succeeded_at, interval)
        if binding["terminal"]:
            self._retire(attempt_id)

    def observe(self, attempt_id: str) -> None:
        """Observe one context. Public only to keep single-attempt tests deterministic."""
        context_path = _local_path(self.cfg, "progress-contexts", attempt_id)
        context = read_advisory_snapshot(context_path)
        if (
            type(context.get("protocol_version")) is not int
            or context["protocol_version"] != 1
            or context.get("attempt_id") != attempt_id
        ):
            raise ValueError("invalid progress context")
        interval = _context_interval(context)
        binding = self._binding(context)
        if binding is None:
            self._retire(attempt_id)
            return
        state = self._entries.get(attempt_id)
        if state is None:
            state = self._restore(attempt_id, binding, interval=interval)
            self._entries[attempt_id] = state
            if len(self._entries) > 4096:
                self._entries.popitem(last=False)
        self._entries.move_to_end(attempt_id)
        if interval is not None:
            self._observe_policy(attempt_id, context_path, context, binding, state)
            return
        latest = state["latest"]
        try:
            payload = validate_payload(
                read_advisory_snapshot(
                    local_progress_path(self.cfg.runtime_root, attempt_id),
                    max_bytes=MAX_PAYLOAD_BYTES,
                )
            )
        except FileNotFoundError:
            payload = None
        except (OSError, ValueError, TypeError, RecursionError):
            self._diagnostic(attempt_id, "invalid_payload")
            payload = None
        if payload is not None and (latest is None or payload["update_id"] != latest["source_update_id"]):
            now = self._wall_clock()
            advanced = latest is None or semantic_key(payload) != semantic_key(latest["progress"])
            latest = {
                "protocol_version": 1,
                **{key: binding[key] for key in _IDENTITY},
                "registration_generation": binding["registration_generation"],
                "fencing_token": binding["fencing_token"],
                "source_update_id": payload["update_id"],
                "sequence": 1 if latest is None else latest["sequence"] + 1,
                "reported_at": now,
                "advanced_at": now if advanced else latest["advanced_at"],
                "progress": {
                    key: item for key, item in payload.items() if key not in {"protocol_version", "update_id"}
                },
            }
            state["latest"] = latest
            if context_path.exists():
                try:
                    self._write_local("progress-observed", attempt_id, latest)
                except OSError:
                    pass
        if latest is None:
            if binding["terminal"]:
                self._retire(attempt_id)
            return
        latest = {
            **latest,
            "fencing_token": binding["fencing_token"],
            "registration_generation": self._registration_generation,
        }
        state["latest"] = latest
        published = state["published"]
        if _signature(published) == _signature(latest):
            if binding["terminal"]:
                self._retire(attempt_id)
            return
        now_monotonic = self._clock()
        urgent = (
            published is None or binding["terminal"] or latest["progress"]["stage"] != published["progress"]["stage"]
        )
        interval = 1.0 if urgent else 5.0
        if now_monotonic - state["last_write"] < interval:
            return

        # This advisory lock is deliberately outside qexp's Task/Attempt authority
        # lock order. Cleanup uses the same lock after marking the Task as cleaning,
        # which fences stale projection writes without delaying lease renewal.
        with exclusive(_progress_lock_path(self.cfg.shared_root, binding["task_id"]), blocking=False) as acquired:
            if not acquired:
                return
            current_binding = self._binding(context)
            if current_binding is None or not context_path.exists():
                self._retire(attempt_id)
                return
            if (
                current_binding["fencing_token"] != binding["fencing_token"]
                or current_binding["registration_generation"] != self._registration_generation
            ):
                return
            shared_path = shared_progress_path(self.cfg.shared_root, binding["task_id"], attempt_id)
            try:
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                replace_advisory_snapshot(shared_path, latest)
            except OSError:
                self._diagnostic(attempt_id, "projection_unavailable")
                return
        state["last_write"] = now_monotonic
        state["published"] = latest
        if current_binding["terminal"]:
            self._retire(attempt_id)


def _running_registration_generation(cfg: Any, machine_name: str) -> str:
    from .paths import machine_registration_path
    from .store import read_json

    registration = read_json(machine_registration_path(cfg.shared_root, machine_name)).get("registration", {})
    generation = identifier(registration.get("generation"))
    if registration.get("state") == "superseded":
        raise ValueError("progress machine registration is superseded")
    return generation


def inspect_progress(cfg: Any, task: Any) -> ProgressObservation:
    """Return evidence-based progress state without writing any observation state."""
    from .paths import attempt_path
    from .records import AttemptRecord
    from .store import read_json

    def result(state: str, reason: str | None, value: dict[str, Any] | None = None) -> ProgressObservation:
        output: dict[str, Any] = {
            "status": "available" if state == "available" else "unavailable",
            "observation_state": state,
            "reason": reason,
        }
        if value is not None:
            output.update(value)
        return output

    projection = task.state.get("projection")
    current_attempt_id = task.attempt_control.get("current_attempt_id")
    if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
        return result("unavailable", "cleanup")
    if projection == "queued" and current_attempt_id is None:
        return result("pending", "not_started")
    number = task.attempt_control.get("current_attempt_number")
    if type(number) is not int:
        return result("unavailable", "identity_mismatch")

    try:
        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, number)))
    except FileNotFoundError:
        return result("unavailable", "read_failed")
    except (OSError, ValueError, TypeError, KeyError):
        return result("unavailable", "invalid_snapshot")

    task_terminal = projection in _TERMINAL
    if task_terminal:
        if attempt.phase not in _TERMINAL or current_attempt_id not in {None, attempt.attempt_id}:
            return result("unavailable", "identity_mismatch")
        attempt_id = attempt.attempt_id
    else:
        if projection != "running" or current_attempt_id is None or attempt.attempt_id != current_attempt_id:
            return result("unavailable", "identity_mismatch")
        attempt_id = current_attempt_id
        claim = task.claim_control.get("active_claim") or {}
        if (
            attempt.phase not in {"starting", "running"}
            or claim.get("attempt_id") != attempt_id
            or claim.get("fencing_token") != attempt.current_fencing_token
            or claim.get("machine_name") != attempt.machine_name
        ):
            return result("unavailable", "identity_mismatch")

    identity = {
        "task_id": task.task_id,
        "attempt_id": attempt_id,
        "attempt_number": number,
        "machine_name": attempt.machine_name,
        "launch_id": attempt.authorization.get("launch_id"),
        "wrapper_pid": attempt.process.get("wrapper_pid"),
        "wrapper_start_time_ticks": attempt.process.get("wrapper_start_time_ticks"),
        "fencing_token": attempt.current_fencing_token,
    }
    if task_terminal:
        require_generation = False
    else:
        try:
            generation = _running_registration_generation(cfg, attempt.machine_name)
        except (FileNotFoundError, OSError):
            return result("unavailable", "read_failed")
        except (ValueError, TypeError, KeyError):
            return result("unavailable", "identity_mismatch")
        identity["registration_generation"] = generation
        require_generation = True
    try:
        value = _validate_projection(
            read_advisory_snapshot(shared_progress_path(cfg.shared_root, task.task_id, attempt_id)),
            identity,
            require_token=True,
            require_generation=require_generation,
        )
    except FileNotFoundError:
        return result("no_report", "no_snapshot")
    except OSError:
        return result("unavailable", "read_failed")
    except ValueError as exc:
        message = str(exc)
        if any(token in message for token in ("identity", "fencing", "superseded", "registration")):
            return result("unavailable", "identity_mismatch")
        return result("unavailable", "invalid_snapshot")
    except Exception:
        return result("unavailable", "unknown")
    if not task_terminal:
        try:
            if _running_registration_generation(cfg, attempt.machine_name) != identity["registration_generation"]:
                return result("unavailable", "identity_mismatch")
        except (FileNotFoundError, OSError):
            return result("unavailable", "read_failed")
        except (ValueError, TypeError, KeyError):
            return result("unavailable", "identity_mismatch")
    return result("available", None, value)


def _unlink_record(path: Path, removed: list[str]) -> None:
    try:
        path.unlink()
        removed.append(str(path))
    except OSError:
        # Advisory cleanup must never block authoritative Task cleanup.
        pass


def cleanup_local_progress(cfg: Any, task_id: str, attempt_ids: set[str]) -> list[str]:
    """Best-effort removal of local advisory state without TOCTOU failures."""
    removed: list[str] = []
    contexts = Path(cfg.runtime_root) / "progress-contexts"
    try:
        context_paths = list(contexts.glob("*.json")) if contexts.is_dir() else []
    except OSError:
        context_paths = []
    for path in context_paths:
        try:
            if read_advisory_snapshot(path).get("task_id") == task_id:
                attempt_ids.add(identifier(path.stem))
        except (OSError, ValueError, TypeError):
            continue
    for attempt_id in attempt_ids:
        _unlink_record(_local_path(cfg, "progress-contexts", attempt_id), removed)
        mailbox_dir = local_progress_path(cfg.runtime_root, attempt_id).parent
        try:
            existed = mailbox_dir.exists()
            shutil.rmtree(mailbox_dir, ignore_errors=True)
            if existed and not mailbox_dir.exists():
                removed.append(str(mailbox_dir))
        except OSError:
            pass
        for directory in ("progress-observed", "progress-diagnostics"):
            path = _local_path(cfg, directory, attempt_id)
            _unlink_record(path, removed)
            try:
                temporaries = list(path.parent.glob(f".{path.name}.*"))
            except OSError:
                temporaries = []
            for temporary in temporaries:
                _unlink_record(temporary, removed)
    return removed


def cleanup_shared_progress(cfg: Any, task_id: str) -> list[str]:
    """Fence shared projection removal, but never fail authoritative cleanup."""
    try:
        directory = Path(cfg.shared_root) / "progress" / identifier(task_id)
        with exclusive(_progress_lock_path(cfg.shared_root, task_id)):
            existed = directory.exists()
            shutil.rmtree(directory, ignore_errors=True)
            return [str(directory)] if existed and not directory.exists() else []
    except (OSError, RuntimeError, ValueError):
        return []

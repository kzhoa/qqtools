"""Attempt-bound metric observations, isolated from execution authority.

The v2 channel is opt-in per frozen Task selection. It has separate local
context, observation cache, and shared projection paths while sharing the v1
Attempt identity resolver, advisory lock, and write cadence.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any

from qqtools.qexp._progress_protocol import (
    MAX_SNAPSHOT_BYTES,
    identifier,
    read_advisory_snapshot,
    replace_advisory_snapshot,
)
from qqtools.qexp._progress_protocol_v2 import MAX_PAYLOAD_V2_BYTES, semantic_key_v2, validate_payload_v2

from .locks import exclusive
from .progress import (
    _IDENTITY,
    _TERMINAL,
    _initial_deadline,
    _mailbox_signature,
    _next_deadline,
    _progress_lock_path,
    _running_registration_generation,
    _valid_interval,
    resolve_progress_binding,
)

_CONTEXT_FIELDS = frozenset(("protocol_version", "interval_seconds", *_IDENTITY))
_ENVELOPE_FIELDS = frozenset(
    (
        "protocol_version",
        *_IDENTITY,
        "registration_generation",
        "fencing_token",
        "source_update_id",
        "sequence",
        "reported_at",
        "advanced_at",
        "progress",
    )
)
_PROGRESS_FIELDS = frozenset(("stage", "current", "total", "unit", "message", "metrics", "completeness"))
_LOCAL_MAILBOX_NAME = "latest-v2.json"
_ADVANCED_FIELDS = ("stage", "current", "total", "unit")
_POLICY_SCAN_BUDGET = 64
_MAX_SCAN_BUDGET = 256
_MAX_TRACKED_ATTEMPTS = 4096


def _local_path(cfg: Any, directory: str, attempt_id: str) -> Path:
    return Path(cfg.runtime_root) / directory / f"{identifier(attempt_id)}.json"


def _local_mailbox_path(runtime_root: Path, attempt_id: str) -> Path:
    return Path(runtime_root) / "progress" / identifier(attempt_id) / _LOCAL_MAILBOX_NAME


def _context_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, "progress-v2-contexts", attempt_id)


def _observed_path(cfg: Any, attempt_id: str) -> Path:
    return _local_path(cfg, "progress-v2-observed", attempt_id)


def _shared_path(cfg: Any, task_id: str, attempt_id: str) -> Path:
    return Path(cfg.shared_root) / "progress-v2" / identifier(task_id) / f"{identifier(attempt_id)}.json"


def _validate_context(value: Any, attempt_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONTEXT_FIELDS:
        raise ValueError("invalid progress v2 context fields")
    if type(value.get("protocol_version")) is not int or value["protocol_version"] != 2:
        raise ValueError("unsupported progress v2 context")
    identifier(value.get("task_id"))
    identifier(value.get("attempt_id"))
    if value["attempt_id"] != attempt_id:
        raise ValueError("progress v2 context identity mismatch")
    if type(value.get("attempt_number")) is not int or value["attempt_number"] < 1:
        raise ValueError("invalid progress v2 Attempt number")
    if not isinstance(value.get("machine_name"), str) or not value["machine_name"]:
        raise ValueError("invalid progress v2 machine identity")
    if value.get("launch_id") is not None and not isinstance(value["launch_id"], str):
        raise ValueError("invalid progress v2 launch identity")
    if type(value.get("wrapper_pid")) is not int or value["wrapper_pid"] < 1:
        raise ValueError("invalid progress v2 wrapper process identity")
    if type(value.get("wrapper_start_time_ticks")) is not int or value["wrapper_start_time_ticks"] < 0:
        raise ValueError("invalid progress v2 wrapper start time")
    _valid_interval(value.get("interval_seconds"))
    return dict(value)


def _context_interval(context: dict[str, Any]) -> float:
    value = _valid_interval(context["interval_seconds"])
    try:
        return float(value)
    except OverflowError:
        return float("inf")


def prepare_progress_v2_channel(
    cfg: Any,
    task: Any,
    attempt: Any,
    *,
    wrapper_start_time_ticks: int | None,
    interval_seconds: int | float,
) -> str | None:
    """Provision a v2 mailbox and its frozen identity using local I/O only.

    The runner calls this only after resolving the Task's frozen v2 selection
    and after releasing authority locks. Existing context is never rewritten.
    """
    try:
        if type(wrapper_start_time_ticks) is not int or wrapper_start_time_ticks < 0:
            return None
        if attempt.task_id != task.task_id:
            return None
        frozen_interval = _valid_interval(interval_seconds)
        context = {
            "protocol_version": 2,
            "interval_seconds": frozen_interval,
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "machine_name": cfg.machine_name,
            "launch_id": attempt.authorization["launch_id"],
            "wrapper_pid": os.getpid(),
            "wrapper_start_time_ticks": wrapper_start_time_ticks,
        }
        attempt_id = identifier(attempt.attempt_id)
        _validate_context(context, attempt_id)
        runtime_root = Path(cfg.runtime_root)
        mailbox = _local_mailbox_path(runtime_root, attempt_id)
        mailbox.parent.mkdir(parents=True, exist_ok=True)
        context_root = runtime_root / "progress-v2-contexts"
        observed_root = runtime_root / "progress-v2-observed"
        context_root.mkdir(parents=True, exist_ok=True)
        observed_root.mkdir(parents=True, exist_ok=True)
        path = context_root / f"{attempt_id}.json"
        if path.exists():
            existing = _validate_context(read_advisory_snapshot(path), attempt_id)
            if any(existing.get(key) != context.get(key) for key in _IDENTITY):
                return None
            if existing["interval_seconds"] != frozen_interval:
                return None
            return str(mailbox)
        replace_advisory_snapshot(path, context)
        return str(mailbox)
    except Exception:
        return None


def has_local_progress_v2_mailbox(runtime_root: Path) -> bool:
    """Look for a regular v2 producer mailbox without touching shared state."""
    root = Path(runtime_root) / "progress"
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    mailbox = Path(entry.path) / _LOCAL_MAILBOX_NAME
                    if stat.S_ISREG(mailbox.lstat().st_mode):
                        return True
                except OSError:
                    continue
    except OSError:
        return False
    return False


def _validate_projection_v2(
    value: Any,
    identity: dict[str, Any],
    *,
    require_token: bool = False,
    require_generation: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("protocol_version")) is not int or value["protocol_version"] != 2:
        raise ValueError("unsupported progress v2 snapshot")
    if set(value) != _ENVELOPE_FIELDS or not isinstance(value.get("progress"), dict):
        raise ValueError("invalid progress v2 snapshot fields")
    if any(value.get(key) != identity.get(key) for key in _IDENTITY):
        raise ValueError("progress v2 snapshot identity mismatch")
    identifier(value.get("registration_generation"))
    if require_generation and value["registration_generation"] != identity.get("registration_generation"):
        raise ValueError("superseded progress v2 registration generation")
    if type(value.get("fencing_token")) is not int:
        raise ValueError("invalid progress v2 fencing token")
    if require_token and value["fencing_token"] != identity.get("fencing_token"):
        raise ValueError("superseded progress v2 snapshot")
    identifier(value.get("source_update_id"))
    if type(value.get("sequence")) is not int or value["sequence"] < 1:
        raise ValueError("invalid progress v2 sequence")
    for key in ("reported_at", "advanced_at"):
        if not isinstance(value.get(key), str):
            raise ValueError("invalid progress v2 timestamp")
        try:
            stamp = datetime.fromisoformat(value[key].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("invalid progress v2 timestamp") from exc
        if stamp.tzinfo is None:
            raise ValueError("progress v2 timestamp must have a timezone")
    if set(value["progress"]) != _PROGRESS_FIELDS:
        raise ValueError("invalid nested progress v2 fields")
    normalized = validate_payload_v2(
        {"protocol_version": 2, "update_id": value["source_update_id"], **value["progress"]}
    )
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


def _same_content(payload: dict[str, Any], projection: dict[str, Any]) -> bool:
    return semantic_key_v2(payload) == semantic_key_v2(projection.get("progress", projection))


class ProgressV2Projector:
    """Bounded latest-only v2 ingestion for one machine registration generation."""

    def __init__(
        self,
        cfg: Any,
        *,
        registration_generation: str,
        resolver=resolve_progress_binding,
        clock=time.monotonic,
        wall_clock=None,
    ) -> None:
        self.cfg = cfg
        self._registration_generation = identifier(registration_generation)
        self._resolve = resolver
        self._clock = clock
        if wall_clock is None:
            from .progress import _now

            wall_clock = _now
        self._wall_clock = wall_clock
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

    def tick(self, *, budget: int = _POLICY_SCAN_BUDGET) -> None:
        """Observe no more than a bounded number of v2 contexts per call."""
        try:
            if self._scan is None:
                self._scan = os.scandir(Path(self.cfg.runtime_root) / "progress-v2-contexts")
            for _ in range(max(1, min(budget, _MAX_SCAN_BUDGET))):
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
                    self._observe(attempt_id)
                except Exception:
                    continue
        except Exception:
            self.close()

    def _write_local(self, context_path: Path, directory: str, attempt_id: str, value: dict[str, Any]) -> bool:
        if not context_path.exists():
            return False
        path = _local_path(self.cfg, directory, attempt_id)
        replace_advisory_snapshot(path, value)
        if not context_path.exists():
            path.unlink(missing_ok=True)
            return False
        return True

    def _restore(
        self,
        attempt_id: str,
        binding: dict[str, Any],
        interval: float,
    ) -> dict[str, Any]:
        local_latest = None
        shared_latest = None
        try:
            local_latest = _validate_projection_v2(
                read_advisory_snapshot(_observed_path(self.cfg, attempt_id), max_bytes=MAX_SNAPSHOT_BYTES),
                binding,
            )
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            pass
        try:
            shared_latest = _validate_projection_v2(
                read_advisory_snapshot(
                    _shared_path(self.cfg, binding["task_id"], attempt_id),
                    max_bytes=MAX_SNAPSHOT_BYTES,
                ),
                binding,
                require_generation=True,
            )
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            pass
        values = [value for value in (local_latest, shared_latest) if value is not None]
        latest = max(values, key=lambda item: item["sequence"]) if values else None
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
        }

    def _retire(self, attempt_id: str) -> None:
        self._entries.pop(attempt_id, None)
        try:
            _context_path(self.cfg, attempt_id).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _local_mailbox_path(self.cfg.runtime_root, attempt_id).unlink(missing_ok=True)
        except OSError:
            pass
        try:
            _local_mailbox_path(self.cfg.runtime_root, attempt_id).parent.rmdir()
        except OSError:
            pass
        try:
            _observed_path(self.cfg, attempt_id).unlink(missing_ok=True)
        except OSError:
            pass

    def _projection_for_payload(
        self,
        payload: dict[str, Any],
        latest: dict[str, Any] | None,
        binding: dict[str, Any],
    ) -> dict[str, Any]:
        now = self._wall_clock()
        advanced = latest is None or tuple(payload.get(key) for key in _ADVANCED_FIELDS) != tuple(
            latest["progress"].get(key) for key in _ADVANCED_FIELDS
        )
        return {
            "protocol_version": 2,
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
        interval = state["interval"]
        mailbox = _local_mailbox_path(self.cfg.runtime_root, attempt_id)
        signature = _mailbox_signature(mailbox)
        if signature is None or signature != state.get("mailbox_signature"):
            try:
                payload = validate_payload_v2(read_advisory_snapshot(mailbox, max_bytes=MAX_PAYLOAD_V2_BYTES))
            except FileNotFoundError:
                payload = None
            except (OSError, ValueError, TypeError, RecursionError):
                state["mailbox_signature"] = signature
                payload = None
            else:
                state["mailbox_signature"] = signature
            if payload is not None:
                latest = state["latest"]
                if latest is None:
                    state["candidate"] = payload
                elif payload["update_id"] == latest["source_update_id"] or _same_content(payload, latest):
                    state["candidate"] = None
                else:
                    state["candidate"] = payload

        candidate = state.get("candidate")
        latest = state["latest"]
        if candidate is not None and latest is not None:
            if candidate["update_id"] == latest["source_update_id"] or _same_content(candidate, latest):
                state["candidate"] = None
                candidate = None

        now = self._clock()
        if candidate is not None:
            due = (
                state["cache_final_available"]
                if binding["terminal"]
                else (state["cache_initial_available"] or now >= state["cache_next_due"])
            )
            if due:
                accepted = self._projection_for_payload(candidate, latest, binding)
                try:
                    if not self._write_local(context_path, "progress-v2-observed", attempt_id, accepted):
                        self._retire(attempt_id)
                        return
                except OSError:
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

        latest = {
            **latest,
            "fencing_token": binding["fencing_token"],
            "registration_generation": self._registration_generation,
        }
        state["latest"] = latest
        if _signature(state["published"]) == _signature(latest):
            if binding["terminal"]:
                self._retire(attempt_id)
            return
        now = self._clock()
        due = (
            state["shared_final_available"]
            if binding["terminal"]
            else (state["shared_initial_available"] or now >= state["shared_next_due"])
        )
        if not due:
            return

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
            shared_path = _shared_path(self.cfg, binding["task_id"], attempt_id)
            try:
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                replace_advisory_snapshot(shared_path, latest, max_bytes=MAX_SNAPSHOT_BYTES)
                if not context_path.exists():
                    shared_path.unlink(missing_ok=True)
                    return
            except OSError:
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

    def _observe(self, attempt_id: str) -> None:
        context_path = _context_path(self.cfg, attempt_id)
        context = _validate_context(
            read_advisory_snapshot(context_path, max_bytes=MAX_SNAPSHOT_BYTES),
            attempt_id,
        )
        interval = _context_interval(context)
        binding = self._binding(context)
        if binding is None:
            self._retire(attempt_id)
            return
        state = self._entries.get(attempt_id)
        if state is None:
            state = self._restore(attempt_id, binding, interval)
            self._entries[attempt_id] = state
            if len(self._entries) > _MAX_TRACKED_ATTEMPTS:
                self._entries.popitem(last=False)
        self._entries.move_to_end(attempt_id)
        self._observe_policy(attempt_id, context_path, context, binding, state)


def inspect_progress_v2(cfg: Any, task: Any) -> dict[str, Any]:
    """Read one v2 snapshot for the Task's current Attempt, if authorized."""
    from .paths import attempt_path
    from .records import AttemptRecord
    from .store import read_json
    from .tasks import load_task

    def result(state: str, reason: str | None, value: dict[str, Any] | None = None) -> dict[str, Any]:
        output: dict[str, Any] = {
            "status": "available" if state == "available" else "unavailable",
            "observation_state": state,
            "reason": reason,
        }
        if value is not None:
            output.update(value)
        return output

    try:
        task_id = identifier(task.task_id)
    except (AttributeError, TypeError, ValueError):
        return result("unavailable", "identity_mismatch")
    try:
        current_task = load_task(cfg, task_id)
    except FileNotFoundError:
        return result("unavailable", "cleanup")
    except OSError:
        return result("unavailable", "read_failed")
    except (ValueError, TypeError, KeyError):
        return result("unavailable", "invalid_snapshot")

    projection = current_task.state.get("projection")
    current_attempt_id = current_task.attempt_control.get("current_attempt_id")
    if current_task.control.get("cleanup_operation_id") or current_task.control.get("cleanup_state"):
        return result("unavailable", "cleanup")
    if projection == "queued" and current_attempt_id is None:
        return result("pending", "not_started")
    number = current_task.attempt_control.get("current_attempt_number")
    if type(number) is not int:
        return result("unavailable", "identity_mismatch")

    try:
        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, number)))
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
        claim = current_task.claim_control.get("active_claim") or {}
        if (
            attempt.phase not in {"starting", "running"}
            or claim.get("attempt_id") != attempt_id
            or claim.get("fencing_token") != attempt.current_fencing_token
            or claim.get("machine_name") != attempt.machine_name
        ):
            return result("unavailable", "identity_mismatch")

    identity = {
        "task_id": task_id,
        "attempt_id": attempt_id,
        "attempt_number": number,
        "machine_name": attempt.machine_name,
        "launch_id": attempt.authorization.get("launch_id"),
        "wrapper_pid": attempt.process.get("wrapper_pid"),
        "wrapper_start_time_ticks": attempt.process.get("wrapper_start_time_ticks"),
        "fencing_token": attempt.current_fencing_token,
    }
    if not task_terminal:
        try:
            generation = _running_registration_generation(cfg, attempt.machine_name)
        except (FileNotFoundError, OSError):
            return result("unavailable", "read_failed")
        except (ValueError, TypeError, KeyError):
            return result("unavailable", "identity_mismatch")
        identity["registration_generation"] = generation

    try:
        value = _validate_projection_v2(
            read_advisory_snapshot(
                _shared_path(cfg, task_id, attempt_id),
                max_bytes=MAX_SNAPSHOT_BYTES,
            ),
            identity,
            require_token=True,
            require_generation=not task_terminal,
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

    # Old cleanup may leave an advisory file behind. Re-read Task truth after
    # the bounded snapshot read so deletion or cleanup cannot surface it.
    try:
        final_task = load_task(cfg, task_id)
    except FileNotFoundError:
        return result("unavailable", "cleanup")
    except OSError:
        return result("unavailable", "read_failed")
    except (ValueError, TypeError, KeyError):
        return result("unavailable", "invalid_snapshot")
    if final_task.control.get("cleanup_operation_id") or final_task.control.get("cleanup_state"):
        return result("unavailable", "cleanup")
    if final_task.attempt_control.get("current_attempt_number") != number:
        return result("unavailable", "identity_mismatch")
    final_projection = final_task.state.get("projection")
    final_attempt_id = final_task.attempt_control.get("current_attempt_id")
    if final_projection == "queued" and final_attempt_id is None:
        return result("pending", "not_started")
    if final_projection in _TERMINAL:
        if final_attempt_id not in {None, attempt_id}:
            return result("unavailable", "identity_mismatch")
    elif final_projection == "running":
        if final_attempt_id != attempt_id:
            return result("unavailable", "identity_mismatch")
    else:
        return result("unavailable", "identity_mismatch")
    return result("available", None, value)


def _unlink_record(path: Path, removed: list[str]) -> None:
    try:
        path.unlink()
        removed.append(str(path))
    except OSError:
        pass


def cleanup_local_progress_v2(cfg: Any, task_id: str, attempt_ids: set[str]) -> list[str]:
    """Remove v2 contexts first, then only v2 local mailbox and cache files."""
    task_id = identifier(task_id)
    removed: list[str] = []
    targets: set[str] = set()
    for attempt_id in attempt_ids:
        try:
            targets.add(identifier(attempt_id))
        except ValueError:
            continue
    context_root = Path(cfg.runtime_root) / "progress-v2-contexts"
    try:
        context_paths = list(context_root.glob("*.json")) if context_root.is_dir() else []
    except OSError:
        context_paths = []
    for path in context_paths:
        try:
            context = read_advisory_snapshot(path)
            if context.get("task_id") == task_id:
                targets.add(identifier(path.stem))
        except (OSError, ValueError, TypeError):
            continue

    # Remove all discoverable contexts before mailbox or cache cleanup begins.
    for attempt_id in targets:
        _unlink_record(_context_path(cfg, attempt_id), removed)
        context_path = _context_path(cfg, attempt_id)
        try:
            temporaries = list(context_path.parent.glob(f".{context_path.name}.*"))
        except OSError:
            temporaries = []
        for temporary in temporaries:
            _unlink_record(temporary, removed)

    for attempt_id in targets:
        mailbox = _local_mailbox_path(cfg.runtime_root, attempt_id)
        _unlink_record(mailbox, removed)
        try:
            temporaries = list(mailbox.parent.glob(f".{mailbox.name}.*"))
        except OSError:
            temporaries = []
        for temporary in temporaries:
            _unlink_record(temporary, removed)
        observed_path = _observed_path(cfg, attempt_id)
        _unlink_record(observed_path, removed)
        try:
            temporaries = list(observed_path.parent.glob(f".{observed_path.name}.*"))
        except OSError:
            temporaries = []
        for temporary in temporaries:
            _unlink_record(temporary, removed)
    return removed


def cleanup_shared_progress_v2(cfg: Any, task_id: str) -> list[str]:
    """Remove the v2 Task projection under the shared v1 progress lock."""
    try:
        task_id = identifier(task_id)
        directory = Path(cfg.shared_root) / "progress-v2" / task_id
        with exclusive(_progress_lock_path(cfg.shared_root, task_id)):
            existed = directory.exists()
            shutil.rmtree(directory, ignore_errors=True)
            return [str(directory)] if existed and not directory.exists() else []
    except (OSError, RuntimeError, ValueError):
        return []

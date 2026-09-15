"""Attempt-bound application observations, isolated from execution authority.

Only the passive runner provisions a channel, under its existing launch lock.
The agent projects it on a separate observability thread. No function in this
module renews a lease, changes a Task/Attempt, sends a signal or releases a GPU.
"""

from __future__ import annotations

import os
import shutil
import time
from collections import OrderedDict
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

_LOCAL_DIRS = ("progress", "progress-contexts", "progress-observed", "progress-diagnostics")
_IDENTITY = ("task_id", "attempt_id", "attempt_number", "machine_name", "launch_id",
             "wrapper_pid", "wrapper_start_time_ticks")
_TERMINAL = frozenset(("succeeded", "failed", "cancelled"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_progress_path(runtime_root: Path, attempt_id: str) -> Path:
    return Path(runtime_root) / "progress" / f"{identifier(attempt_id)}.json"


def shared_progress_path(shared_root: Path, task_id: str, attempt_id: str) -> Path:
    return Path(shared_root) / "progress" / identifier(task_id) / f"{identifier(attempt_id)}.json"


def _local_path(cfg: Any, directory: str, attempt_id: str) -> Path:
    return Path(cfg.runtime_root) / directory / f"{identifier(attempt_id)}.json"


def prepare_progress_channel(cfg: Any, task: Any, attempt: Any, *, wrapper_start_time_ticks: int | None) -> str | None:
    """Provision optional advisory files while the runner holds its launch lock.

    The context is runner-owned identity evidence, separate from the payload.
    It survives removal of normal process records long enough to collect a
    short-lived command's final update. It cannot grant execution authority.
    """
    try:
        context = {
            "protocol_version": 1,
            "task_id": task.task_id,
            "attempt_id": attempt.attempt_id,
            "attempt_number": attempt.attempt_number,
            "machine_name": cfg.machine_name,
            "launch_id": attempt.authorization["launch_id"],
            "wrapper_pid": os.getpid(),
            "wrapper_start_time_ticks": wrapper_start_time_ticks,
        }
        if type(wrapper_start_time_ticks) is not int:
            return None
        path = local_progress_path(cfg.runtime_root, attempt.attempt_id)
        shared_progress_path(cfg.shared_root, task.task_id, attempt.attempt_id).parent.mkdir(parents=True, exist_ok=True)
        for directory in _LOCAL_DIRS:
            (Path(cfg.runtime_root) / directory).mkdir(parents=True, exist_ok=True)
        context_path = _local_path(cfg, "progress-contexts", attempt.attempt_id)
        if context_path.exists():
            return None
        replace_advisory_snapshot(context_path, context)
        return str(path)
    except Exception:
        return None


def resolve_progress_binding(cfg: Any, context: dict[str, Any]) -> dict[str, Any] | None:
    """Read current authority; None means superseded/cleaned, never task failure.

    This intentionally takes no authority lock. A read-side identity/token check
    rejects a projection racing retry or recovery instead of delaying leases.
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
    if task.attempt_control.get("current_attempt_id") != attempt_id:
        return None
    number = task.attempt_control.get("current_attempt_number")
    if type(number) is not int or number != context.get("attempt_number"):
        return None
    attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task_id, number)))
    if attempt.attempt_id != attempt_id or attempt.task_id != task_id or attempt.machine_name != cfg.machine_name:
        return None
    if context.get("machine_name") != attempt.machine_name or context.get("launch_id") != attempt.authorization.get("launch_id"):
        return None
    for key in ("wrapper_pid", "wrapper_start_time_ticks"):
        if type(context.get(key)) is not int or attempt.process.get(key) != context[key]:
            # Registration may not have been materialized yet. Keep the channel.
            raise ValueError("progress process identity not yet verified")
    terminal = task.state["projection"] in _TERMINAL and attempt.phase in _TERMINAL
    if not terminal:
        claim = task.claim_control.get("active_claim") or {}
        if (task.state["projection"] != "running" or attempt.phase not in {"starting", "running"}
                or claim.get("attempt_id") != attempt_id
                or claim.get("fencing_token") != attempt.current_fencing_token
                or claim.get("machine_name") != cfg.machine_name):
            # An orphan can recover without launching a different process.
            raise ValueError("progress authority currently unavailable")
    return {**{key: context[key] for key in _IDENTITY},
            "fencing_token": attempt.current_fencing_token, "terminal": terminal}


def _validate_projection(value: Any, identity: dict[str, Any], *, require_token: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict) or type(value.get("protocol_version")) is not int or value["protocol_version"] != 1:
        raise ValueError("unsupported progress snapshot")
    expected = {"protocol_version", *_IDENTITY, "fencing_token", "source_update_id",
                "sequence", "reported_at", "advanced_at", "progress"}
    if set(value) != expected or not isinstance(value.get("progress"), dict):
        raise ValueError("invalid progress snapshot fields")
    if any(value.get(key) != identity.get(key) for key in _IDENTITY):
        raise ValueError("progress snapshot identity mismatch")
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
    normalized = validate_payload({**value["progress"], "protocol_version": 1,
                                   "update_id": value["source_update_id"]})
    if any(key in value["progress"] for key in ("protocol_version", "update_id")):
        raise ValueError("invalid nested progress payload")
    value = dict(value)
    value["progress"] = {key: item for key, item in normalized.items() if key not in {"protocol_version", "update_id"}}
    return value


def _signature(value: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return value["source_update_id"], value["sequence"], value["fencing_token"]


class ProgressProjector:
    """Bounded, latest-only ingestion. Call exclusively on an observation thread."""

    def __init__(self, cfg: Any, *, clock=time.monotonic, wall_clock=_now, resolver=resolve_progress_binding) -> None:
        self.cfg = cfg
        self._clock = clock
        self._wall_clock = wall_clock
        self._resolve = resolver
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._scan = None

    def close(self) -> None:
        if self._scan is not None:
            self._scan.close()
            self._scan = None

    def tick(self, *, budget: int = 64) -> None:
        """Observe at most budget contexts; one malformed producer cannot abort a sweep."""
        try:
            if self._scan is None:
                directory = Path(self.cfg.runtime_root) / "progress-contexts"
                self._scan = os.scandir(directory)
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
                    # Never append raw payloads or unbounded per-update events.
                    self._diagnostic(attempt_id, "invalid_or_unavailable_progress")
        except Exception:
            self.close()

    def _write_local(self, directory: str, attempt_id: str, value: dict[str, Any]) -> None:
        # Cleanup removes the context first. Recheck after replace as well, so
        # a racing cache/diagnostic write cannot resurrect cleaned artifacts.
        context_path = _local_path(self.cfg, "progress-contexts", attempt_id)
        if not context_path.exists():
            return
        path = _local_path(self.cfg, directory, attempt_id)
        replace_advisory_snapshot(path, value)
        if not context_path.exists():
            path.unlink(missing_ok=True)

    def _diagnostic(self, attempt_id: str, reason: str) -> None:
        try:
            context_path = _local_path(self.cfg, "progress-contexts", attempt_id)
            if not context_path.exists():
                return
            path = _local_path(self.cfg, "progress-diagnostics", attempt_id)
            try:
                previous = read_advisory_snapshot(path)
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(previous["at"].replace("Z", "+00:00"))).total_seconds()
                if previous.get("reason") == reason and elapsed < 60:
                    return
            except (OSError, ValueError, KeyError, TypeError):
                pass
            self._write_local("progress-diagnostics", attempt_id, {"attempt_id": attempt_id, "reason": reason, "at": _now()})
        except Exception:
            pass

    def _restore(self, attempt_id: str, binding: dict[str, Any]) -> dict[str, Any]:
        shared_path = shared_progress_path(self.cfg.shared_root, binding["task_id"], attempt_id)
        values: list[dict[str, Any]] = []
        published = None
        for path in (_local_path(self.cfg, "progress-observed", attempt_id), shared_path):
            try:
                value = _validate_projection(read_advisory_snapshot(path), binding)
                values.append(value)
                if path == shared_path:
                    published = value
            except (OSError, ValueError, KeyError, TypeError):
                pass
        latest = max(values, key=lambda item: item["sequence"]) if values else None
        return {"latest": latest, "published": published, "last_write": float("-inf")}

    def _retire(self, attempt_id: str) -> None:
        self._entries.pop(attempt_id, None)
        for directory in ("progress-contexts", "progress", "progress-observed", "progress-diagnostics"):
            try:
                _local_path(self.cfg, directory, attempt_id).unlink(missing_ok=True)
            except OSError:
                pass

    def observe(self, attempt_id: str) -> None:
        """Observe one context. Public only to make single-attempt tests deterministic."""
        context_path = _local_path(self.cfg, "progress-contexts", attempt_id)
        context = read_advisory_snapshot(context_path)
        if type(context.get("protocol_version")) is not int or context["protocol_version"] != 1 or context.get("attempt_id") != attempt_id:
            raise ValueError("invalid progress context")
        binding = self._resolve(self.cfg, context)
        if binding is None:
            self._retire(attempt_id)
            return
        state = self._entries.get(attempt_id)
        if state is None:
            state = self._restore(attempt_id, binding)
            self._entries[attempt_id] = state
            if len(self._entries) > 4096:
                self._entries.popitem(last=False)
        self._entries.move_to_end(attempt_id)
        latest = state["latest"]
        try:
            payload = validate_payload(read_advisory_snapshot(local_progress_path(self.cfg.runtime_root, attempt_id), max_bytes=MAX_PAYLOAD_BYTES))
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
                **{key: binding[key] for key in (*_IDENTITY, "fencing_token")},
                "source_update_id": payload["update_id"],
                "sequence": 1 if latest is None else latest["sequence"] + 1,
                "reported_at": now,
                "advanced_at": now if advanced else latest["advanced_at"],
                "progress": {key: item for key, item in payload.items() if key not in {"protocol_version", "update_id"}},
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
        # Recovery may change authority without a new application report. Never
        # refresh report/advance timestamps merely because the agent restarted.
        latest = {**latest, "fencing_token": binding["fencing_token"]}
        state["latest"] = latest
        published = state["published"]
        if _signature(published) == _signature(latest):
            if binding["terminal"]:
                self._retire(attempt_id)
            return
        now_monotonic = self._clock()
        urgent = published is None or binding["terminal"] or latest["progress"]["stage"] != published["progress"]["stage"]
        # Stage changes bypass the normal five-second interval, but a malicious
        # toggling stage still cannot cause more than one shared write/second.
        interval = 1.0 if urgent else 5.0
        if now_monotonic - state["last_write"] < interval:
            return
        state["last_write"] = now_monotonic
        current_binding = self._resolve(self.cfg, context)
        if current_binding is None or not context_path.exists():
            self._retire(attempt_id)
            return
        if current_binding["fencing_token"] != binding["fencing_token"]:
            return
        try:
            replace_advisory_snapshot(shared_progress_path(self.cfg.shared_root, binding["task_id"], attempt_id), latest)
        except OSError:
            self._diagnostic(attempt_id, "projection_unavailable")
            return
        state["published"] = latest
        if current_binding["terminal"]:
            self._retire(attempt_id)


def inspect_progress(cfg: Any, task: Any) -> dict[str, Any]:
    """Join the explicit current Attempt, never the newest file by mtime."""
    from .paths import attempt_path
    from .records import AttemptRecord
    from .store import read_json

    unavailable = {"status": "unavailable"}
    try:
        if task.control.get("cleanup_operation_id") or task.control.get("cleanup_state"):
            return unavailable
        attempt_id = task.attempt_control.get("current_attempt_id")
        number = task.attempt_control.get("current_attempt_number")
        if attempt_id is None or type(number) is not int:
            return unavailable
        attempt = AttemptRecord.from_dict(read_json(attempt_path(cfg.shared_root, task.task_id, number)))
        if attempt.attempt_id != attempt_id:
            return unavailable
        identity = {"task_id": task.task_id, "attempt_id": attempt_id,
                    "attempt_number": number, "machine_name": attempt.machine_name,
                    "launch_id": attempt.authorization.get("launch_id"),
                    "wrapper_pid": attempt.process.get("wrapper_pid"),
                    "wrapper_start_time_ticks": attempt.process.get("wrapper_start_time_ticks"),
                    "fencing_token": attempt.current_fencing_token}
        value = _validate_projection(read_advisory_snapshot(shared_progress_path(cfg.shared_root, task.task_id, attempt_id)), identity, require_token=True)
        return {"status": "available", **value}
    except Exception:
        return unavailable


def _age(stamp: str) -> str:
    try:
        seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(stamp.replace("Z", "+00:00"))).total_seconds()
        if seconds < 0:
            return "unknown (clock difference)"
        if seconds < 60:
            return f"{int(seconds)}s ago"
        if seconds < 3600:
            return f"{int(seconds // 60)}m ago"
        return f"{int(seconds // 3600)}h ago"
    except (ValueError, TypeError, AttributeError):
        return "unknown"


def progress_details(result: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Human-only presentation; missing observations never break task show."""
    observation = result.get("progress") or {}
    if observation.get("status") != "available":
        return (("Progress", "unavailable"),)
    payload = observation["progress"]
    current, total = payload.get("current"), payload.get("total")
    count = "unknown" if current is None else str(current)
    if total is not None:
        count += f"/{total}"
    if payload.get("unit"):
        count += f" {payload['unit']}"
    if current is not None and total is not None and total > 0:
        count += f" ({100 * current / total:.1f}%)"
    return (("Stage", payload["stage"]), ("Progress", count),
            ("Message", payload.get("message")),
            ("Progress reported", _age(observation["reported_at"])),
            ("Progress advanced", _age(observation["advanced_at"])))


def cleanup_local_progress(cfg: Any, task_id: str, attempt_ids: set[str]) -> list[str]:
    """Called only from the existing task cleanup lifecycle, after process checks."""
    removed = []
    contexts = Path(cfg.runtime_root) / "progress-contexts"
    if contexts.is_dir():
        for path in contexts.glob("*.json"):
            try:
                if read_advisory_snapshot(path).get("task_id") == task_id:
                    attempt_ids.add(identifier(path.stem))
            except (OSError, ValueError, TypeError):
                continue
    for attempt_id in attempt_ids:
        for directory in ("progress-contexts", "progress", "progress-observed", "progress-diagnostics"):
            path = _local_path(cfg, directory, attempt_id)
            if path.exists():
                path.unlink()
                removed.append(str(path))
            for temporary in path.parent.glob(f".{path.name}.*"):
                temporary.unlink(missing_ok=True)
                removed.append(str(temporary))
    return removed


def cleanup_shared_progress(cfg: Any, task_id: str) -> list[str]:
    directory = Path(cfg.shared_root) / "progress" / identifier(task_id)
    if directory.exists():
        shutil.rmtree(directory)
        return [str(directory)]
    return []

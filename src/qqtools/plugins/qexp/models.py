from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

TASK_PENDING = "pending"
TASK_RUNNING = "running"
TASK_DONE = "done"
TASK_FAILED = "failed"
TASK_CANCELLED = "cancelled"

TASK_STATES = (
    TASK_PENDING,
    TASK_RUNNING,
    TASK_DONE,
    TASK_FAILED,
    TASK_CANCELLED,
)

STATE_DIRECTORY_MAP = {
    TASK_PENDING: "pending",
    TASK_RUNNING: "running",
    TASK_DONE: "done",
    TASK_FAILED: "failed",
    TASK_CANCELLED: "cancelled",
}

SUPPORTED_ENV_KINDS = {"conda", "venv", "none"}
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_optional_path(value: str | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve())


def _normalize_env(env: dict[str, Any] | None) -> dict[str, Any] | None:
    if env is None:
        return None

    if not isinstance(env, dict):
        raise ValueError("Task env must be a dictionary when provided.")

    normalized = dict(env)
    kind = normalized.get("kind", "none")
    if kind not in SUPPORTED_ENV_KINDS:
        raise ValueError(
            "Task env.kind must be one of 'conda', 'venv', or 'none'."
        )

    normalized["kind"] = kind
    if kind == "conda":
        if not normalized.get("name"):
            raise ValueError("Conda env requires a non-empty 'name'.")
        if not normalized.get("activate_script"):
            raise ValueError("Conda env requires 'activate_script'.")
        normalized["activate_script"] = _normalize_optional_path(normalized["activate_script"])
    elif kind == "venv":
        if not normalized.get("path"):
            raise ValueError("Venv env requires a non-empty 'path'.")
        normalized["path"] = _normalize_optional_path(normalized["path"])

    extra_env = normalized.get("extra_env")
    if extra_env is not None:
        if not isinstance(extra_env, dict):
            raise ValueError("Task env.extra_env must be a dictionary when provided.")
        normalized["extra_env"] = {str(key): str(value) for key, value in extra_env.items()}

    return normalized


def ensure_valid_task_state(state: str) -> str:
    if state not in TASK_STATES:
        raise ValueError(
            "Task status must be one of pending, running, done, failed, or cancelled."
        )
    return state


def get_state_directory_name(state: str) -> str:
    ensure_valid_task_state(state)
    return STATE_DIRECTORY_MAP[state]


def ensure_valid_task_id(task_id: str) -> str:
    if not task_id or not isinstance(task_id, str):
        raise ValueError("Task task_id must be a non-empty string.")
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise ValueError(
            "Task task_id contains illegal path characters. Only letters, digits, '.', '_', and '-' are allowed."
        )
    if not TASK_ID_PATTERN.fullmatch(task_id):
        raise ValueError(
            "Task task_id contains illegal characters. Only letters, digits, '.', '_', and '-' are allowed."
        )
    return task_id


@dataclass(slots=True)
class qExpTask:
    task_id: str
    argv: list[str]
    num_gpus: int
    name: str | None = None
    workdir: str | None = None
    env: dict[str, Any] | None = None
    version: str = "1.0"
    created_at: str | None = None
    status: str = TASK_PENDING
    scheduled_at: str | None = None
    assigned_gpus: list[int] | None = None
    tmux_session: str | None = None
    tmux_window_id: str | None = None
    wrapper_pid: int | None = None
    process_group_id: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    exit_reason: str | None = None

    def __post_init__(self) -> None:
        self.task_id = ensure_valid_task_id(self.task_id)

        if not isinstance(self.argv, list) or not self.argv:
            raise ValueError("Task argv must be a non-empty argv list.")
        if any(not isinstance(arg, str) or not arg for arg in self.argv):
            raise ValueError("Task argv entries must be non-empty strings.")

        if not isinstance(self.num_gpus, int) or self.num_gpus <= 0:
            raise ValueError("Task num_gpus must be a positive integer.")

        self.status = ensure_valid_task_state(self.status)
        self.created_at = self.created_at or utc_now_iso()
        self.workdir = _normalize_optional_path(self.workdir)
        self.env = _normalize_env(self.env)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "qExpTask":
        if not isinstance(data, dict):
            raise ValueError("Task payload must be a dictionary.")
        return cls(**data)

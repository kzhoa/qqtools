"""Read-only failure evidence for test-owned Project I/O worker processes."""

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qqtools.plugins.qexp.runtime.store import read_json_limited


def trace_project_io_requests() -> None:
    """Print a bounded actual request chronology inside a test-owned agent."""
    from qqtools.plugins.qexp.agent.project_io_admission import ProjectIOAdmission
    from qqtools.plugins.qexp.agent.project_io_executor import ProjectIOExecutor

    original_start = ProjectIOExecutor.start
    original_consume = ProjectIOExecutor.consume
    original_offer = ProjectIOAdmission.offer
    started_at = time.monotonic()
    remaining = 128
    seen_offers = set()

    def record(event, request, result=None):
        nonlocal remaining
        if remaining <= 0:
            return
        remaining -= 1
        evidence = {
            "elapsed": round(time.monotonic() - started_at, 3),
            "event": event,
            "project_id": getattr(request, "project_id", None) or request.owner[1],
            "kind": request.operation_kind,
        }
        if result is not None:
            evidence.update(status=result.status, reason_code=result.reason_code)
            if isinstance(result.evidence, Mapping):
                evidence.update(
                    evidence_state=result.evidence.get("state"),
                    evidence_outcome=result.evidence.get("outcome"),
                )
        print("project_io_trace " + json.dumps(evidence, sort_keys=True), flush=True)

    def offer(self, intent, action):
        key = (intent.owner, intent.operation_kind, intent.work_family)
        if len(seen_offers) < 64 and key not in seen_offers:
            seen_offers.add(key)
            record("first_offer", intent)
        return original_offer(self, intent, action)

    def start(self, request_id, **options):
        process = original_start(self, request_id, **options)
        if process is not None:
            record("start", process.request)
        return process

    def consume(self, request_id, current_request):
        result = original_consume(self, request_id, current_request)
        if result is not None:
            record("consume", result.request, result)
        return result

    ProjectIOExecutor.start = start
    ProjectIOExecutor.consume = consume
    ProjectIOAdmission.offer = offer


def describe_project_io_workers(executor: Any) -> dict[str, Any]:
    """Inspect only the executor's bounded owned children, without sending signals."""
    children = {}
    for request_id, child in tuple(executor._children.items())[:4]:
        process = {"pid": child.pid, "returncode": child.poll()}
        for name in ("wchan", "syscall", "io"):
            try:
                process[name] = (Path("/proc") / str(child.pid) / name).read_text()[:4096]
            except OSError as exc:
                process[name] = type(exc).__name__
        try:
            process["read_fd"] = str((Path("/proc") / str(child.pid) / "fd" / "3").readlink())
        except OSError as exc:
            process["read_fd"] = type(exc).__name__
        children[request_id] = process
    history = []
    try:
        paths = sorted(executor.paths["project_io_resolved"].glob("*.json"))[-32:]
        for path in paths:
            record = read_json_limited(path, max_bytes=65_536)["project_io_resolution"]
            result = record.get("result")
            if isinstance(result, dict):
                history.append(
                    {
                        "sequence": record["sequence"],
                        **{name: result.get(name) for name in ("completed_at", "status", "reason_code")},
                    }
                )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        history.append({"diagnostic_error": type(exc).__name__})
    pending = [
        {
            "project_id": request.project_id,
            "operation_kind": request.operation_kind,
            "prepared_at": request.prepared_at,
        }
        for request in executor.unresolved_requests()
    ]
    return {"status": executor.status_view(), "children": children, "history": history, "pending": pending}

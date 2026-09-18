"""Summarize workload reports, including comparable interior steady-state windows."""

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def summary(values):
    return {
        "count": len(values),
        "median_seconds": statistics.median(values) if values else None,
        "maximum_seconds": max(values) if values else None,
    }


def steady_operations(report):
    """Subtract cumulative snapshots inside the measured steady phase.

    Missing or too-short windows stay unavailable. Counts are normalized by the
    actual sampled duration, not by requested sleep or total test duration.
    """
    start = report.get("steady_started_monotonic")
    finish = report.get("steady_finished_monotonic")
    if start is None or finish is None:
        return None
    samples = [
        sample
        for sample in report.get("agent", {}).get("operation_samples", [])
        if start <= sample["monotonic"] <= finish
    ]
    if len(samples) < 2:
        return None
    first, last = samples[0], samples[-1]
    seconds = last["monotonic"] - first["monotonic"]
    if seconds <= 0:
        raise ValueError("steady counter snapshots must advance monotonically")
    by_thread = {}
    for thread, counters in last["operations"].items():
        previous = first["operations"].get(thread, {})
        by_thread[thread] = {}
        for name, values in counters.items():
            delta = {key: values[key] - previous.get(name, {}).get(key, 0) for key in ("calls", "errors", "total_ns")}
            if any(value < 0 for value in delta.values()):
                raise ValueError("cumulative operation counters decreased")
            by_thread[thread][name] = dict(delta, calls_per_second=delta["calls"] / seconds)
    return {
        "started_monotonic": first["monotonic"],
        "finished_monotonic": last["monotonic"],
        "seconds": seconds,
        "sample_count": len(samples),
        "samples_dropped": report.get("agent", {}).get("operation_samples_dropped", 0),
        "phase_seconds": finish - start,
        "coverage_ratio": seconds / (finish - start),
        "by_thread": by_thread,
    }


def startup_summary(report):
    """Summarize completed launch phases and agent writes before all-running."""
    profiles = report.get("startup_profiles", {})
    phases = defaultdict(list)
    incomplete = []
    invalid = {}
    missing_shell = []
    for name, raw in profiles.items():
        if not name.endswith(".requested"):
            continue
        identity = name.removesuffix(".requested")
        encoded = profiles.get(identity + ".json")
        if encoded is None:
            incomplete.append(identity)
            continue
        try:
            events = json.loads(encoded)
            times = {event["stage"]: float(event["at"]) for event in events}
            times["command_requested"] = float(raw)
            shell = profiles.get(identity + ".shell", "").strip()
            if shell:
                times["shell_entered"] = float(shell)
            else:
                missing_shell.append(identity)
        except (ValueError, KeyError, TypeError) as error:
            invalid[identity] = f"{type(error).__name__}: {error}"
            continue
        for before, after in (
            ("command_requested", "shell_entered"),
            ("shell_entered", "python_entered"),
            ("import_started", "import_finished"),
            ("authority_lock_requested", "authority_lock_acquired"),
            ("_publish_launch_intent:started", "_publish_launch_intent:finished"),
            ("command_requested", "_publish_launch_intent:finished"),
        ):
            if before in times and after in times:
                phases[f"{before}_to_{after}"].append(times[after] - times[before])
    for call in report.get("agent", {}).get("launch_calls", []):
        phases[call["stage"]].append(call["finished"] - call["started"])
    writes = defaultdict(lambda: {"calls": 0, "total_ns": 0, "fsync_calls": 0, "fsync_ns": 0})
    duration = report.get("all_running_seconds")
    if duration is not None:
        boundary = report["agent_requested_monotonic"] + duration
        for write in report.get("agent", {}).get("write_observations", []):
            if write["finished_monotonic"] > boundary:
                continue
            counter = writes[Path(write["path"]).name]
            counter["calls"] += 1
            for key in ("total_ns", "fsync_calls", "fsync_ns"):
                counter[key] += write[key]
    return {
        "phases": {key: summary(values) for key, values in phases.items()},
        "environment_is_valid": report.get("startup_environment_is_valid"),
        "profile_status": report.get("startup_profile_status"),
        "profile_errors": report.get("startup_profile_errors", []),
        "incomplete_launch_profiles": incomplete,
        "invalid_launch_profiles": invalid,
        "missing_shell_timestamps": missing_shell,
        "agent_import_seconds": report.get("agent", {}).get("startup_import_seconds"),
        "agent_writes_by_filename": dict(writes),
        "write_observations_dropped": report.get("agent", {}).get("write_observations_dropped", 0),
    }


def control_plane_coverage(report):
    """Check the last diagnostic sample against observed registration generations."""
    snapshot = report.get("control_plane_snapshot", {}).get("authority_control_plane")
    expected = {}
    for event in report.get("agent", {}).get("events", []):
        if event["record"] == "registration" and event.get("project_id") and event.get("generation"):
            expected[event["project_id"]] = event["generation"]
    observed = {
        project["project_id"]: project.get("registration_generation")
        for project in (snapshot or {}).get("projects", [])
    }
    count = report.get("profile", {}).get("bindings")
    return {
        "is_available": snapshot is not None,
        "has_complete_generation_coverage": (
            snapshot is not None and count is not None and len(expected) == count and observed == expected
        ),
        "expected_generations": expected,
        "observed_generations": observed,
        "expected_project_count": count,
        "scope": "Last persisted diagnostic sample only; ages and per-visit operations are not run maxima/totals.",
    }


def summarize(path):
    report = json.loads(path.read_text())
    stages = defaultdict(dict)
    renewals = defaultdict(list)
    for event in report.get("agent", {}).get("events", []):
        record = event["record"]
        identity = event.get("attempt_id")
        if identity:
            stage = stages[identity]
            if record == "process_registration" and event.get("source_at"):
                stage.setdefault("registration", timestamp(event["source_at"]))
            if record == "process":
                stage.setdefault("manifest", timestamp(event["at"]))
            if record == "attempt" and event.get("phase") == "running":
                stage.setdefault("running", timestamp(event["at"]))
            if record == "exit_observation" and event.get("source_at"):
                stage.setdefault("exit", timestamp(event["source_at"]))
            if record == "attempt" and event.get("phase") in {"succeeded", "failed", "cancelled"}:
                stage.setdefault("terminal", timestamp(event["at"]))
            if record == "reservation" and event.get("phase") == "released":
                stage.setdefault("accounting", timestamp(event["at"]))
        if record in {"attempt", "registration"} and event.get("expires_at"):
            renewals[
                (record, identity or event.get("project_id"), event.get("generation"), event.get("fencing_token"))
            ].append(event)
    latencies = {}
    for before, after in (
        ("registration", "manifest"),
        ("registration", "running"),
        ("exit", "terminal"),
        ("terminal", "accounting"),
    ):
        values = [s[after] - s[before] for s in stages.values() if before in s and after in s]
        latencies[f"{before}_to_{after}"] = summary(values)
    lateness = defaultdict(list)
    for (record, _identity, _generation, _token), events in renewals.items():
        previous = None
        seen = set()
        for event in events:
            expiry = event["expires_at"]
            if expiry in seen:
                continue
            seen.add(expiry)
            if previous is not None:
                # Workload uses unchanged defaults: TTL 120s, renewal interval 10s.
                target = timestamp(previous) - timedelta(seconds=110).total_seconds()
                lateness[record].append(max(0, timestamp(event["at"]) - target))
            previous = expiry
    operations = defaultdict(int)
    for counters in report.get("agent", {}).get("operations", {}).values():
        for name, counter in counters.items():
            operations[name] += counter["calls"]
    return {
        "sample": path.name,
        "summary_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile": report["profile"],
        "outcome": report["outcome"],
        "source_sha": report["source_sha"],
        "instrumentation_sha256": report["instrumentation_sha256"],
        "workload_test_sha256": report["workload_test_sha256"],
        "latencies": latencies,
        "expiry_publication_lateness": {k: summary(v) for k, v in lateness.items()},
        "operations": dict(operations),
        "operations_by_thread": report.get("agent", {}).get("operations", {}),
        "control_plane_snapshot": report.get("control_plane_snapshot"),
        "control_plane_coverage": control_plane_coverage(report),
        "steady_operations": steady_operations(report),
        "startup": startup_summary(report),
        "all_running_seconds": report.get("all_running_seconds"),
        "finish_to_accounting_seconds": (
            report["accounting_seen_seconds"] - report["finish_requested_seconds"]
            if "accounting_seen_seconds" in report and "finish_requested_seconds" in report
            else None
        ),
        "caveats": [
            "Single samples are not acceptance or speedup evidence.",
            "Source timestamps precede durable publication; writes use observed return time.",
            "Unnormalized totals span unequal whole-run windows; use steady_operations for steady rates.",
            "Source UTC timestamps may be quantized to whole seconds.",
            "Nested operation timings overlap and cannot be summed as disjoint work.",
            "Same-expiry advances are deduplicated; renewal defaults are TTL120s/interval10s.",
            "Startup profiling replaces only the test runner entry; observer overhead is included.",
            "Launch wall timestamps require a stable host clock; shell timing requires bash EPOCHREALTIME.",
            "Command-to-shell includes send-command work; phase durations overlap.",
            "Agent write totals exclude runner and guardian I/O; filenames aggregate across bindings.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps([summarize(Path(path)) for path in sys.argv[1:]], indent=2))

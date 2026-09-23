"""Compare qexp progress producer cost with a synthetic CUDA workload.

This qualification harness uses isolated worker processes and a shared-GPU
matrix-multiplication workload. Its file-polling clients are deliberately only
a transport proxy; they are not qexp viewers or agent projections.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any

_PROFILES = ("baseline_v1", "candidate_v1", "candidate_v2")
_THROUGHPUT_TARGET_PERCENT = 3.0
_CASE_LIMIT_SECONDS = 600.0
_CASE_CLEANUP_RESERVE_SECONDS = 10.0
_LATENCY_SAMPLE_LIMIT = 8192
_SNAPSHOT_LIMIT_BYTES = 16 * 1024
_IN_MOVED_TO = 0x80
_INOTIFY_EVENT = struct.Struct("iIII")


def _positive_int(value: str) -> int:
    try:
        number = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    try:
        number = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def _package_paths(value: str, *, label: str) -> tuple[Path, Path]:
    supplied = Path(value).expanduser().resolve()
    package = supplied if supplied.name == "qqtools" else supplied / "qqtools"
    if not package.is_dir():
        raise ValueError(f"{label} must identify a directory containing the qqtools package: {supplied}")
    return package.parent, package


def _percentile(values: deque[float], percentile: float = 0.99) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _rss_bytes() -> int:
    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum * 1024) if sys.platform.startswith("linux") else int(maximum)


def _io_counters() -> dict[str, int | None]:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"read_bytes", "write_bytes"}:
                values[key] = int(value.strip())
        return {"read_bytes": values.get("read_bytes"), "write_bytes": values.get("write_bytes")}
    except (OSError, ValueError):
        return {"read_bytes": None, "write_bytes": None}


def _cpu_seconds() -> dict[str, float]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"user": usage.ru_utime, "system": usage.ru_stime}


def _counter_delta(before: dict[str, int | None], after: dict[str, int | None]) -> dict[str, int | None]:
    return {
        name: after[name] - before[name] if before[name] is not None and after[name] is not None else None
        for name in before
    }


def _thread_count() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("Threads:"):
                return int(line.partition(":")[2].strip())
    except (OSError, ValueError):
        pass
    return threading.active_count()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def _write_json_atomically(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _wait_for_start(ready_path: Path, go_path: Path) -> tuple[float, float]:
    ready_path.write_text("ready\n", encoding="ascii")
    while not go_path.is_file():
        time.sleep(0.005)
    start_at = json.loads(go_path.read_text(encoding="ascii"))["start_monotonic"]
    while True:
        remaining = start_at - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(0.005, remaining))
    started_at = time.monotonic()
    return float(start_at), started_at


def _new_snapshot_watcher() -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("snapshot write counting requires Linux inotify")
    libc = ctypes.CDLL(None, use_errno=True)
    init = libc.inotify_init1
    init.argtypes = [ctypes.c_int]
    init.restype = ctypes.c_int
    fd = init(os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _watch_snapshot_directory(fd: int, path: Path) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    add_watch = libc.inotify_add_watch
    add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    add_watch.restype = ctypes.c_int
    watch = add_watch(fd, os.fsencode(path), _IN_MOVED_TO)
    if watch < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(path))
    return watch


def _drain_snapshot_events(
    fd: int,
    watches: dict[int, tuple[int, Path]],
    counts: dict[int, dict[str, int]],
    start_wall_ns: int,
    end_wall_ns: int,
) -> None:
    while True:
        try:
            events = os.read(fd, 64 * 1024)
        except BlockingIOError:
            return
        if not events:
            return
        offset = 0
        while offset + _INOTIFY_EVENT.size <= len(events):
            watch, mask, _cookie, name_size = _INOTIFY_EVENT.unpack_from(events, offset)
            offset += _INOTIFY_EVENT.size
            raw_name = events[offset : offset + name_size].split(b"\0", 1)[0]
            offset += name_size
            watched = watches.get(watch)
            name = raw_name.decode("utf-8", errors="replace")
            channel = {"progress-v1.json": "v1", "progress-v2.json": "v2"}.get(name)
            if mask & _IN_MOVED_TO and watched is not None and channel is not None:
                worker_index, directory = watched
                try:
                    modified_ns = (directory / name).stat().st_mtime_ns
                except OSError:
                    continue
                if start_wall_ns <= modified_ns <= end_wall_ns:
                    counts[worker_index][channel] += 1


def _validate_import_origin(expected_package: Path) -> tuple[str, str]:
    import qqtools
    import qqtools.qexp.progress as progress

    package_file = getattr(qqtools, "__file__", None)
    progress_file = getattr(progress, "__file__", None)
    if package_file is None or progress_file is None:
        raise RuntimeError("qqtools package or progress module has no source path")
    expected = expected_package.resolve()
    actual_package = Path(package_file).resolve()
    actual_progress = Path(progress_file).resolve()
    if not actual_package.is_relative_to(expected) or not actual_progress.is_relative_to(expected):
        raise RuntimeError(
            "imported qqtools did not come from the selected package path: "
            f"expected {expected}, got {actual_package} and {actual_progress}"
        )
    return str(actual_package), str(actual_progress)


def _worker_main(args: argparse.Namespace) -> int:
    import torch

    expected_package = Path(args.expected_package)
    package_origin, progress_origin = _validate_import_origin(expected_package)
    from qqtools.qexp.progress import flush, update

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A framework may already have initialized the inter-op pool.
        pass
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected interpreter")
    device = torch.device("cuda:0")
    if args.mode in {"v1", "v2"}:
        Path(args.v1_path).parent.mkdir(parents=True, exist_ok=True)
        os.environ["QEXP_PROGRESS_PATH"] = args.v1_path
    if args.mode == "v2":
        Path(args.v2_path).parent.mkdir(parents=True, exist_ok=True)
        os.environ["QEXP_PROGRESS_V2_PATH"] = args.v2_path

    left = torch.rand((args.matrix_size, args.matrix_size), device=device, dtype=torch.float32)
    right = torch.rand((args.matrix_size, args.matrix_size), device=device, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(args.warmup_steps):
            _ = torch.mm(left, right)
        torch.cuda.synchronize(device)

    start_at, started_at = _wait_for_start(Path(args.ready_path), Path(args.go_path))
    start_lateness = max(0.0, started_at - start_at)
    if start_lateness > args.max_start_lateness_seconds:
        raise RuntimeError(f"worker start was late by {start_lateness:.3f}s")
    stop_at = started_at + args.window_seconds
    cpu_start = _cpu_seconds()
    io_start = _io_counters()
    step_count = 0
    producer_samples: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
    enqueue_samples: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
    gpu_samples: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
    completed_step_samples: deque[float] = deque(maxlen=_LATENCY_SAMPLE_LIMIT)
    event_batch: list[tuple[Any, Any, int]] = []
    peak_threads = _thread_count()

    def complete_batch() -> None:
        nonlocal peak_threads
        if not event_batch:
            return
        torch.cuda.synchronize(device)
        completed_at_ns = time.perf_counter_ns()
        for begin_event, end_event, step_started_ns in event_batch:
            gpu_samples.append(float(begin_event.elapsed_time(end_event)))
            if args.sync_every == 1:
                completed_step_samples.append((completed_at_ns - step_started_ns) / 1_000_000)
        event_batch.clear()
        peak_threads = max(peak_threads, _thread_count())

    with torch.no_grad():
        while time.monotonic() < stop_at:
            step_started = time.perf_counter_ns()
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            begin_event.record()
            _product = torch.mm(left, right)
            end_event.record()

            step_count += 1
            if args.mode == "v2":
                loss = 1.0 / (step_count + 1)
                producer_started = time.perf_counter_ns()
                update(
                    stage="train",
                    current=step_count,
                    total=None,
                    unit="step",
                    metrics={"loss": loss, "lr": 0.001},
                )
                producer_samples.append((time.perf_counter_ns() - producer_started) / 1_000_000)
            else:
                producer_started = time.perf_counter_ns()
                update(stage="train", current=step_count, total=None, unit="step")
                producer_samples.append((time.perf_counter_ns() - producer_started) / 1_000_000)

            enqueue_samples.append((time.perf_counter_ns() - step_started) / 1_000_000)
            event_batch.append((begin_event, end_event, step_started))
            if len(event_batch) >= args.sync_every:
                complete_batch()

    complete_batch()
    finished_at = time.monotonic()
    elapsed_seconds = max(0.0, finished_at - started_at)
    cpu_finish = _cpu_seconds()
    io_finish = _io_counters()
    try:
        flush(timeout=0.1)
    except Exception:
        # Closing the advisory producer is outside the timed training window.
        pass
    peak_threads = max(peak_threads, _thread_count())
    io = _counter_delta(io_start, io_finish)
    result = {
        "worker_index": args.worker_index,
        "profile": args.profile,
        "steps": step_count,
        "elapsed_seconds": elapsed_seconds,
        "window_started_at_monotonic": started_at,
        "window_ended_at_monotonic": finished_at,
        "steps_per_second": step_count / elapsed_seconds if elapsed_seconds else 0.0,
        "start_lateness_seconds": start_lateness,
        "user_cpu_seconds": cpu_finish["user"] - cpu_start["user"],
        "system_cpu_seconds": cpu_finish["system"] - cpu_start["system"],
        "producer_call_p99_ms": _percentile(producer_samples),
        "producer_call_samples_total": step_count,
        "producer_call_samples_retained": len(producer_samples),
        "producer_call_samples_truncated": step_count > len(producer_samples),
        "step_enqueue_p99_ms": _percentile(enqueue_samples),
        "step_enqueue_samples_total": step_count,
        "step_enqueue_samples_retained": len(enqueue_samples),
        "step_enqueue_samples_truncated": step_count > len(enqueue_samples),
        "full_completed_step_latency_p99_ms": _percentile(completed_step_samples),
        "full_completed_step_samples_total": step_count if args.sync_every == 1 else 0,
        "full_completed_step_samples_retained": len(completed_step_samples),
        "full_completed_step_samples_truncated": args.sync_every == 1 and step_count > len(completed_step_samples),
        "gpu_kernel_p99_ms": _percentile(gpu_samples),
        "gpu_kernel_samples_total": step_count,
        "gpu_kernel_samples_retained": len(gpu_samples),
        "gpu_kernel_samples_truncated": step_count > len(gpu_samples),
        "peak_rss_bytes": _rss_bytes(),
        "peak_thread_count": peak_threads,
        "io_read_bytes": io["read_bytes"],
        "io_write_bytes": io["write_bytes"],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "visible_gpu_index": args.gpu_index,
        "device_in_process": str(device),
        "package_origin": package_origin,
        "progress_module_origin": progress_origin,
        "mode": args.mode,
    }
    _write_json(Path(args.result_path), result)
    return 0


def _client_read(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    read_size = 0
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return "missing", 0
    except OSError:
        return "error", 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _SNAPSHOT_LIMIT_BYTES:
            return "invalid", 0
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            data = handle.read(_SNAPSHOT_LIMIT_BYTES + 1)
        read_size = len(data)
        if len(data) > _SNAPSHOT_LIMIT_BYTES:
            return "invalid", read_size
        json.loads(data.decode("utf-8"))
        return "read", read_size
    except (OSError, UnicodeError, json.JSONDecodeError):
        return "malformed", read_size
    finally:
        if fd >= 0:
            os.close(fd)


def _client_main(args: argparse.Namespace) -> int:
    files = json.loads(Path(args.snapshot_manifest).read_text(encoding="utf-8"))
    start_at, started_at = _wait_for_start(Path(args.ready_path), Path(args.go_path))
    start_lateness = max(0.0, started_at - start_at)
    if start_lateness > args.max_start_lateness_seconds:
        raise RuntimeError(f"client start was late by {start_lateness:.3f}s")
    stop_at = started_at + args.window_seconds
    cpu_start = _cpu_seconds()
    io_start = _io_counters()
    counts = {
        key: 0
        for key in (
            "poll_iterations",
            "files_polled",
            "json_objects_read",
            "missing_reads",
            "read_errors",
            "malformed_reads",
        )
    }
    bytes_read = 0
    peak_threads = _thread_count()
    poll_interval = args.poll_interval_seconds
    while time.monotonic() < stop_at:
        poll_started = time.monotonic()
        counts["poll_iterations"] += 1
        for entry in files:
            for channel in entry["channels"]:
                counts["files_polled"] += 1
                status, size = _client_read(Path(entry[channel]))
                bytes_read += size
                if status == "read":
                    counts["json_objects_read"] += 1
                elif status == "missing":
                    counts["missing_reads"] += 1
                elif status == "malformed":
                    counts["malformed_reads"] += 1
                else:
                    counts["read_errors"] += 1
        peak_threads = max(peak_threads, _thread_count())
        remaining = poll_interval - (time.monotonic() - poll_started)
        if remaining > 0:
            time.sleep(remaining)
    finished_at = time.monotonic()
    cpu_finish = _cpu_seconds()
    io = _counter_delta(io_start, _io_counters())
    result = {
        "client_index": args.client_index,
        "channels": args.channels,
        "poll_interval_seconds": poll_interval,
        "elapsed_seconds": max(0.0, finished_at - started_at),
        "window_started_at_monotonic": started_at,
        "window_ended_at_monotonic": finished_at,
        "start_lateness_seconds": start_lateness,
        "user_cpu_seconds": cpu_finish["user"] - cpu_start["user"],
        "system_cpu_seconds": cpu_finish["system"] - cpu_start["system"],
        "bytes_read": bytes_read,
        "peak_rss_bytes": _rss_bytes(),
        "peak_thread_count": peak_threads,
        **counts,
        **{f"io_{key}": value for key, value in io.items()},
    }
    _write_json(Path(args.result_path), result)
    return 0


def _launch_environment(
    import_root: Path, gpu_index: int, cpu_threads: int, progress_interval: float
) -> dict[str, str]:
    env = os.environ.copy()
    # Put only the selected package on PYTHONPATH so an inherited editable
    # checkout cannot silently replace either side of the comparison.
    env["PYTHONPATH"] = str(import_root)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env.pop("QEXP_PROGRESS_PATH", None)
    env.pop("QEXP_PROGRESS_V2_PATH", None)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(cpu_threads)
    env["QEXP_PROGRESS_INTERVAL_SECONDS"] = str(progress_interval)
    return env


def _spawn(
    *,
    config: dict[str, Any],
    env: dict[str, str],
    log_root: Path,
    label: str,
) -> dict[str, Any]:
    config_path = log_root / f"{label}.json"
    stderr_path = log_root / f"{label}.stderr"
    _write_json(config_path, config)
    stderr_stream = stderr_path.open("wb")
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--child-config", str(config_path)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=stderr_stream,
        )
    except Exception:
        stderr_stream.close()
        raise
    return {
        "process": process,
        "stderr_path": stderr_path,
        "stderr_stream": stderr_stream,
        "label": label,
    }


def _stop_children(children: list[dict[str, Any]], *, deadline: float) -> None:
    for child in children:
        process = child["process"]
        if process.poll() is None:
            process.terminate()
    stop_deadline = min(deadline, time.monotonic() + 5.0)
    for child in children:
        process = child["process"]
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, stop_deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)
        child["stderr_stream"].close()


def _child_error(child: dict[str, Any]) -> str:
    stderr_path: Path = child["stderr_path"]
    try:
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stderr = ""
    return stderr[-4000:]


def _run_profile(
    *,
    profile: str,
    measurement_kind: str,
    sync_every: int,
    import_root: Path,
    expected_package: Path,
    workers: int,
    clients: int,
    repetition: int,
    case_index: int,
    case_deadline: float,
    args: argparse.Namespace,
    work_root: Path,
) -> dict[str, Any]:
    if time.monotonic() >= case_deadline:
        raise TimeoutError(f"benchmark cell {workers} workers/{clients} clients exceeded ten minutes")
    run_root = work_root / f"case-{case_index:02d}-{measurement_kind}-{profile}-rep-{repetition:02d}"
    children: list[dict[str, Any]] = []
    watch_fd: int | None = None
    try:
        run_root.mkdir(parents=True)
        watch_fd = _new_snapshot_watcher()
        ready_paths: list[Path] = []
        worker_results: list[Path] = []
        client_results: list[Path] = []
        snapshot_entries = []
        worker_watches: dict[int, tuple[int, Path]] = {}
        write_counts = {index: {"v1": 0, "v2": 0} for index in range(workers)}
        mode = "v2" if profile == "candidate_v2" else "v1"
        channels = ["v1", "v2"] if mode == "v2" else ["v1"]
        env = _launch_environment(import_root, args.gpu_index, args.cpu_threads, args.progress_interval_seconds)

        for index in range(workers):
            worker_root = run_root / f"worker-{index:03d}"
            worker_root.mkdir()
            worker_watches[_watch_snapshot_directory(watch_fd, worker_root)] = (index, worker_root)
            v1_path = worker_root / "progress-v1.json"
            v2_path = worker_root / "progress-v2.json"
            ready = worker_root / "ready"
            result_path = worker_root / "result.json"
            ready_paths.append(ready)
            worker_results.append(result_path)
            snapshot_entries.append({"v1": str(v1_path), "v2": str(v2_path), "channels": channels})
            config = {
                "role": "worker",
                "profile": profile,
                "mode": mode,
                "expected_package": str(expected_package),
                "ready_path": str(ready),
                "go_path": str(run_root / "go.json"),
                "result_path": str(result_path),
                "v1_path": str(v1_path),
                "v2_path": str(v2_path),
                "worker_index": index,
                "gpu_index": args.gpu_index,
                "matrix_size": args.matrix_size,
                "window_seconds": args.window_seconds,
                "warmup_steps": args.warmup_steps,
                "sync_every": sync_every,
                "cpu_threads": args.cpu_threads,
                "seed": 1729 + repetition * 101 + index,
                "max_start_lateness_seconds": args.max_start_lateness_seconds,
            }
            children.append(_spawn(config=config, env=env, log_root=run_root, label=f"worker-{index:03d}"))

        manifest_path = run_root / "snapshots.json"
        manifest_path.write_text(json.dumps(snapshot_entries, separators=(",", ":")), encoding="utf-8")
        for index in range(clients):
            client_root = run_root / f"client-{index:03d}"
            client_root.mkdir()
            ready = client_root / "ready"
            result_path = client_root / "result.json"
            ready_paths.append(ready)
            client_results.append(result_path)
            config = {
                "role": "client",
                "profile": profile,
                "ready_path": str(ready),
                "go_path": str(run_root / "go.json"),
                "result_path": str(result_path),
                "snapshot_manifest": str(manifest_path),
                "client_index": index,
                "channels": channels,
                "poll_interval_seconds": args.poll_interval_seconds,
                "window_seconds": args.window_seconds,
                "max_start_lateness_seconds": args.max_start_lateness_seconds,
            }
            children.append(_spawn(config=config, env=env, log_root=run_root, label=f"client-{index:03d}"))

        deadline = min(time.monotonic() + args.startup_timeout_seconds, case_deadline)
        while not all(path.is_file() for path in ready_paths):
            failed = [child for child in children if child["process"].poll() not in (None, 0)]
            if failed:
                details = "\n".join(f"{item['label']}: {_child_error(item)}" for item in failed)
                raise RuntimeError(f"benchmark child failed before start barrier:\n{details}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"children did not reach the start barrier for {profile}")
            time.sleep(0.05)
        start_clock = time.monotonic()
        wall_clock_ns = time.time_ns()
        start_at = start_clock + args.start_lead_seconds
        start_wall_ns = wall_clock_ns + int(args.start_lead_seconds * 1_000_000_000)
        end_wall_ns = start_wall_ns + int(args.window_seconds * 1_000_000_000)
        _write_json_atomically(run_root / "go.json", {"start_monotonic": start_at, "start_wall_ns": start_wall_ns})
        run_deadline = min(start_at + args.window_seconds + args.completion_timeout_seconds, case_deadline)
        while any(child["process"].poll() is None for child in children):
            _drain_snapshot_events(watch_fd, worker_watches, write_counts, start_wall_ns, end_wall_ns)
            failed = [child for child in children if child["process"].poll() not in (None, 0)]
            if failed:
                details = "\n".join(f"{item['label']}: {_child_error(item)}" for item in failed)
                raise RuntimeError(f"benchmark child failed during {profile}:\n{details}")
            if time.monotonic() >= run_deadline:
                raise TimeoutError(f"benchmark children did not finish profile {profile}")
            time.sleep(0.05)
        _drain_snapshot_events(watch_fd, worker_watches, write_counts, start_wall_ns, end_wall_ns)
        for child in children:
            child["stderr_stream"].close()
        worker_data = [json.loads(path.read_text(encoding="utf-8")) for path in worker_results]
        client_data = [json.loads(path.read_text(encoding="utf-8")) for path in client_results]
        if profile == "candidate_v2":
            insufficient = [index for index, count in write_counts.items() if count["v1"] < 3 or count["v2"] < 3]
            if insufficient:
                raise RuntimeError(f"fewer than three in-window v1/v2 snapshot replacements for workers {insufficient}")
        for item in worker_data:
            item["snapshot_replace_count"] = write_counts[item["worker_index"]]

        windows = worker_data + client_data
        shared_start = max(item["window_started_at_monotonic"] for item in windows)
        shared_end = min(item["window_ended_at_monotonic"] for item in windows)
        shared_overlap = max(0.0, shared_end - shared_start)
        minimum_overlap = max(0.0, args.window_seconds - args.max_start_lateness_seconds)
        if clients and shared_overlap < minimum_overlap:
            raise RuntimeError(
                f"client/worker measurement-window overlap {shared_overlap:.3f}s is below "
                f"the required {minimum_overlap:.3f}s"
            )

        elapsed = max((item["elapsed_seconds"] for item in worker_data), default=0.0)
        total_steps = sum(item["steps"] for item in worker_data)
        return {
            "profile": profile,
            "measurement_kind": measurement_kind,
            "sync_every": sync_every,
            "workers": workers,
            "clients_requested": clients,
            "client_channels": channels,
            "repetition": repetition,
            "steps": total_steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": total_steps / elapsed if elapsed else 0.0,
            "shared_worker_client_window_overlap_seconds": shared_overlap,
            "minimum_required_window_overlap_seconds": minimum_overlap if clients else None,
            "max_worker_start_lateness_seconds": max(item["start_lateness_seconds"] for item in worker_data),
            "max_client_start_lateness_seconds": max(
                (item["start_lateness_seconds"] for item in client_data), default=None
            ),
            "max_worker_producer_call_p99_ms": max(
                (item["producer_call_p99_ms"] or 0.0 for item in worker_data), default=0.0
            ),
            "max_worker_step_enqueue_p99_ms": max(
                (item["step_enqueue_p99_ms"] or 0.0 for item in worker_data), default=0.0
            ),
            "max_worker_full_completed_step_latency_p99_ms": max(
                (item["full_completed_step_latency_p99_ms"] or 0.0 for item in worker_data), default=0.0
            )
            if sync_every == 1
            else None,
            "max_worker_gpu_kernel_p99_ms": max(
                (item["gpu_kernel_p99_ms"] or 0.0 for item in worker_data), default=0.0
            ),
            "worker_user_cpu_seconds_sum": sum(item["user_cpu_seconds"] for item in worker_data),
            "worker_system_cpu_seconds_sum": sum(item["system_cpu_seconds"] for item in worker_data),
            "client_user_cpu_seconds_sum": sum(item["user_cpu_seconds"] for item in client_data),
            "client_system_cpu_seconds_sum": sum(item["system_cpu_seconds"] for item in client_data),
            "worker_peak_rss_bytes_sum": sum(item["peak_rss_bytes"] for item in worker_data),
            "worker_peak_thread_count_sum": sum(item["peak_thread_count"] for item in worker_data),
            "worker_io_read_bytes_sum": _sum_optional(item["io_read_bytes"] for item in worker_data),
            "worker_io_write_bytes_sum": _sum_optional(item["io_write_bytes"] for item in worker_data),
            "workers_detail": worker_data,
            "clients_detail": client_data,
            "process_count_during_window": workers + len(client_data),
            "client_model": "local snapshot JSON polling proxy; not the qexp viewer or agent projection",
        }
    finally:
        _stop_children(children, deadline=case_deadline + _CASE_CLEANUP_RESERVE_SECONDS)
        if watch_fd is not None:
            os.close(watch_fd)


def _sum_optional(values: Any) -> int | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return sum(materialized)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _robust_spread(values: list[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    deviations = [abs(value - center) for value in values]
    return 1.4826 * statistics.median(deviations)


def _compare(group: list[dict[str, Any]], base: str, candidate: str, repeats: int) -> dict[str, Any]:
    base_values = [run["steps_per_second"] for run in group if run["profile"] == base]
    candidate_values = [run["steps_per_second"] for run in group if run["profile"] == candidate]
    base_median = _median(base_values)
    candidate_median = _median(candidate_values)
    base_by_repeat = {run["repetition"]: run["steps_per_second"] for run in group if run["profile"] == base}
    candidate_by_repeat = {run["repetition"]: run["steps_per_second"] for run in group if run["profile"] == candidate}
    paired_repeats = sorted(base_by_repeat.keys() & candidate_by_repeat.keys())
    paired_results = [
        {
            "repetition": repetition,
            "regression_percent": (base_by_repeat[repetition] - candidate_by_repeat[repetition])
            / base_by_repeat[repetition]
            * 100.0,
        }
        for repetition in paired_repeats
        if base_by_repeat[repetition] > 0
    ]
    paired_regressions = [item["regression_percent"] for item in paired_results]
    regression = _median(paired_regressions)
    spread = _robust_spread(paired_regressions)
    observed_range = max(paired_regressions) - min(paired_regressions) if paired_regressions else None
    if (
        len(paired_regressions) < max(3, repeats)
        or regression is None
        or spread is None
        or spread > 1.0
        or observed_range is None
        or observed_range > _THROUGHPUT_TARGET_PERCENT
    ):
        classification = "inconclusive_noise_or_insufficient_repeats"
    elif regression - spread <= _THROUGHPUT_TARGET_PERCENT < regression + spread or min(
        paired_regressions
    ) <= _THROUGHPUT_TARGET_PERCENT < max(paired_regressions):
        classification = "inconclusive_noise_crosses_3_percent_target"
    elif regression <= _THROUGHPUT_TARGET_PERCENT:
        classification = "within_3_percent_target"
    else:
        classification = "above_3_percent_target"
    return {
        "candidate": candidate,
        "baseline": base,
        "baseline_median_steps_per_second": base_median,
        "candidate_median_steps_per_second": candidate_median,
        "paired_repetitions": len(paired_regressions),
        "paired_regressions_by_repetition": paired_results,
        "paired_median_regression_percent": regression,
        "paired_regression_spread_percentage_points": spread,
        "paired_regression_range_percentage_points": observed_range,
        "target_percent": _THROUGHPUT_TARGET_PERCENT,
        "classification": classification,
    }


def _comparisons(runs: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    runs = [run for run in runs if run["measurement_kind"] == "throughput"]
    keys = sorted({(run["workers"], run["clients_requested"]) for run in runs})
    comparisons = []
    for workers, clients in keys:
        group = [run for run in runs if run["workers"] == workers and run["clients_requested"] == clients]
        for base, candidate in (
            ("baseline_v1", "candidate_v1"),
            ("baseline_v1", "candidate_v2"),
            ("candidate_v1", "candidate_v2"),
        ):
            comparisons.append({"workers": workers, "clients": clients, **_compare(group, base, candidate, repeats)})
    return comparisons


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-package-path",
        required=True,
        help="installed baseline package directory or its parent containing qqtools/",
    )
    parser.add_argument(
        "--candidate-source-path",
        required=True,
        help="candidate source root containing qqtools/, normally the repository src/ directory",
    )
    parser.add_argument(
        "--gpu-index", type=_nonnegative_int, required=True, help="physical GPU index exposed as cuda:0"
    )
    parser.add_argument("--workers", type=_positive_int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--clients", type=_nonnegative_int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--matrix-size", type=_positive_int, default=4096)
    parser.add_argument("--window-seconds", type=_positive_float, default=8.0)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--latency-case",
        type=_nonnegative_int,
        nargs=2,
        metavar=("WORKERS", "CLIENTS"),
        help="choose the one representative sync-every-step cell (default: 4 workers/1 client, or first listed cell)",
    )
    parser.add_argument("--cpu-threads", type=_positive_int, default=1)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=5)
    parser.add_argument("--sync-every", type=_positive_int, default=16)
    parser.add_argument("--progress-interval-seconds", type=_positive_float, default=1.0)
    parser.add_argument("--poll-interval-seconds", type=_positive_float, default=0.2)
    parser.add_argument("--start-lead-seconds", type=_positive_float, default=5.0)
    parser.add_argument("--max-start-lateness-seconds", type=_positive_float, default=0.25)
    parser.add_argument("--startup-timeout-seconds", type=_positive_float, default=180.0)
    parser.add_argument("--completion-timeout-seconds", type=_positive_float, default=120.0)
    parser.add_argument("--output", default="-", help="JSON result path, or - for stdout")
    return parser


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    baseline_import_root, baseline_package = _package_paths(args.baseline_package_path, label="baseline path")
    candidate_import_root, candidate_package = _package_paths(args.candidate_source_path, label="candidate source path")
    if args.window_seconds < 3 * args.progress_interval_seconds:
        raise ValueError(
            "window must be at least three progress intervals to measure an initial and two periodic writes"
        )
    cases = [(worker_count, client_count) for worker_count in args.workers for client_count in args.clients]
    latency_case = (
        tuple(args.latency_case) if args.latency_case is not None else ((4, 1) if (4, 1) in cases else cases[0])
    )
    if latency_case[0] == 0 or latency_case not in cases:
        raise ValueError("--latency-case must select a listed worker/client matrix cell")
    throughput_runs: list[dict[str, Any]] = []
    latency_runs: list[dict[str, Any]] = []
    case_labels = {
        "baseline_v1": (baseline_import_root, baseline_package),
        "candidate_v1": (candidate_import_root, candidate_package),
        "candidate_v2": (candidate_import_root, candidate_package),
    }
    output_parent = None
    if args.output != "-":
        output_parent = Path(args.output).expanduser().resolve().parent
        output_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qexp-progress-gpu-", dir=output_parent) as temporary_root:
        work_root = Path(temporary_root)
        for case_index, (worker_count, client_count) in enumerate(cases):
            case_deadline = time.monotonic() + _CASE_LIMIT_SECONDS - _CASE_CLEANUP_RESERVE_SECONDS
            for repetition in range(args.repeats):
                rotation = (case_index + repetition) % len(_PROFILES)
                profile_order = _PROFILES[rotation:] + _PROFILES[:rotation]
                for profile in profile_order:
                    import_root, expected_package = case_labels[profile]
                    throughput_runs.append(
                        _run_profile(
                            profile=profile,
                            measurement_kind="throughput",
                            sync_every=args.sync_every,
                            import_root=import_root,
                            expected_package=expected_package,
                            workers=worker_count,
                            clients=client_count,
                            repetition=repetition,
                            case_index=case_index,
                            case_deadline=case_deadline,
                            args=args,
                            work_root=work_root,
                        )
                    )
            if latency_case == (worker_count, client_count):
                for profile in _PROFILES:
                    import_root, expected_package = case_labels[profile]
                    latency_runs.append(
                        _run_profile(
                            profile=profile,
                            measurement_kind="full_step_latency",
                            sync_every=1,
                            import_root=import_root,
                            expected_package=expected_package,
                            workers=worker_count,
                            clients=client_count,
                            repetition=0,
                            case_index=case_index,
                            case_deadline=case_deadline,
                            args=args,
                            work_root=work_root,
                        )
                    )

    report = {
        "schema_version": 1,
        "benchmark": "qexp-progress-torch-gpu-producer",
        "interpretation": {
            "workload": "synthetic single-GPU float32 matrix multiplication; not a representative model or distributed training run",
            "gpu_allocation": "all worker processes share the selected physical GPU; each process sees it as cuda:0",
            "client_model": "clients poll only their profile's available local channels: baseline and candidate-v1 read v1, candidate-v2 reads v1 and v2; they do not perform qexp identity/freshness validation, agent projection, shared-storage reads, or terminal rendering",
            "timing": "throughput_runs use the configured periodic synchronization; an optional representative latency case separately synchronizes every step and reports completed-step wall p99 including CUDA synchronization",
            "io": "/proc/self/io read_bytes/write_bytes deltas over timed worker/client windows where available; these are process kernel counters, not full physical-device accounting",
            "comparison": "per-repetition throughput regressions are paired by repetition; classification uses their median and robust spread in percentage points, while raw throughput medians are descriptive",
            "snapshot_writes": "Linux inotify counts atomic replacements whose file modification times fall in the timed window; candidate-v2 requires an initial and two periodic replacements on both channels per worker",
            "latency_samples": f"p99 uses the most recent at most {_LATENCY_SAMPLE_LIMIT} samples; each worker reports total, retained, and truncation status",
            "acceptance": "this microbenchmark is one performance input and does not establish the pitch's complete Torch, DDP, isolation, rollout, or viewer acceptance",
        },
        "configuration": {
            "python_executable": sys.executable,
            "baseline_package_path": str(baseline_package),
            "candidate_source_path": str(candidate_package),
            "gpu_index": args.gpu_index,
            "workers": args.workers,
            "clients": args.clients,
            "matrix_size": args.matrix_size,
            "window_seconds": args.window_seconds,
            "case_limit_seconds": _CASE_LIMIT_SECONDS,
            "case_cleanup_reserve_seconds": _CASE_CLEANUP_RESERVE_SECONDS,
            "repeats": args.repeats,
            "latency_case": list(latency_case) if latency_case is not None else None,
            "cpu_threads_per_worker": args.cpu_threads,
            "warmup_steps": args.warmup_steps,
            "sync_every": args.sync_every,
            "step_latency_sync_every": 1,
            "progress_interval_seconds": args.progress_interval_seconds,
            "poll_interval_seconds": args.poll_interval_seconds,
            "start_lead_seconds": args.start_lead_seconds,
            "max_start_lateness_seconds": args.max_start_lateness_seconds,
            "minimum_window_seconds": 3 * args.progress_interval_seconds,
            "minimum_candidate_v2_snapshot_replacements_per_channel": 3,
            "profiles": [
                {"name": "baseline_v1", "package": str(baseline_package), "writer": "v1 only"},
                {
                    "name": "candidate_v1",
                    "package": str(candidate_package),
                    "writer": "v1 only; extended protocol disabled",
                },
                {
                    "name": "candidate_v2",
                    "package": str(candidate_package),
                    "writer": "v1 and v2; finite scalar metrics",
                },
            ],
        },
        "throughput_runs": throughput_runs,
        "full_step_latency_runs": latency_runs,
        "comparisons": _comparisons(throughput_runs, args.repeats),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--child-config":
        try:
            config = argparse.Namespace(**json.loads(Path(argv[1]).read_text(encoding="utf-8")))
            return _client_main(config) if config.role == "client" else _worker_main(config)
        except Exception:
            traceback.print_exc()
            return 1

    parser = _cli_parser()
    args = parser.parse_args(argv)
    try:
        report = _run_cli(args)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output == "-":
        print(encoded)
    else:
        output = Path(args.output).expanduser().resolve()
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

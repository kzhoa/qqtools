"""Shared subprocess helpers for installed-wheel qexp E2E tests."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


# A release-wheel task starts an agent and a runner in fresh Python interpreters.
TASK_TERMINAL_TIMEOUT_SECONDS = 90.0


def ensure_site_packages_import() -> str:
    import qqtools

    imported_from = Path(qqtools.__file__).resolve()
    if "site-packages" not in str(imported_from):
        raise RuntimeError(f"qqtools was not imported from site-packages: {imported_from}")
    return str(imported_from)


def make_env(base: Path) -> dict[str, str]:
    home = base / "home"
    home.mkdir(parents=True, exist_ok=True)
    bin_dir = base / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    chronyc = bin_dir / "chronyc"
    chronyc.write_text(
        "#!/bin/sh\n"
        "cat <<'EOF'\n"
        "System time     : 0.000001 seconds fast of NTP time\n"
        "Root dispersion : 0.000001 seconds\n"
        "Leap status     : Normal\n"
        "EOF\n",
        encoding="utf-8",
    )
    chronyc.chmod(0o755)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["HOME"] = str(home)
    env["MPLCONFIGDIR"] = str(base / "mplconfig")
    env["QEXP_VISIBLE_GPUS"] = "0"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def make_layout(base: Path) -> tuple[Path, Path, Path]:
    shared_root = base / ".qexp"
    runtime_root = base / "runtime"
    return base, shared_root, runtime_root


def run(
    args: list[str], *, env: dict[str, str], check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, capture_output=True, env=env)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\n"
            f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def jrun(args: list[str], *, env: dict[str, str]) -> object:
    result = run(args, env=env)
    text = result.stdout.strip()
    return json.loads(text) if text else None


def wait_for(predicate, *, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {label}")


def stop_agent(common: list[str], *, env: dict[str, str], pid_path: Path) -> None:
    run([*common, "agent", "stop"], env=env, check=False)
    wait_for(lambda: not pid_path.exists(), timeout=10, label="background agent pid cleanup")

"""Shared subprocess helpers for installed-wheel qexp E2E tests."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# A release-wheel task starts an agent and a runner in fresh Python interpreters.
TASK_TERMINAL_TIMEOUT_SECONDS = 90.0
COMMAND_TIMEOUT_SECONDS = TASK_TERMINAL_TIMEOUT_SECONDS + 30.0


def ensure_site_packages_import() -> str:
    import qqtools

    imported_from = Path(qqtools.__file__).resolve()
    if "site-packages" not in str(imported_from):
        raise RuntimeError(f"qqtools was not imported from site-packages: {imported_from}")
    return str(imported_from)


def make_env(base: Path) -> dict[str, str]:
    base = base.resolve()
    home = base / "home"
    home.mkdir(parents=True, exist_ok=True)
    bin_dir = base / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    local_temp = base / "tmp"
    local_temp.mkdir(parents=True, exist_ok=True)
    tmux_temp = base / "tmux"
    tmux_temp.mkdir(parents=True, exist_ok=True)
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
    for name in ("PYTHONPATH", "TMUX", "TMUX_PANE"):
        env.pop(name, None)
    env["HOME"] = str(home)
    env["XDG_CACHE_HOME"] = str(base / "xdg" / "cache")
    env["XDG_CONFIG_HOME"] = str(base / "xdg" / "config")
    env["XDG_DATA_HOME"] = str(base / "xdg" / "data")
    env["MPLCONFIGDIR"] = str(base / "mplconfig")
    env["TMPDIR"] = str(local_temp)
    env["TMP"] = str(local_temp)
    env["TEMP"] = str(local_temp)
    env["TMUX_TMPDIR"] = str(tmux_temp)
    env["QEXP_MACHINE_RUNTIME_ROOT"] = str(base / "machine-runtime")
    env["QEXP_VISIBLE_GPUS"] = "0"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def make_layout(base: Path) -> tuple[Path, Path, Path]:
    shared_root = base / ".qexp"
    runtime_root = base / "runtime"
    return base, shared_root, runtime_root


def run(args: list[str], *, env: dict[str, str], check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            text=True,
            capture_output=True,
            env=env,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"command timed out after {COMMAND_TIMEOUT_SECONDS:.0f}s: {' '.join(args)}\n"
            f"stdout:\n{error.stdout or ''}\nstderr:\n{error.stderr or ''}"
        ) from error
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(args)}\n"
            f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def jrun(args: list[str], *, env: dict[str, str]) -> object:
    result = run([*args, "--format=json"], env=env)
    text = result.stdout.strip()
    return json.loads(text) if text else None


def initialize_machine_project(common: list[str], *, env: dict[str, str], agent_mode: str | None = None) -> None:
    """Initialize one machine and enroll the Project selected by an E2E command prefix."""
    shared_root = _single_argument_value(common, "--project")
    machine = _single_argument_value(common, "--machine")
    if shared_root is None or machine is None:
        raise ValueError("E2E setup requires --project and --machine")
    machine_runtime_root = _single_argument_value(common, "--machine-runtime-root")
    prefix = ["qexp"]
    if machine_runtime_root is not None:
        prefix.extend(("--machine-runtime-root", machine_runtime_root))
    init = [*prefix, "init", "--machine", machine]
    if agent_mode is not None:
        init.extend(("--agent-mode", agent_mode))
    run(init, env=env)
    project_root = str(Path(shared_root).resolve().parent)
    run([*prefix, "project", "init", project_root], env=env)
    run([*prefix, "project", "register", project_root], env=env)


def wait_for(predicate, *, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {label}")


def is_machine_agent_running(common: list[str], *, env: dict[str, str]) -> bool:
    status = jrun([*common, "agent", "status"], env=env)
    return bool(status["is_running"])


def _single_argument_value(args: list[str], name: str) -> str | None:
    values: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        option = argument.partition("=")[0]
        if option != name and option.startswith("--machine-") and name.startswith(option):
            raise RuntimeError(f"refusing cleanup with abbreviated {name} argument")
        if argument == name:
            if index + 1 >= len(args) or not args[index + 1] or args[index + 1].startswith("--"):
                raise RuntimeError(f"refusing cleanup with missing {name} value")
            values.append(args[index + 1])
            index += 2
            continue
        if argument.startswith(f"{name}="):
            value = argument.partition("=")[2]
            if not value:
                raise RuntimeError(f"refusing cleanup with missing {name} value")
            values.append(value)
        index += 1
    if len(values) > 1:
        raise RuntimeError(f"refusing cleanup with duplicate {name} arguments")
    return values[0] if values else None


def _require_test_owned_cleanup(common: list[str], env: dict[str, str]) -> None:
    home = Path(env["HOME"]).resolve()
    test_root = home.parent
    machine_runtime = Path(
        _single_argument_value(common, "--machine-runtime-root")
        or env.get("QEXP_MACHINE_RUNTIME_ROOT", home / ".qqtools" / "qexp-machine")
    ).resolve()
    local_temp = Path(env["TMPDIR"]).resolve()
    for label, path in (("machine runtime", machine_runtime), ("temporary directory", local_temp)):
        if not path.is_relative_to(test_root):
            raise RuntimeError(f"refusing to clean non-test {label}: {path}")


def stop_agent(common: list[str], *, env: dict[str, str]) -> None:
    _require_test_owned_cleanup(common, env)
    run([*common, "agent", "stop"], env=env, check=False)
    wait_for(
        lambda: not is_machine_agent_running(common, env=env),
        timeout=10,
        label="background machine agent cleanup",
    )

from __future__ import annotations

from qexp_e2e import ensure_site_packages_import, make_env, run, stop_agent

import pytest


pytestmark = [pytest.mark.e2e, pytest.mark.host_exclusive]


def _common(base, name: str) -> list[str]:
    return [
        "qexp",
        "--shared-root",
        str(base / name / ".qexp"),
        "--machine",
        name,
        "--runtime-root",
        str(base / name / "runtime"),
        "--machine-runtime-root",
        str(base / name / "machine-runtime"),
    ]


def test_installed_wheel_allows_one_default_host_authority(tmp_path) -> None:
    env = make_env(tmp_path / "host-authority")
    for name in ("TMPDIR", "TMP", "TEMP"):
        env.pop(name, None)
    first = _common(tmp_path, "first")
    second = _common(tmp_path, "second")
    try:
        run([*first, "init", "--agent-mode", "daemon"], env=env)
        run([*second, "init", "--agent-mode", "daemon"], env=env)
        run([*first, "agent", "start"], env=env)

        rejected = run([*second, "agent", "start"], env=env, check=False)
        assert rejected.returncode != 0
        assert "scheduler authority" in rejected.stderr

        stop_agent(first, env=env)
        run([*second, "agent", "start"], env=env)
        assert "site-packages" in ensure_site_packages_import()
    finally:
        stop_agent(first, env=env)
        stop_agent(second, env=env)

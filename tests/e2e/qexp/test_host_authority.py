from __future__ import annotations

import pytest
from qexp_e2e import ensure_site_packages_import, initialize_machine_project, make_env, run, stop_agent

pytestmark = [pytest.mark.e2e, pytest.mark.host_exclusive]


def _common(base, name: str) -> list[str]:
    return [
        "qexp",
        "--project",
        str(base / name / ".qexp"),
        "--machine",
        name,
        "--runtime-root",
        str(base / name / "runtime"),
        "--machine-runtime-root",
        str(base / name / "machine-runtime"),
    ]


def test_installed_wheel_allows_one_default_host_authority(tmp_path) -> None:
    base = tmp_path / "host-authority"
    env = make_env(base)
    first = _common(base, "first")
    second = _common(base, "second")
    try:
        initialize_machine_project(first, env=env, agent_mode="daemon")
        initialize_machine_project(second, env=env, agent_mode="daemon")
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

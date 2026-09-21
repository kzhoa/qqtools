"""Installed public pagination, legacy listing and structured errors."""

import json

from qexp_e2e import (
    ensure_site_packages_import,
    initialize_machine_project,
    jrun,
    make_env,
    make_layout,
    run,
    stop_agent,
)


def test_installed_task_pagination(tmp_path):
    base, shared_root, runtime_root = make_layout(tmp_path / "pages")
    env = make_env(base)
    ensure_site_packages_import()
    common = ["qexp", "--shared-root", str(shared_root), "--machine", "gpu-1", "--runtime-root", str(runtime_root)]
    try:
        initialize_machine_project(common, env=env)
        for key in ["page-a", "page-b", "page-c"]:
            run([*common, "submit", "--no-activate", "--task-id", key, "--", "true"], env=env)
        legacy = jrun([*common, "task", "list"], env=env)
        first = jrun([*common, "task", "list", "--page-size", "1"], env=env)
        assert first["items"] == legacy[:1]
        assert first["consistency"] == "live"
        rest = jrun([*common, "task", "list", "--cursor", first["next_cursor"]], env=env)
        assert rest["items"] == legacy[1:]
        assert rest["next_cursor"] is None
        invalid = run([*common, "task", "list", "--page-size", "oops", "--format=json"], env=env, check=False)
        assert invalid.returncode == 2
        assert json.loads(invalid.stdout)["error"]["code"] == "invalid_argument"
    finally:
        stop_agent(common, env=env)

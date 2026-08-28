from __future__ import annotations

from qexp_e2e import ensure_site_packages_import, jrun, make_env, make_layout, run, stop_agent


def test_installed_wheel_cli_flow(tmp_path):
    base, shared_root, runtime_root = make_layout(tmp_path / "qexp-cli")
    env = make_env(base)
    imported_from = ensure_site_packages_import()
    common = [
        "qexp",
        "--shared-root",
        str(shared_root),
        "--machine",
        "gpu-1",
        "--runtime-root",
        str(runtime_root),
    ]
    try:
        run([*common, "init", "--agent-mode", "daemon"], env=env)
        group = jrun([*common, "group", "create", "release-e2e", "--workers", "gpu-1"], env=env)
        submit = run(
            [
                *common,
                "submit",
                "--group",
                "release-e2e",
                "--name",
                "cli-release-e2e",
                "--",
                "python",
                "-c",
                "print('cli ok')",
            ],
            env=env,
        )
        task_id = submit.stdout.strip()
        task = jrun([*common, "task", "show", task_id], env=env)
        tasks = jrun([*common, "task", "list"], env=env)
        groups = jrun([*common, "group", "list"], env=env)
        machines = jrun([*common, "machines"], env=env)

        assert "site-packages" in imported_from
        assert group["group"]["name"] == "release-e2e"
        assert task["task"]["task_id"] == task_id
        assert task["task"]["group_name"] == "release-e2e"
        assert any(item["task_id"] == task_id for item in tasks)
        assert any(item["group"]["name"] == "release-e2e" for item in groups)
        assert any(item["machine"]["machine_name"] == "gpu-1" for item in machines)
    finally:
        stop_agent(common, env=env)

from __future__ import annotations

from qexp_e2e import (
    ensure_site_packages_import,
    initialize_machine_project,
    jrun,
    make_env,
    make_layout,
    run,
    stop_agent,
)


def test_installed_wheel_cli_flow(tmp_path):
    base, shared_root, runtime_root = make_layout(tmp_path / "qexp-cli")
    env = make_env(base)
    imported_from = ensure_site_packages_import()
    common = [
        "qexp",
        "--project",
        str(shared_root),
        "--machine",
        "gpu-1",
        "--runtime-root",
        str(runtime_root),
    ]
    try:
        initialize_machine_project(common, env=env, agent_mode="daemon")
        tmux_policy = jrun([*common, "config", "show", "tmux"], env=env)
        group = jrun([*common, "group", "create", "release-e2e", "--workers", "gpu-1"], env=env)
        submit = run(
            [
                *common,
                "submit",
                "--group",
                "release-e2e",
                "--name",
                "cli-release-e2e",
                "--no-tmux",
                "--quiet",
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
        machines = jrun([*common, "machine", "list"], env=env)
        followed = run([*common, "task", "logs", task_id, "--follow", "--interval-seconds", "1"], env=env)

        assert "site-packages" in imported_from
        assert tmux_policy == {
            "action": "show",
            "scope": "project",
            "section": "tmux",
            "source": "default",
            "applies_to": "new_observer_decisions",
            "values": {
                "enabled": False,
                "source": "default",
                "applies_to": "new_observer_decisions",
            },
            "effective_values": {
                "enabled": False,
                "source": "default",
                "applies_to": "new_observer_decisions",
            },
        }
        assert group["group"]["name"] == "release-e2e"
        assert task["task"]["task_id"] == task_id
        assert task["task"]["group_name"] == "release-e2e"
        assert task["observation"]["tmux_override"] == "disabled"
        assert any(item["task_id"] == task_id for item in tasks)
        assert any(item["group"]["name"] == "release-e2e" for item in groups)
        assert any(item["machine"]["machine_name"] == "gpu-1" for item in machines)
        assert "cli ok" in followed.stdout
    finally:
        stop_agent(common, env=env)

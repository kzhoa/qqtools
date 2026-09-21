from __future__ import annotations

import json

from qexp_e2e import ensure_site_packages_import, make_env, run


def test_installed_cli_persists_machine_gpu_policy_without_project_context(tmp_path) -> None:
    base = tmp_path / "gpu-policy"
    runtime_root = base / "machine-runtime"
    env = make_env(base)
    command = ["qexp", "--machine-runtime-root", str(runtime_root), "agent", "gpus"]

    changed = run(
        [*command, "set", "--visible", "2,0", "--expected-revision", "0", "--format=json"],
        env=env,
    )
    result = json.loads(changed.stdout)
    assert result["current_revision"] == 1
    assert result["configured_gpu_ids"] == [0, 2]

    shown = run([*command, "show", "--format=json"], env=env)
    policy = json.loads(shown.stdout)
    assert policy["revision"] == 1
    assert policy["mode"] == "explicit"
    assert policy["configured_gpu_ids"] == [0, 2]
    assert "site-packages" in ensure_site_packages_import()

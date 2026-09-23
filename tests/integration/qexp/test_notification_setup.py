"""Global notification setup and sparse per-project policy through the public CLI."""

import json
from contextlib import contextmanager

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.notification_policy import load_policy
from qqtools.plugins.qexp.notification_resolver import resolve_policy

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


WEBHOOK = "https://open.feishu.cn/open-apis/bot/v2/hook/global-example"


def _invoke(runtime_root, *arguments):
    return ["--machine-runtime-root", str(runtime_root), *arguments, "--format=json"]


def _result(capsys):
    return json.loads(capsys.readouterr().out)


def test_setup_outside_project_enables_new_and_existing_registered_projects(tmp_path, capsys, monkeypatch):
    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    for name in ("first", "second"):
        assert main(_invoke(root, "project", "init", str(tmp_path / name))) == 0
        _result(capsys)
        assert main(_invoke(root, "project", "register", str(tmp_path / name))) == 0
        _result(capsys)

    assert main(_invoke(root, "notifications", "setup", "--webhook", WEBHOOK)) == 0
    setup = _result(capsys)
    assert setup["scope"] == "global"
    assert WEBHOOK not in repr(setup)
    assert load_policy(root, "global")["override"]["enabled"] is True

    runtime = MachineRuntime(root)
    for binding in runtime.load_registry()[1]:
        assert resolve_policy(root, binding.project_id)["enabled"] is True
        assert resolve_policy(root, binding.project_id)["provenance"]["destination"] == "global"
        assert load_policy(root, "project", binding.project_id)["override"] is None

    assert main(_invoke(root, "project", "init", str(tmp_path / "new"))) == 0
    _result(capsys)
    assert main(_invoke(root, "project", "register", str(tmp_path / "new"))) == 0
    _result(capsys)
    new_binding = runtime.resolve_project_binding(tmp_path / "new" / ".qexp")
    assert resolve_policy(root, new_binding.project_id)["enabled"] is True


def test_project_disable_and_reset_do_not_change_global_or_other_project(tmp_path, capsys, monkeypatch):
    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    for name in ("first", "second"):
        assert main(_invoke(root, "project", "init", str(tmp_path / name))) == 0
        _result(capsys)
        assert main(_invoke(root, "project", "register", str(tmp_path / name))) == 0
        _result(capsys)
    assert main(_invoke(root, "notifications", "setup", "--webhook", WEBHOOK)) == 0
    _result(capsys)

    first_root = tmp_path / "first"
    first_id = MachineRuntime(root).resolve_project_binding(first_root / ".qexp").project_id
    assert (
        main(_invoke(root, "notifications", "set", "--scope", "project", "--project", str(first_root), "--disabled"))
        == 0
    )
    _result(capsys)
    assert resolve_policy(root, first_id)["enabled"] is False
    assert load_policy(root, "global")["override"]["enabled"] is True

    assert main(_invoke(root, "notifications", "reset", "--scope", "project", "--project", str(first_root))) == 0
    _result(capsys)
    assert resolve_policy(root, first_id)["enabled"] is True
    assert load_policy(root, "project", first_id)["revision"] == 3


def test_missing_identity_fails_before_staging_or_policy_mutation(tmp_path, capsys, monkeypatch):
    root = tmp_path / "missing-machine"
    monkeypatch.chdir(tmp_path)

    assert main(_invoke(root, "notifications", "setup", "--webhook", WEBHOOK)) != 0
    failure = _result(capsys)["error"]
    assert "init" in str(failure)
    assert not (root / "notifications").exists()
    assert not (root / "identity.json").exists()


def test_global_show_reports_retention_without_writing(tmp_path, capsys, monkeypatch):
    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    assert main(_invoke(root, "notifications", "setup", "--webhook", WEBHOOK)) == 0
    _result(capsys)
    retention_path = root / "notifications" / "retention.json"
    original = retention_path.read_bytes()
    assert main(_invoke(root, "notifications", "show")) == 0
    result = _result(capsys)
    assert result["retention"]["retained"] == 1
    assert result["retention"]["eligible"] == 0
    assert retention_path.read_bytes() == original
    assert main(["--machine-runtime-root", str(root), "notifications", "show"]) == 0
    text = capsys.readouterr().out
    assert "Retention:" in text
    assert "retained" in text


def test_global_show_on_fresh_runtime_does_not_create_policy_storage(tmp_path, capsys, monkeypatch):
    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    assert main(_invoke(root, "notifications", "show")) == 0
    shown = _result(capsys)
    assert shown["effective_values"]["enabled"] is False
    assert shown["retention"]["retained"] == 0
    assert not (root / "notifications").exists()


def test_agent_loop_binds_runtime_for_foreground_and_background_paths(tmp_path, capsys, monkeypatch):
    from qqtools.plugins.qexp.agent.lifecycle import run_machine_agent_loop
    from qqtools.plugins.qexp.notifications import _selected_runtime_root

    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    observed = []

    @contextmanager
    def no_authority(_self, *, blocking):
        observed.append((_selected_runtime_root.get(), blocking))
        yield True

    def check_context(_pid):
        observed.append((_selected_runtime_root.get(), None))
        return None

    monkeypatch.setattr(MachineRuntime, "scheduler_authority", no_authority)
    monkeypatch.setattr("qqtools.plugins.qexp.agent.lifecycle._pid_start_time_ticks", check_context)
    with pytest.raises(RuntimeError, match="process identity"):
        run_machine_agent_loop(root)
    assert observed == [(None, False), (root, None)]
    assert _selected_runtime_root.get() is None


def test_test_command_reports_cli_sender_and_does_not_claim_task_delivery(tmp_path, capsys, monkeypatch):
    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    assert main(_invoke(root, "notifications", "setup", "--webhook", WEBHOOK)) == 0
    _result(capsys)
    sent = []

    def fake_send(_self, event, *, webhook, secret, timeout_seconds):
        sent.append((event.phase, webhook, secret, timeout_seconds))
        return {"http_status": 200, "business_code": "0"}

    monkeypatch.setattr("qqtools.plugins.qexp.commands.notifications.FeishuNotifier.send", fake_send)
    assert main(_invoke(root, "notifications", "test")) == 0
    result = _result(capsys)
    assert result["scope"] == "global"
    assert result["tested_by"] == "cli"
    assert result["revision"] == 1
    assert sent == [("test", WEBHOOK, None, 5)]
    assert not (root / "notifications" / "claims").exists()


def test_malformed_identity_never_triggers_implicit_replacement(tmp_path, capsys, monkeypatch):
    root = tmp_path / "machine"
    monkeypatch.chdir(tmp_path)
    assert main(_invoke(root, "init", "--machine", "g1")) == 0
    _result(capsys)
    (root / "identity.json").write_text("not-json")

    assert main(_invoke(root, "notifications", "setup", "--webhook", WEBHOOK)) != 0
    failure = _result(capsys)["error"]
    assert "identity" in str(failure)
    assert "malformed" in str(failure)
    assert not (root / "notifications").exists()

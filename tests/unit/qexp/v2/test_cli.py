from __future__ import annotations

import json
import os

import pytest
import yaml

from qqtools.plugins.qexp.v2.layout import init_shared_root


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1")


def _base_args(cfg):
    return ["--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name]


# ---------------------------------------------------------------------------
# V2 CLI direct tests
# ---------------------------------------------------------------------------


class TestV2Init:
    def test_init(self, tmp_path, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main([
            "--shared-root", str(tmp_path / "new_shared"),
            "--machine", "test-m",
            "init",
        ])
        assert ret == 0
        out = capsys.readouterr().out
        assert "test-m" in out


class TestV2Submit:
    def test_submit(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["submit", "--", "echo", "hello"])
        assert ret == 0
        task_id = capsys.readouterr().out.strip()
        assert len(task_id) > 0

    def test_submit_with_name(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + [
            "submit", "--name", "test-run", "--", "echo"
        ])
        assert ret == 0

    def test_submit_no_command_fails(self, cfg):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        with pytest.raises((ValueError, SystemExit)):
            v2_main(_base_args(cfg) + ["submit"])


class TestV2Cancel:
    def test_cancel(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        v2_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        ret = v2_main(_base_args(cfg) + ["cancel", "t1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "cancelled" in out


class TestV2Retry:
    def test_retry(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        from qqtools.plugins.qexp.v2.api import cancel
        v2_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        cancel(cfg, "t1")
        ret = v2_main(_base_args(cfg) + ["retry", "t1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "retry of t1" in out


class TestV2List:
    def test_list_empty(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["list"])
        assert ret == 0
        assert "No tasks" in capsys.readouterr().out

    def test_list_with_tasks(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        v2_main(_base_args(cfg) + ["submit", "--", "echo"])
        capsys.readouterr()
        ret = v2_main(_base_args(cfg) + ["list"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "queued" in out


class TestV2Inspect:
    def test_inspect(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        v2_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        ret = v2_main(_base_args(cfg) + ["inspect", "t1"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["task"]["task_id"] == "t1"


class TestV2BatchSubmit:
    def test_batch_submit(self, cfg, tmp_path, capsys):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "sweep"},
            "tasks": [
                {"command": ["echo", "1"]},
                {"command": ["echo", "2"]},
            ],
        }), encoding="utf-8")
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["batch-submit", "--file", str(manifest)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "tasks=2" in out


class TestV2Machines:
    def test_machines(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["machines"])
        assert ret == 0
        assert "dev1" in capsys.readouterr().out


class TestV2Doctor:
    def test_verify(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["doctor", "verify"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True

    def test_rebuild_index(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["doctor", "rebuild-index"])
        assert ret == 0


# ---------------------------------------------------------------------------
# V1 routing tests
# ---------------------------------------------------------------------------


class TestV2Logs:
    def test_logs(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        from qqtools.plugins.qexp.v2.layout import runtime_log_path

        v2_main(_base_args(cfg) + ["submit", "--task-id", "log1", "--", "echo"])
        capsys.readouterr()

        # Create a log file
        log_path = runtime_log_path(cfg, "log1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("hello from log\n", encoding="utf-8")

        ret = v2_main(_base_args(cfg) + ["logs", "log1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "hello from log" in out

    def test_logs_missing(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        v2_main(_base_args(cfg) + ["submit", "--task-id", "nolog", "--", "echo"])
        capsys.readouterr()

        with pytest.raises(FileNotFoundError):
            v2_main(_base_args(cfg) + ["logs", "nolog"])


class TestV2Clean:
    def test_clean_dry_run(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main(_base_args(cfg) + ["clean", "--dry-run"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Dry run" in out

    def test_clean(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main(_base_args(cfg) + ["clean"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Cleaned" in out


class TestRouting:
    def test_default_routes_to_v2(self, cfg, capsys, monkeypatch):
        """v2 is the default since 1.2.7."""
        monkeypatch.delenv("QEXP_VERSION", raising=False)
        monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.shared_root))
        monkeypatch.setenv("QEXP_MACHINE", cfg.machine_name)
        from qqtools.plugins.qexp.cli import main
        ret = main(["machines"])
        assert ret == 0
        assert "dev1" in capsys.readouterr().out

    def test_v1_env_fallback(self, monkeypatch):
        monkeypatch.setenv("QEXP_VERSION", "1")
        from qqtools.plugins.qexp.cli import _should_use_v1
        assert _should_use_v1(None)

    def test_v1_flag_fallback(self, monkeypatch):
        monkeypatch.delenv("QEXP_VERSION", raising=False)
        from qqtools.plugins.qexp.cli import _should_use_v1
        assert _should_use_v1(["--v1", "submit", "--", "echo"])

    def test_v1_flag_after_separator_ignored(self, monkeypatch):
        """--v1 after '--' is user payload, not a routing flag."""
        monkeypatch.delenv("QEXP_VERSION", raising=False)
        from qqtools.plugins.qexp.cli import _should_use_v1
        assert not _should_use_v1(["submit", "--", "python", "--v1"])

    def test_v1_flag_stripped_preserves_payload(self):
        from qqtools.plugins.qexp.cli import _strip_v1_flag
        result = _strip_v1_flag(["--v1", "submit", "--", "python", "--v1", "arg"])
        assert result == ["submit", "--", "python", "--v1", "arg"]

    def test_should_use_v2_compat(self, monkeypatch):
        """_should_use_v2 is now the inverse of _should_use_v1."""
        monkeypatch.delenv("QEXP_VERSION", raising=False)
        from qqtools.plugins.qexp.cli import _should_use_v2
        assert _should_use_v2(["submit", "--", "echo"])

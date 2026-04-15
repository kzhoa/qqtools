from __future__ import annotations

import json
import os

import pytest
import yaml

from qqtools.plugins.qexp.v2.layout import init_shared_root


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / "shared", "dev1", runtime_root=tmp_path / "runtime")


def _base_args(cfg):
    return ["--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name]


# ---------------------------------------------------------------------------
# V2 CLI direct tests
# ---------------------------------------------------------------------------


class TestV2Init:
    def test_init(self, tmp_path, monkeypatch, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        import qqtools.plugins.qexp.v2.layout as _layout

        monkeypatch.setattr(_layout, "_context_file_override", str(tmp_path / "context.json"))
        ret = v2_main([
            "--shared-root", str(tmp_path / "new_shared"),
            "--machine", "test-m",
            "--runtime-root", str(tmp_path / "runtime"),
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

    def test_repair(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main
        ret = v2_main(_base_args(cfg) + ["doctor", "repair"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "repaired_batch_count" in out


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

    def test_clean_rejects_incompatible_single_task_flags(self, cfg, capsys):
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main(_base_args(cfg) + ["clean", "--task-id", "t1", "--include-failed"])
        assert ret == 1
        assert "--task-id cannot be combined" in capsys.readouterr().err


class TestV2Use:
    """Tests for the 'use' command and context persistence."""

    def _patch_context(self, monkeypatch, tmp_path):
        ctx_file = str(tmp_path / "qexp-context.json")
        import qqtools.plugins.qexp.v2.layout as _layout
        monkeypatch.setattr(_layout, "_context_file_override", ctx_file)
        monkeypatch.delenv("QEXP_SHARED_ROOT", raising=False)
        monkeypatch.delenv("QEXP_MACHINE", raising=False)
        monkeypatch.delenv("QEXP_RUNTIME_ROOT", raising=False)
        return ctx_file

    def test_use_set_and_resolve(self, cfg, tmp_path, monkeypatch, capsys):
        """After 'use', commands work without flags."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main([
            "use",
            "--shared-root", str(cfg.shared_root),
            "--machine", cfg.machine_name,
        ])
        assert ret == 0
        capsys.readouterr()

        ret = v2_main(["list"])
        assert ret == 0

    def test_use_show(self, cfg, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        v2_main([
            "use",
            "--shared-root", str(cfg.shared_root),
            "--machine", cfg.machine_name,
        ])
        capsys.readouterr()

        ret = v2_main(["use", "--show"])
        assert ret == 0
        out = capsys.readouterr().out
        assert cfg.machine_name in out
        assert str(cfg.shared_root) in out

    def test_use_show_empty(self, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main(["use", "--show"])
        assert ret == 0
        assert "No context" in capsys.readouterr().out

    def test_use_clear(self, cfg, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        v2_main([
            "use",
            "--shared-root", str(cfg.shared_root),
            "--machine", cfg.machine_name,
        ])
        capsys.readouterr()

        ret = v2_main(["use", "--clear"])
        assert ret == 0
        assert "cleared" in capsys.readouterr().out.lower()

        with pytest.raises(SystemExit):
            v2_main(["list"])

    def test_use_requires_shared_root_and_machine(self, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main(["use", "--shared-root", "/tmp/x"])
        assert ret == 1

    def test_init_saves_context(self, tmp_path, monkeypatch, capsys):
        """init should auto-save context so subsequent commands work."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        ret = v2_main([
            "--shared-root", str(tmp_path / "shared"),
            "--machine", "auto-ctx",
            "init",
        ])
        assert ret == 0
        assert "context saved" in capsys.readouterr().out.lower()

        ret = v2_main(["list"])
        assert ret == 0

    def test_cli_flags_override_context(self, tmp_path, monkeypatch, capsys):
        """CLI flags take precedence over saved context."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        # Save context pointing to one machine
        cfg1 = init_shared_root(tmp_path / "s1", "m1")
        v2_main([
            "use",
            "--shared-root", str(cfg1.shared_root),
            "--machine", "m1",
        ])
        capsys.readouterr()

        # Init a second machine
        cfg2 = init_shared_root(tmp_path / "s2", "m2")

        # Explicit flags should override saved context
        ret = v2_main([
            "--shared-root", str(cfg2.shared_root),
            "--machine", "m2",
            "machines",
        ])
        assert ret == 0
        out = capsys.readouterr().out
        assert "m2" in out

    def test_env_overrides_context(self, tmp_path, monkeypatch, capsys):
        """Environment variables take precedence over saved context."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.v2.cli import main as v2_main

        # Save context pointing to one machine
        cfg1 = init_shared_root(tmp_path / "s1", "m1")
        v2_main([
            "use",
            "--shared-root", str(cfg1.shared_root),
            "--machine", "m1",
        ])
        capsys.readouterr()

        # Init a second machine and set env
        cfg2 = init_shared_root(tmp_path / "s2", "m2")
        monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg2.shared_root))
        monkeypatch.setenv("QEXP_MACHINE", "m2")

        ret = v2_main(["machines"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "m2" in out


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

from __future__ import annotations

import json
import os

import pytest
import yaml

from qqtools.plugins.qexp.layout import init_shared_root


@pytest.fixture()
def cfg(tmp_path):
    return init_shared_root(tmp_path / ".qexp", "dev1", runtime_root=tmp_path / "runtime")


def _base_args(cfg):
    return ["--shared-root", str(cfg.shared_root), "--machine", cfg.machine_name]


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestInit:
    def test_init(self, tmp_path, monkeypatch, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        import qqtools.plugins.qexp.layout as _layout

        monkeypatch.setattr(_layout, "_context_file_override", str(tmp_path / "context.json"))
        ret = cli_main([
            "--shared-root", str(tmp_path / ".qexp"),
            "--machine", "test-m",
            "--runtime-root", str(tmp_path / "runtime"),
            "init",
        ])
        assert ret == 0
        out = capsys.readouterr().out
        assert "test-m" in out


class TestSubmit:
    def test_submit_help_explains_group_vs_batch(self, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main

        with pytest.raises(SystemExit) as exc:
            cli_main(["submit", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "group is a long-lived grouping key" in out
        assert "group is not batch" in out
        assert "tmux session" in out

    def test_submit(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["submit", "--", "echo", "hello"])
        assert ret == 0
        task_id = capsys.readouterr().out.strip()
        assert len(task_id) > 0

    def test_submit_with_name(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + [
            "submit", "--name", "test-run", "--", "echo"
        ])
        assert ret == 0

    def test_submit_with_group(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + [
            "submit", "--group", "contract_n_4and6", "--", "echo"
        ])
        assert ret == 0

    def test_submit_no_command_fails(self, cfg):
        from qqtools.plugins.qexp.cli import main as cli_main
        with pytest.raises((ValueError, SystemExit)):
            cli_main(_base_args(cfg) + ["submit"])


class TestCancel:
    def test_cancel(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        cli_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        ret = cli_main(_base_args(cfg) + ["cancel", "t1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "cancelled" in out


class TestRetry:
    def test_retry(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import cancel
        cli_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        cancel(cfg, "t1")
        ret = cli_main(_base_args(cfg) + ["retry", "t1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "retry of t1" in out

    def test_retry_with_group_override(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import cancel

        cli_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--group", "contract_n_4and6", "--", "echo"])
        capsys.readouterr()
        cancel(cfg, "t1")
        ret = cli_main(_base_args(cfg) + ["retry", "t1", "--group", "regrouped_debug"])
        assert ret == 0


class TestResubmit:
    def test_resubmit(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import cancel

        cli_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        cancel(cfg, "t1")

        ret = cli_main(_base_args(cfg) + ["resubmit", "t1", "--", "echo", "again"])
        assert ret == 0
        assert capsys.readouterr().out.strip() == "t1"

    def test_inspect_shows_pending_resubmit_operation(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import submit, _build_resubmit_operation, _advance_resubmit_operation, _delete_task_truth
        from qqtools.plugins.qexp.indexes import update_index_on_phase_change
        from qqtools.plugins.qexp.storage import cas_update_task, load_task, save_resubmit_operation

        submit(cfg, command=["echo"], task_id="pending1")
        task = load_task(cfg, "pending1")
        task.status.phase = "failed"
        task.timestamps.finished_at = "2026-01-01T00:00:00Z"
        cas_update_task(cfg, task, task.meta.revision)
        update_index_on_phase_change(cfg, "pending1", "queued", "failed")

        operation = _build_resubmit_operation(
            cfg,
            task,
            command=["echo", "new"],
            requested_gpus=1,
            name=task.name,
            group=task.group,
        )
        save_resubmit_operation(cfg, operation)
        _advance_resubmit_operation(cfg, operation, "deleting_old")
        _delete_task_truth(cfg, task)

        ret = cli_main(_base_args(cfg) + ["inspect", "pending1"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["task"] is None
        assert data["operation"]["state"] == "deleting_old"


class TestList:
    def test_list_empty(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["list"])
        assert ret == 0
        assert "No tasks" in capsys.readouterr().out

    def test_list_with_tasks(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        cli_main(_base_args(cfg) + ["submit", "--", "echo"])
        capsys.readouterr()
        ret = cli_main(_base_args(cfg) + ["list"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "queued" in out

    def test_list_filter_by_group(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        cli_main(_base_args(cfg) + ["submit", "--group", "contract_n_4and6", "--", "echo"])
        capsys.readouterr()
        ret = cli_main(_base_args(cfg) + ["list", "--group", "contract_n_4and6"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "contract_n_4and6" in out


class TestInspect:
    def test_inspect(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        cli_main(_base_args(cfg) + ["submit", "--task-id", "t1", "--", "echo"])
        capsys.readouterr()
        ret = cli_main(_base_args(cfg) + ["inspect", "t1"])
        assert ret == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["task"]["task_id"] == "t1"


class TestBatchSubmit:
    def test_batch_submit_help_explains_batch_vs_group(self, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main

        with pytest.raises(SystemExit) as exc:
            cli_main(["batch-submit", "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "batch = one bulk submit operation" in out
        assert "batch.group = the default" in out
        assert "long-lived" in out
        assert "for tasks in this batch" in out
        assert "another related set tomorrow" in out
        assert "reuse the same" in out

    def test_batch_submit(self, cfg, tmp_path, capsys):
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.dump({
            "batch": {"name": "sweep", "group": "contract_n_4and6"},
            "tasks": [
                {"command": ["echo", "1"]},
                {"command": ["echo", "2"]},
            ],
        }), encoding="utf-8")
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["batch-submit", "--file", str(manifest)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "tasks=2" in out

    def test_batches_hides_non_committed_batches(self, cfg, tmp_path, capsys):
        from qqtools.plugins.qexp.api import batch_submit
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.models import BATCH_COMMIT_PREPARING
        from qqtools.plugins.qexp.storage import save_batch

        manifest = tmp_path / "hidden-batch.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo", "1"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        batch.commit_state = BATCH_COMMIT_PREPARING
        save_batch(cfg, batch)

        ret = cli_main(_base_args(cfg) + ["batches"])
        assert ret == 0
        assert "No batches found." in capsys.readouterr().out

    def test_batch_inspect_raises_for_non_committed_batch(self, cfg, tmp_path, capsys):
        from qqtools.plugins.qexp.api import batch_submit
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.models import BATCH_COMMIT_PREPARING
        from qqtools.plugins.qexp.storage import save_batch

        manifest = tmp_path / "hidden-inspect.yaml"
        manifest.write_text(yaml.dump({
            "tasks": [{"command": ["echo", "1"]}],
        }), encoding="utf-8")
        batch = batch_submit(cfg, manifest)
        batch.commit_state = BATCH_COMMIT_PREPARING
        save_batch(cfg, batch)

        with pytest.raises(FileNotFoundError, match="not committed"):
            cli_main(_base_args(cfg) + ["batch", batch.batch_id])


class TestMachines:
    def test_machines(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["machines"])
        assert ret == 0
        assert "dev1" in capsys.readouterr().out

    def test_machines_prints_gpu_summary_columns(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.storage import write_atomic_json

        write_atomic_json(gpu_state_path(cfg), {
            "gpu_count": 4,
            "visible_gpu_ids": [0, 1, 2, 3],
            "reserved_gpu_ids": [1, 2],
            "task_to_gpu_ids": {"t1": [1], "t2": [2]},
            "backend": "stub",
            "probe_succeeded": True,
            "probe_error": None,
            "updated_at": "2099-01-01T00:00:00Z",
        })

        ret = cli_main(_base_args(cfg) + ["machines"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "gpu_status=live" in out
        assert "visible=4" in out
        assert "reserved=2" in out
        assert "free=2" in out

    def test_machines_prints_cross_machine_gpu_summary(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.agent import start_agent_record
        from qqtools.plugins.qexp.layout import gpu_state_path
        from qqtools.plugins.qexp.storage import write_atomic_json

        other_cfg = init_shared_root(
            cfg.shared_root,
            "gpu2",
            runtime_root=cfg.runtime_root.parent / "runtime2",
        )
        start_agent_record(other_cfg)
        write_atomic_json(gpu_state_path(other_cfg), {
            "gpu_count": 6,
            "visible_gpu_ids": [0, 1, 2, 3, 4, 5],
            "reserved_gpu_ids": [0, 5],
            "task_to_gpu_ids": {"t1": [0], "t2": [5]},
            "backend": "stub",
            "probe_succeeded": True,
            "probe_error": None,
            "updated_at": "2099-01-01T00:00:00Z",
        })

        ret = cli_main(_base_args(cfg) + ["machines"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "gpu2" in out
        assert "state=starting" in out
        assert "visible=6" in out
        assert "reserved=2" in out
        assert "free=4" in out


class TestDoctor:
    def test_verify(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["doctor", "verify"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["ok"] is True
        assert data["severity"] == "ok"
        assert data["recommended_actions"] == []
        assert data["issues"] == []

    def test_verify_non_strict_does_not_fail_on_detected_issues(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import submit
        from qqtools.plugins.qexp.layout import task_path

        task = submit(cfg, command=["echo"], task_id="verify-broken")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")

        ret = cli_main(_base_args(cfg) + ["doctor", "verify"])
        assert ret == 0
        data = json.loads(capsys.readouterr().out)
        assert data["severity"] == "high"
        assert any(issue["code"] == "task_truth_unreadable" for issue in data["issues"])
        assert any(
            action["action_code"] == "manual_fix_truth_corruption"
            and action["blocking"] is True
            for action in data["recommended_actions"]
        )

    def test_verify_strict_fails_on_any_governed_issue(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import submit
        from qqtools.plugins.qexp.layout import task_path

        task = submit(cfg, command=["echo"], task_id="verify-strict")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")

        ret = cli_main(_base_args(cfg) + ["doctor", "verify", "--strict"])
        assert ret == 2
        data = json.loads(capsys.readouterr().out)
        assert data["severity"] == "high"
        assert data["issue_count_by_category"]["task_truth"] == 1

    def test_verify_fail_on_threshold(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import submit
        from qqtools.plugins.qexp.layout import index_by_state_dir
        from qqtools.plugins.qexp.storage import write_atomic_json

        submit(cfg, command=["echo"], task_id="verify-threshold")
        write_atomic_json(index_by_state_dir(cfg) / "queued.json", {"task_ids": []})

        ret = cli_main(_base_args(cfg) + ["doctor", "verify", "--fail-on", "high"])
        assert ret == 2
        data = json.loads(capsys.readouterr().out)
        assert data["severity"] == "high"
        assert data["issue_count_by_code"]["derived_index_state_drift"] == 1
        assert any(
            action["action_code"] == "run_doctor_rebuild_index_state"
            and action["blocking"] is True
            for action in data["recommended_actions"]
        )

    def test_verify_jsonl_outputs_machine_records(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.api import submit
        from qqtools.plugins.qexp.layout import task_path

        task = submit(cfg, command=["echo"], task_id="verify-jsonl")
        task_path(cfg, task.task_id).write_text("not json", encoding="utf-8")

        ret = cli_main(_base_args(cfg) + ["doctor", "verify", "--jsonl", "--strict"])
        assert ret == 2
        lines = [line for line in capsys.readouterr().out.strip().splitlines() if line]
        records = [json.loads(line) for line in lines]
        assert records[0]["type"] == "verify_summary"
        assert records[0]["exit_code"] == 2
        issue_records = [record for record in records if record["type"] == "verify_issue"]
        assert issue_records
        assert issue_records[0]["issue_code"] == "task_truth_unreadable"
        assert issue_records[0]["category"] == "task_truth"
        recommendation_records = [
            record for record in records if record["type"] == "verify_recommendation"
        ]
        assert recommendation_records
        assert any(
            record["action_code"] == "manual_fix_truth_corruption"
            and record["blocking"] is True
            for record in recommendation_records
        )
        assert records[-1]["type"] == "verify_result"
        assert records[-1]["fail_on"] == "low"

    def test_rebuild_index(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["doctor", "rebuild-index"])
        assert ret == 0

    def test_repair(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        ret = cli_main(_base_args(cfg) + ["doctor", "repair"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "repaired_batch_count" in out


class TestLogs:
    def test_logs(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main
        from qqtools.plugins.qexp.layout import runtime_log_path

        cli_main(_base_args(cfg) + ["submit", "--task-id", "log1", "--", "echo"])
        capsys.readouterr()

        # Create a log file
        log_path = runtime_log_path(cfg, "log1")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("hello from log\n", encoding="utf-8")

        ret = cli_main(_base_args(cfg) + ["logs", "log1"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "hello from log" in out

    def test_logs_missing(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main

        cli_main(_base_args(cfg) + ["submit", "--task-id", "nolog", "--", "echo"])
        capsys.readouterr()

        with pytest.raises(FileNotFoundError):
            cli_main(_base_args(cfg) + ["logs", "nolog"])


class TestClean:
    def test_clean_dry_run(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main(_base_args(cfg) + ["clean", "--dry-run"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Dry run" in out

    def test_clean(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main(_base_args(cfg) + ["clean"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Cleaned" in out

    def test_clean_rejects_incompatible_single_task_flags(self, cfg, capsys):
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main(_base_args(cfg) + ["clean", "--task-id", "t1", "--include-failed"])
        assert ret == 1
        assert "--task-id cannot be combined" in capsys.readouterr().err


class TestUse:
    """Tests for the 'use' command and context persistence."""

    def _patch_context(self, monkeypatch, tmp_path):
        ctx_file = str(tmp_path / "qexp-context.json")
        import qqtools.plugins.qexp.layout as _layout
        monkeypatch.setattr(_layout, "_context_file_override", ctx_file)
        monkeypatch.delenv("QEXP_SHARED_ROOT", raising=False)
        monkeypatch.delenv("QEXP_MACHINE", raising=False)
        monkeypatch.delenv("QEXP_RUNTIME_ROOT", raising=False)
        return ctx_file

    def test_use_set_and_resolve(self, cfg, tmp_path, monkeypatch, capsys):
        """After 'use', commands work without flags."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main([
            "use",
            "--shared-root", str(cfg.shared_root),
            "--machine", cfg.machine_name,
        ])
        assert ret == 0
        capsys.readouterr()

        ret = cli_main(["list"])
        assert ret == 0

    def test_use_show(self, cfg, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        cli_main([
            "use",
            "--shared-root", str(cfg.shared_root),
            "--machine", cfg.machine_name,
        ])
        capsys.readouterr()

        ret = cli_main(["use", "--show"])
        assert ret == 0
        out = capsys.readouterr().out
        assert cfg.machine_name in out
        assert str(cfg.shared_root) in out

    def test_use_show_empty(self, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main(["use", "--show"])
        assert ret == 0
        assert "No context" in capsys.readouterr().out

    def test_use_clear(self, cfg, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        cli_main([
            "use",
            "--shared-root", str(cfg.shared_root),
            "--machine", cfg.machine_name,
        ])
        capsys.readouterr()

        ret = cli_main(["use", "--clear"])
        assert ret == 0
        assert "cleared" in capsys.readouterr().out.lower()

        with pytest.raises(SystemExit):
            cli_main(["list"])

    def test_use_requires_shared_root_and_machine(self, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main(["use", "--shared-root", "/tmp/x"])
        assert ret == 1

    def test_use_rejects_uninitialized_root(self, tmp_path, monkeypatch, capsys):
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main([
            "use",
            "--shared-root", str(tmp_path / ".qexp"),
            "--machine", "m1",
        ])
        assert ret == 1
        assert "cannot save qexp context" in capsys.readouterr().err

    def test_init_saves_context(self, tmp_path, monkeypatch, capsys):
        """init should auto-save context so subsequent commands work."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        ret = cli_main([
            "--shared-root", str(tmp_path / ".qexp"),
            "--machine", "auto-ctx",
            "init",
        ])
        assert ret == 0
        assert "context saved" in capsys.readouterr().out.lower()

        ret = cli_main(["list"])
        assert ret == 0

    def test_cli_flags_override_context(self, tmp_path, monkeypatch, capsys):
        """CLI flags take precedence over saved context."""
        self._patch_context(monkeypatch, tmp_path)
        from qqtools.plugins.qexp.cli import main as cli_main

        # Save context pointing to one machine
        cfg1 = init_shared_root(tmp_path / "p1" / ".qexp", "m1")
        cli_main([
            "use",
            "--shared-root", str(cfg1.shared_root),
            "--machine", "m1",
        ])
        capsys.readouterr()

        # Init a second machine
        cfg2 = init_shared_root(tmp_path / "p2" / ".qexp", "m2")

        # Explicit flags should override saved context
        ret = cli_main([
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
        from qqtools.plugins.qexp.cli import main as cli_main

        # Save context pointing to one machine
        cfg1 = init_shared_root(tmp_path / "p1" / ".qexp", "m1")
        cli_main([
            "use",
            "--shared-root", str(cfg1.shared_root),
            "--machine", "m1",
        ])
        capsys.readouterr()

        # Init a second machine and set env
        cfg2 = init_shared_root(tmp_path / "p2" / ".qexp", "m2")
        monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg2.shared_root))
        monkeypatch.setenv("QEXP_MACHINE", "m2")

        ret = cli_main(["machines"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "m2" in out


class TestRouting:
    def test_root_cli_routes_to_v2(self, cfg, capsys, monkeypatch):
        monkeypatch.setenv("QEXP_SHARED_ROOT", str(cfg.shared_root))
        monkeypatch.setenv("QEXP_MACHINE", cfg.machine_name)
        from qqtools.plugins.qexp.cli import main

        ret = main(["machines"])
        assert ret == 0
        assert "dev1" in capsys.readouterr().out

    def test_daemon_command_is_rejected(self, cfg):
        from qqtools.plugins.qexp.cli import main as cli_main

        with pytest.raises(SystemExit):
            cli_main(_base_args(cfg) + ["daemon", "status"])

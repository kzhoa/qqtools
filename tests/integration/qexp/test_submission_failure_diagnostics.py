from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.commands.group import create_group
from qqtools.plugins.qexp.runtime.paths import shared_paths
from qqtools.plugins.qexp.runtime.ready.group_member_diagnostics import (
    CHECK_REGISTRY,
    PublicationTracker,
    parse_degraded_reason,
)
from qqtools.plugins.qexp.runtime.store import JSONRecordSizeError, read_json

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _base_args(cfg) -> list[str]:
    machine_runtime_root = cfg.runtime_root.parent / "machine-runtime"
    MachineRuntime(machine_runtime_root).ensure_binding(cfg.shared_root, cfg.machine_name)
    return [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "--machine-runtime-root",
        str(machine_runtime_root),
    ]


def _manifest(path: Path) -> Path:
    path.write_text(
        "tasks:\n"
        "  - task_id: diagnostic-first\n"
        "    command: [echo, first]\n"
        "  - task_id: diagnostic-second\n"
        "    command: [echo, SECRET_COMMAND_SENTINEL]\n",
        encoding="utf-8",
    )
    return path


def _fail_second_member_partition(monkeypatch: pytest.MonkeyPatch) -> None:
    from qqtools.plugins.qexp.runtime.ready import group_members

    original = group_members.atomic_replace
    partition_writes = 0

    def fail(path, value, **kwargs):
        nonlocal partition_writes
        if Path(path).parent.name == "partitions":
            partition_writes += 1
            if partition_writes == 2:
                raise OSError(errno.EIO, "SECRET_EXCEPTION_SENTINEL /private/path")
        return original(path, value, **kwargs)

    monkeypatch.setattr(group_members, "atomic_replace", fail)


def _operation(cfg) -> dict[str, object]:
    paths = sorted(shared_paths(cfg.shared_root)["submissions"].glob("*.json"))
    assert len(paths) == 1
    return read_json(paths[0])["submission"]


def _assert_safe_diagnostic(diagnostic: dict[str, object]) -> None:
    assert diagnostic["version"] == 1
    assert diagnostic["component"] == "group_ready_members"
    assert diagnostic["operation"] == "publish"
    assert diagnostic["stage"] == "member_page_write"
    assert diagnostic["check_id"].startswith("write.member_page.")
    assert diagnostic["reason_code"] == "storage_write_failure"
    assert diagnostic["exception_type"] == "OSError"
    assert diagnostic["task_id"] == "diagnostic-second"
    assert diagnostic["generation"] == 1
    assert diagnostic["group_name"] == "diagnostic-group"
    assert diagnostic["input_index"] == 1
    assert diagnostic["errno"] == errno.EIO
    assert diagnostic["json_line"] is None
    assert diagnostic["json_column"] is None
    encoded = json.dumps(diagnostic, sort_keys=True)
    assert "SECRET_EXCEPTION_SENTINEL" not in encoded
    assert "SECRET_COMMAND_SENTINEL" not in encoded
    assert "/private/path" not in encoded


def test_nonfirst_publication_failure_is_durable_bounded_and_json_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / "runs.yaml")
    _fail_second_member_partition(monkeypatch)

    exit_code = main(
        [
            *_base_args(cfg),
            "submit",
            "--file",
            str(manifest),
            "--group",
            "diagnostic-group",
            "--idempotency-key",
            "diagnostic-key",
            "--format=json",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "submission_aborted"
    _assert_safe_diagnostic(payload["error"]["diagnostic"])
    assert captured.out.count("\n") == 1
    assert "Traceback" not in captured.err

    operation = _operation(cfg)
    assert operation["state"] == "aborted"
    assert "ready-member publication failed" in operation["failure_reason"]
    _assert_safe_diagnostic(operation["failure_diagnostic"])

    state = read_json(shared_paths(cfg.shared_root)["ready_group_members"] / "state.json")["group_ready_members"]
    assert state["state"] == "degraded"
    reason = state["degraded_reasons"][-1]
    assert len(reason.encode("utf-8")) <= 512
    parsed = parse_degraded_reason(reason)
    assert parsed["stage"] == operation["failure_diagnostic"]["stage"]
    assert parsed["check_id"] == operation["failure_diagnostic"]["check_id"]
    assert parsed["reason_code"] == operation["failure_diagnostic"]["reason_code"]
    assert "input_index" not in parsed
    assert "SECRET" not in reason

    with pytest.raises(RuntimeError, match="projection is degraded"):
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                "diagnostic-key",
                "--format=json",
            ]
        )
    capsys.readouterr()
    assert _operation(cfg)["failure_diagnostic"] == operation["failure_diagnostic"]


@pytest.mark.parametrize("mode", ["human", "quiet"])
def test_human_and_quiet_failures_keep_stdout_contract_and_render_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, mode: str
) -> None:
    cfg = init_shared_root(tmp_path / mode / ".qexp", "gpu-1", runtime_root=tmp_path / mode / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / mode / "runs.yaml")
    _fail_second_member_partition(monkeypatch)
    mode_args = ["--quiet"] if mode == "quiet" else []

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                f"diagnostic-{mode}",
                *mode_args,
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert any(line.startswith("qexp: ") for line in lines)
    diagnostic_lines = [line for line in lines if line.startswith("Diagnostic: ")]
    assert len(diagnostic_lines) == 1
    line = diagnostic_lines[0]
    assert "stage=member_page_write" in line
    assert "reason=storage_write_failure" in line
    assert "task=diagnostic-second" in line
    assert "input_index=1" in line
    assert "SECRET" not in captured.err
    assert "Traceback" not in captured.err


def test_abort_write_failure_preserves_in_memory_diagnostic_without_claiming_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / "runs.yaml")
    _fail_second_member_partition(monkeypatch)

    from qqtools.plugins.qexp.runtime import submission

    original_writer = submission._submission_atomic_writer

    def fail_abort(path, value, **kwargs):
        if value.get("submission", {}).get("state") == "aborted":
            raise OSError(errno.ENOSPC, "SECRET_ABORT_SENTINEL")
        return original_writer(path, value, **kwargs)

    monkeypatch.setattr(submission, "_submission_atomic_writer", fail_abort)

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                "abort-write-failure",
                "--format=json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    _assert_safe_diagnostic(payload["error"]["diagnostic"])
    assert payload["operation"]["state"] != "aborted"
    assert payload["error"]["code"] != "submission_aborted"
    assert "SECRET_ABORT_SENTINEL" not in captured.out
    operation = _operation(cfg)
    assert operation["state"] != "aborted"
    assert operation["failure_diagnostic"] is None


@pytest.mark.parametrize(
    ("record_type", "check_id", "reason_code"),
    [
        ("member_page", "write.member_page.size", "member_page_too_large"),
        ("member_catalog", "write.member_catalog.size", "member_catalog_too_large"),
        ("member_header", "write.member_header.size", "member_header_too_large"),
    ],
)
def test_post_locator_size_failures_keep_distinct_entry_validation_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    record_type: str,
    check_id: str,
    reason_code: str,
) -> None:
    cfg = init_shared_root(tmp_path / record_type / ".qexp", "gpu-1", runtime_root=tmp_path / record_type / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / record_type / "runs.yaml")

    from qqtools.plugins.qexp.runtime.ready import group_members

    original = group_members.require_json_size

    def reject_selected(value, *, max_bytes, record_type: str):
        if record_type == test_record_type:
            raise JSONRecordSizeError(
                "SECRET_SIZE_SENTINEL",
                record_type=record_type,
                actual_bytes=max_bytes + 1,
                limit_bytes=max_bytes,
            )
        return original(value, max_bytes=max_bytes, record_type=record_type)

    test_record_type = record_type
    monkeypatch.setattr(group_members, "require_json_size", reject_selected)

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                f"size-{record_type}",
                "--format=json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    diagnostic = json.loads(captured.out)["error"]["diagnostic"]
    assert diagnostic["stage"] == "entry_validate"
    assert diagnostic["check_id"] == check_id
    assert diagnostic["reason_code"] == reason_code
    assert diagnostic["facts"] == {
        "actual_bytes": diagnostic["facts"]["limit_bytes"] + 1,
        "limit_bytes": diagnostic["facts"]["limit_bytes"],
        "record_type": record_type,
    }
    assert "SECRET_SIZE_SENTINEL" not in captured.out

    group_roots = list(shared_paths(cfg.shared_root)["ready_group_member_groups"].iterdir())
    assert len(group_roots) == 1
    locator_root = group_roots[0] / "locators"
    assert len(list(locator_root.glob("*.json"))) == 1
    assert not list((locator_root.parent / "partitions").glob("*.json"))


def test_diagnostic_encoder_failure_keeps_primary_site_in_operation_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / "runs.yaml")
    _fail_second_member_partition(monkeypatch)

    from qqtools.plugins.qexp.runtime.ready import group_member_diagnostics

    def fail_encoder(_value) -> str:
        raise ValueError("SECRET_ENCODER_FAILURE")

    monkeypatch.setattr(group_member_diagnostics, "_encode_diagnostic", fail_encoder)

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                "encoder-failure",
                "--format=json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    diagnostic = json.loads(captured.out)["error"]["diagnostic"]
    assert diagnostic["stage"] == "member_page_write"
    assert diagnostic["check_id"] == "write.member_page.temp_write"
    assert diagnostic["reason_code"] == "unexpected_failure"
    assert _operation(cfg)["failure_diagnostic"] == diagnostic
    assert "SECRET" not in captured.out


def test_degradation_persistence_failure_does_not_replace_primary_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / "runs.yaml")
    _fail_second_member_partition(monkeypatch)

    from qqtools.plugins.qexp.runtime.ready import group_members

    monkeypatch.setattr(group_members, "mark_group_ready_members_degraded", lambda *_args: False)

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                "degradation-failure",
                "--format=json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    diagnostic = json.loads(captured.out)["error"]["diagnostic"]
    _assert_safe_diagnostic(diagnostic)
    assert _operation(cfg)["failure_diagnostic"] == diagnostic
    state = read_json(shared_paths(cfg.shared_root)["ready_group_members"] / "state.json")
    assert state["group_ready_members"]["state"] == "active"


def test_cleanup_failure_retains_durable_primary_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    cfg = init_shared_root(tmp_path / ".qexp", "gpu-1", runtime_root=tmp_path / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / "runs.yaml")
    _fail_second_member_partition(monkeypatch)

    from qqtools.plugins.qexp.runtime import submission

    def fail_cleanup(*_args) -> None:
        raise OSError(errno.EIO, "SECRET_CLEANUP_FAILURE")

    monkeypatch.setattr(submission, "_cleanup_aborted_submission", fail_cleanup)

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                "cleanup-failure",
                "--format=json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    diagnostic = json.loads(captured.out)["error"]["diagnostic"]
    _assert_safe_diagnostic(diagnostic)
    operation = _operation(cfg)
    assert operation["state"] == "aborted"
    assert operation["failure_diagnostic"] == diagnostic
    assert "SECRET_CLEANUP_FAILURE" not in captured.out


@pytest.mark.parametrize(
    "target_stage",
    [
        "projection_check",
        "group_load",
        "locator_validate",
        "page_select",
        "entry_validate",
        "locator_write",
        "member_page_write",
        "member_catalog_write",
        "member_header_write",
        "global_state_commit",
    ],
)
def test_every_publication_stage_reaches_all_diagnostic_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    target_stage: str,
) -> None:
    cfg = init_shared_root(tmp_path / target_stage / ".qexp", "gpu-1", runtime_root=tmp_path / target_stage / "rt")
    create_group(cfg, "diagnostic-group")
    manifest = _manifest(tmp_path / target_stage / "runs.yaml")
    original_enter = PublicationTracker.enter
    stage_entries = 0

    def fail_at_stage(self: PublicationTracker, check_id: str) -> None:
        nonlocal stage_entries
        original_enter(self, check_id)
        if CHECK_REGISTRY[check_id].stage != target_stage:
            return
        stage_entries += 1
        if target_stage == "projection_check" and stage_entries == 1:
            return
        raise ValueError("SECRET_STAGE_FAILURE")

    monkeypatch.setattr(PublicationTracker, "enter", fail_at_stage)

    assert (
        main(
            [
                *_base_args(cfg),
                "submit",
                "--file",
                str(manifest),
                "--group",
                "diagnostic-group",
                "--idempotency-key",
                f"stage-{target_stage}",
                "--format=json",
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    diagnostic = json.loads(captured.out)["error"]["diagnostic"]
    assert diagnostic["stage"] == target_stage
    assert diagnostic["reason_code"] == "unexpected_failure"
    assert CHECK_REGISTRY[diagnostic["check_id"]].stage == target_stage
    assert _operation(cfg)["failure_diagnostic"] == diagnostic
    state = read_json(shared_paths(cfg.shared_root)["ready_group_members"] / "state.json")
    parsed = parse_degraded_reason(state["group_ready_members"]["degraded_reasons"][-1])
    assert parsed["stage"] == diagnostic["stage"]
    assert parsed["check_id"] == diagnostic["check_id"]
    assert parsed["reason_code"] == diagnostic["reason_code"]
    assert "SECRET" not in captured.out

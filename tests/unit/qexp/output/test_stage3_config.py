from __future__ import annotations

import json
from pathlib import Path

import pytest

from qqtools.plugins.qexp.cli.output import CliOutput, OutputKind, render
from qqtools.plugins.qexp.commands.configuration import _redact_notifications, reset_config, set_config, show_config
from qqtools.plugins.qexp.config_types import RootConfig


def _cfg(tmp_path: Path) -> RootConfig:
    root = tmp_path / ".qexp"
    (root / "locks").mkdir(parents=True)
    return RootConfig(root, tmp_path, "gpu-1", tmp_path / "runtime")


def test_config_show_exposes_provenance_scope_and_application_timing(tmp_path: Path) -> None:
    result = show_config("progress", cfg=_cfg(tmp_path), runtime=object())

    assert result["scope"] == "project"
    assert result["source"] == "default"
    assert result["applies_to"] == "new_launches"
    assert result["effective_values"] == result["values"]

    human = render(CliOutput(OutputKind.CONFIG, result), "human")
    assert "Section: progress" in human
    assert "Scope: project" in human
    assert "Source: default" in human
    assert "Applies to: new_launches" in human
    assert json.loads(render(CliOutput(OutputKind.CONFIG, result), "json")) == result


def test_config_mutation_distinguishes_changed_from_no_change(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)

    changed = set_config("progress", cfg=cfg, runtime=object(), values={"interval_seconds": 60})
    unchanged = set_config("progress", cfg=cfg, runtime=object(), values={"interval_seconds": 60})
    reset = reset_config("progress", cfg=cfg, runtime=object())
    reset_again = reset_config("progress", cfg=cfg, runtime=object())

    assert changed["outcome"] == "updated"
    assert changed["changed_fields"] == ["interval_seconds"]
    assert unchanged["outcome"] == "no_change"
    assert unchanged["changed_fields"] == []
    assert reset["outcome"] == "updated"
    assert reset_again["outcome"] == "no_change"
    assert reset_again["effective_values"]["source"] == "default"


def test_config_no_op_detection_never_bypasses_type_validation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    set_config("tmux", cfg=cfg, runtime=object(), values={"enabled": True})
    set_config(
        "notifications",
        cfg=cfg,
        runtime=object(),
        provider="feishu",
        values={"timeout_seconds": 1},
    )

    with pytest.raises(ValueError, match="boolean"):
        set_config("tmux", cfg=cfg, runtime=object(), values={"enabled": 1})
    with pytest.raises(ValueError, match="integer"):
        set_config("lease", cfg=cfg, runtime=object(), values={"ttl_seconds": 120.0})
    with pytest.raises(ValueError, match="timeout_seconds"):
        set_config(
            "notifications",
            cfg=cfg,
            runtime=object(),
            provider="feishu",
            values={"timeout_seconds": True},
        )


def test_reset_all_notifications_reports_provider_only_mutation(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    set_config(
        "notifications",
        cfg=cfg,
        runtime=object(),
        provider="feishu",
        values={"webhook_env": "OTHER_FEISHU_WEBHOOK"},
    )

    result = reset_config("notifications", cfg=cfg, runtime=object())

    assert result["changed"] is True
    assert result["outcome"] == "updated"
    assert result["changed_fields"] == ["providers"]


def test_agent_config_source_normalizes_internal_provenance(monkeypatch) -> None:
    monkeypatch.setattr(
        "qqtools.plugins.qexp.commands.configuration.agent_config_payload",
        lambda _runtime: {"agent_name": "gpu-1", "agent_mode": "daemon", "provenance": "init_explicit"},
    )

    result = show_config("agent", cfg=None, runtime=object())

    assert result["source"] == "configured"
    assert result["values"]["provenance"] == "init_explicit"


def test_notification_redaction_removes_secret_values_from_all_output() -> None:
    secret = "https://example.invalid/private-token"
    redacted = _redact_notifications(
        {
            "enabled": True,
            "providers": {
                "feishu": {
                    "webhook": secret,
                    "signing_secret": "also-private",
                    "secret_env": "QEXP_FEISHU_SECRET",
                    "enabled": True,
                }
            },
        }
    )
    result = {
        "action": "show",
        "section": "notifications",
        "scope": "project",
        "source": "configured",
        "applies_to": "future_notifications",
        "values": redacted,
        "effective_values": redacted,
    }

    human = render(CliOutput(OutputKind.CONFIG, result), "human")
    structured = render(CliOutput(OutputKind.CONFIG, result), "json")
    assert secret not in human
    assert secret not in structured
    assert "also-private" not in human
    assert "also-private" not in structured
    assert "QEXP_FEISHU_SECRET" in human
    assert "QEXP_FEISHU_SECRET" in structured

"""Opt-in pagination preserves legacy output and gives structured failures."""

import json

import pytest

from qqtools.plugins.qexp import submit
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.runtime.observation.maintenance import request_rebuild
from tests.helpers.qexp_discovery import isolated_group

pytestmark = pytest.mark.integration


def args(cfg):
    return [
        "--project",
        str(cfg.shared_root),
        "--machine",
        cfg.machine_name,
        "--runtime-root",
        str(cfg.runtime_root),
        "task",
        "list",
    ]


def test_cli_legacy_json_and_live_page_continuation(tmp_path, capsys):
    cfg = isolated_group(tmp_path, tail=0)
    for key in ["a", "b", "c"]:
        submit(cfg, ["true"], task_id=key, group="experiment")
    assert main([*args(cfg), "--limit", "-1", "--format", "json"]) == 0
    assert [item["task_id"] for item in json.loads(capsys.readouterr().out)] == ["a", "b"]
    assert main([*args(cfg), "--page-size", "1", "--format=json"]) == 0
    output = capsys.readouterr()
    assert not output.err
    first = json.loads(output.out)
    assert first["consistency"] == "live"
    assert [item["task_id"] for item in first["items"]] == ["a"]
    assert main([*args(cfg), "--cursor", first["next_cursor"], "--format=json"]) == 0
    assert [item["task_id"] for item in json.loads(capsys.readouterr().out)["items"]] == ["b", "c"]
    assert main([*args(cfg), "--page-size", "1"]) == 0
    human = capsys.readouterr().out
    assert "page_full" in human and "--cursor" in human and "task list" in human


def test_exact_name_without_page_options_uses_indexed_page_contract(tmp_path, capsys):
    cfg = isolated_group(tmp_path, tail=0)
    submit(cfg, ["true"], task_id="first", group="experiment", name="same-name")
    submit(cfg, ["true"], task_id="second", group="experiment", name="same-name")
    submit(cfg, ["true"], task_id="other", group="experiment", name="other-name")

    assert main([*args(cfg), "--name", "same-name", "--format=json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert [item["task_id"] for item in output["items"]] == ["first", "second"]
    assert output["consistency"] == "live"


@pytest.mark.parametrize(
    "options",
    [
        ["--page-size", "bad"],
        ["--page-size", "0"],
        ["--page-size", "1001"],
        ["--page-size", "1", "--limit", "50"],
        ["--cursor", "bad", "--limit", "50"],
        ["--page-size", "1", "--unknown"],
        ["--page-size"],
        ["--page-size", "1", "--limit", "bad"],
        ["--cursor", "bad"],
    ],
)
def test_paginated_json_argument_errors_are_structured(tmp_path, capsys, options):
    cfg = isolated_group(tmp_path, tail=0)
    assert main([*args(cfg), "--format=json", *options]) == 2
    output = capsys.readouterr()
    value = json.loads(output.out)
    assert set(value) == {"error"}
    assert value["error"]["code"] in {"invalid_argument", "invalid_cursor"}
    assert value["error"]["message"]


def test_unavailable_index_has_exit_one_not_empty_success(tmp_path, capsys):
    cfg = isolated_group(tmp_path, tail=0)
    request_rebuild(cfg)
    assert main([*args(cfg), "--page-size", "10", "--format=json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] in {"index_unavailable", "index_not_ready"}

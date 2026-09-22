import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from qqtools.plugins.qexp.agent.context import MachineRuntime
from qqtools.plugins.qexp.cli.entrypoint import main
from qqtools.plugins.qexp.layout import load_root_config

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


def _invoke(runtime_root: Path, *args: str) -> list[str]:
    return ["--machine-runtime-root", str(runtime_root), *args]


def _json(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def test_machine_init_is_project_independent_and_same_name_reset_is_fresh(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_root = tmp_path / "machine"

    assert main(_invoke(runtime_root, "init", "--machine", "g1", "--format=json")) == 0
    first = _json(capsys)
    assert first["action"] == "initialized"
    assert first["agent_name"] == "g1"
    assert first["agent_mode"] == "daemon"
    assert first["old_runtime_id"] is None
    assert len(first["new_runtime_id"]) == 64
    assert MachineRuntime(runtime_root).load_registry()[1] == []

    assert main(_invoke(runtime_root, "init", "--machine", "g1", "--format=json")) == 1
    duplicate = _json(capsys)["error"]
    assert duplicate["code"] == "operational_failure"
    assert "--yes" in duplicate["message"]

    assert main(_invoke(runtime_root, "init", "--machine", "g1", "--yes", "--format=json")) == 0
    second = _json(capsys)
    assert second["old_runtime_id"] == first["new_runtime_id"]
    assert second["new_runtime_id"] != first["new_runtime_id"]


def test_project_init_register_inventory_name_and_local_only_removal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_root = tmp_path / "machine"
    first_project = tmp_path / "first"
    second_project = tmp_path / "second"
    assert main(_invoke(runtime_root, "init", "--machine", "g1", "--format=json")) == 0
    capsys.readouterr()

    for project in (first_project, second_project):
        assert main(_invoke(runtime_root, "project", "init", str(project), "--format=json")) == 0
        result = _json(capsys)
        assert result["shared_root"] == str((project / ".qexp").resolve())

    assert (
        main(
            _invoke(
                runtime_root,
                "project",
                "register",
                str(first_project),
                str(second_project),
                "--format=json",
            )
        )
        == 0
    )
    registered = _json(capsys)
    assert [item["status"] for item in registered["projects"]] == ["registered", "registered"]
    assert {item["machine_name"] for item in registered["projects"]} == {"g1"}

    assert main(_invoke(runtime_root, "agent", "name", "--set-to", "g2", "--format=json")) == 0
    renamed = _json(capsys)
    assert renamed["agent_name"] == "g2"
    assert renamed["runtime_id"] == registered["runtime_id"]

    assert main(_invoke(runtime_root, "project", "list", "--format=json")) == 0
    listing = _json(capsys)
    assert {item["machine_name"] for item in listing["projects"]} == {"g1"}
    first_id = next(
        item["project_id"]
        for item in listing["projects"]
        if item["shared_root"] == str((first_project / ".qexp").resolve())
    )
    second_id = next(
        item["project_id"]
        for item in listing["projects"]
        if item["shared_root"] == str((second_project / ".qexp").resolve())
    )
    assert main(_invoke(runtime_root, "project", "disable", second_id, "--format=json")) == 0
    assert _json(capsys)["enabled"] is False

    shutil.rmtree(first_project)
    assert main(_invoke(runtime_root, "init", "--machine", "g2", "--yes", "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "remove", first_id, "--format=json")) == 0
    removed = _json(capsys)
    assert removed["project_id"] == first_id
    assert removed["local_only"] is True
    assert (second_project / ".qexp" / "machines" / "g1").exists()

    assert main(_invoke(runtime_root, "project", "register", "--from-pool", "--format=json")) == 0
    pool_result = _json(capsys)
    assert first_id not in [item["project_id"] for item in pool_result["projects"]]
    assert pool_result["projects"] == [
        {
            **pool_result["projects"][0],
            "project_id": second_id,
            "enabled": False,
            "status": "disabled",
        }
    ]
    assert MachineRuntime(runtime_root).load_registry()[1][0].enabled is False


def test_interactive_reset_rejects_identity_changed_during_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from qqtools.plugins.qexp.agent.setup import initialize_machine

    runtime_root = tmp_path / "machine"
    runtime = MachineRuntime(runtime_root)
    assert main(_invoke(runtime_root, "init", "--machine", "source", "--format=json")) == 0
    first_id = _json(capsys)["new_runtime_id"]
    competing: dict[str, str] = {}

    class InteractiveInput:
        @staticmethod
        def isatty() -> bool:
            return True

    def replace_before_consent(_prompt: str) -> str:
        result = initialize_machine(runtime, "competitor", confirmed=True)
        competing["runtime_id"] = result["new_runtime_id"]
        return "y"

    monkeypatch.setattr(sys, "stdin", InteractiveInput())
    monkeypatch.setattr("builtins.input", replace_before_consent)

    assert main(_invoke(runtime_root, "init", "--machine", "requested")) == 2
    captured = capsys.readouterr()
    assert first_id in captured.err
    assert "identity changed after confirmation" in captured.err
    assert runtime.instance_id == competing["runtime_id"]


def test_explicit_alias_is_frozen_and_pool_inputs_are_exclusive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_root = tmp_path / "machine"
    project = tmp_path / "project"
    assert main(_invoke(runtime_root, "init", "--machine", "g1", "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "init", str(project), "--format=json")) == 0
    capsys.readouterr()
    assert (
        main(
            _invoke(
                runtime_root,
                "project",
                "register",
                str(project),
                "--machine",
                "trainer",
                "--name-source",
                "explicit",
                "--format=json",
            )
        )
        == 0
    )
    first = _json(capsys)["projects"][0]
    assert first["machine_name"] == "trainer"
    assert first["name_source"] == "explicit"

    assert main(_invoke(runtime_root, "agent", "name", "--set-to", "g2", "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "register", str(project), "--format=json")) == 0
    repeated = _json(capsys)["projects"][0]
    assert repeated["machine_name"] == "trainer"
    assert repeated["name_source"] == "explicit"

    assert (
        main(
            _invoke(
                runtime_root,
                "project",
                "register",
                str(project),
                "--from-pool",
                "--format=json",
            )
        )
        == 2
    )
    conflict = _json(capsys)["error"]
    assert conflict["code"] == "invalid_argument"
    assert "mutually exclusive" in conflict["message"]


def test_legacy_registry_migrates_to_unresolved_inventory_without_changing_binding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from qqtools.plugins.qexp.machine_config import init_shared_root

    runtime_root = tmp_path / "machine"
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "legacy-name", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(runtime_root)
    binding = runtime.add_binding(cfg.shared_root, cfg.machine_name)
    generation = binding.registration_generation

    assert main(_invoke(runtime_root, "project", "list", "--format=json")) == 0
    project = _json(capsys)["projects"][0]
    assert project["name_source"] == "unresolved"
    assert project["machine_name"] == "legacy-name"
    assert runtime.load_registry()[1][0].registration_generation == generation

    assert main(_invoke(runtime_root, "project", "register", "--from-pool", "--format=json")) == 2
    assert _json(capsys)["projects"][0]["reason"] == "name_source_unresolved"

    assert (
        main(
            _invoke(
                runtime_root,
                "project",
                "register",
                str(cfg.project_root),
                "--name-source",
                "explicit",
                "--format=json",
            )
        )
        == 0
    )
    confirmed = _json(capsys)["projects"][0]
    assert confirmed["name_source"] == "explicit"
    assert runtime.load_registry()[1][0].registration_generation == generation


def test_agent_start_rejects_a_registered_project_that_becomes_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from qqtools.plugins.qexp.machine_config import init_shared_root

    runtime_root = tmp_path / "machine"
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "g1", runtime_root=tmp_path / "legacy")
    runtime = MachineRuntime(runtime_root)
    runtime.add_binding(cfg.shared_root, cfg.machine_name)
    record_path = cfg.shared_root / "machines" / cfg.machine_name / "machine.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["machine"].pop("agent_runtime")
    record_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        "qqtools.plugins.qexp.cli.local_handlers.ensure_machine_agent_started",
        lambda _runtime: (None, {}),
    )

    assert main(_invoke(runtime_root, "agent", "start", "--timeout", "0.1", "--format=json")) == 2
    result = _json(capsys)
    assert result["ready"] is False
    assert result["projects"][0]["reason"] == "legacy_project_requires_migration"


def test_detachment_archives_unsettled_evidence_without_shared_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runtime_root = tmp_path / "machine"
    project = tmp_path / "project"
    assert main(_invoke(runtime_root, "init", "--machine", "source", "--format=json")) == 0
    old_runtime_id = _json(capsys)["new_runtime_id"]
    assert main(_invoke(runtime_root, "project", "init", str(project), "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "register", str(project), "--format=json")) == 0
    binding = _json(capsys)["projects"][0]
    cfg = load_root_config(project / ".qexp", "source", require_initialized=True)
    registration_before = (cfg.shared_root / "machines" / "source" / "registration.json").read_bytes()
    evidence = MachineRuntime(runtime_root).project_paths(binding["project_id"])["events"] / "pending.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text('{"event": {"pending": true}}', encoding="utf-8")

    assert main(_invoke(runtime_root, "init", "--machine", "clone", "--yes", "--format=json")) == 1
    unsettled = _json(capsys)["error"]
    assert unsettled["code"] == "operational_failure"
    assert "recovery" in unsettled["message"].lower()
    assert evidence.exists()

    assert (
        main(
            _invoke(
                runtime_root,
                "init",
                "--machine",
                "clone",
                "--detach-old-runtime",
                "--yes",
                "--format=json",
            )
        )
        == 0
    )
    result = _json(capsys)
    assert result["old_runtime_id"] == old_runtime_id
    assert result["detached"] is True
    assert Path(result["archive_path"]).is_dir()
    assert registration_before == (cfg.shared_root / "machines" / "source" / "registration.json").read_bytes()
    assert MachineRuntime(runtime_root).load_registry()[1] == []


def test_detachment_rejects_live_runner_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from qqtools.plugins.qexp.agent.helpers import _pid_start_time_ticks
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime_root = tmp_path / "machine"
    project = tmp_path / "project"
    assert main(_invoke(runtime_root, "init", "--machine", "source", "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "init", str(project), "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "register", str(project), "--format=json")) == 0
    project_id = _json(capsys)["projects"][0]["project_id"]
    runtime = MachineRuntime(runtime_root)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
    )
    try:
        start_ticks = _pid_start_time_ticks(process.pid)
        assert start_ticks is not None
        manifest = runtime.project_paths(project_id)["processes"] / "attempt.json"
        atomic_replace(
            manifest,
            {
                "process": {
                    "process_group_id": process.pid,
                    "process_group_start_time_ticks": start_ticks,
                }
            },
        )

        assert (
            main(
                _invoke(
                    runtime_root,
                    "init",
                    "--machine",
                    "clone",
                    "--detach-old-runtime",
                    "--yes",
                    "--format=json",
                )
            )
            == 1
        )
        live = _json(capsys)["error"]
        assert live["code"] == "operational_failure"
        assert "live or ambiguous" in live["message"]
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_detachment_rejects_malformed_runner_evidence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from qqtools.plugins.qexp.runtime.store import atomic_replace

    runtime_root = tmp_path / "machine"
    project = tmp_path / "project"
    assert main(_invoke(runtime_root, "init", "--machine", "source", "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "init", str(project), "--format=json")) == 0
    capsys.readouterr()
    assert main(_invoke(runtime_root, "project", "register", str(project), "--format=json")) == 0
    project_id = _json(capsys)["projects"][0]["project_id"]
    manifest = MachineRuntime(runtime_root).project_paths(project_id)["processes"] / "attempt.json"
    atomic_replace(manifest, {"process": {"process_group_id": "invalid"}})

    assert (
        main(
            _invoke(
                runtime_root,
                "init",
                "--machine",
                "clone",
                "--detach-old-runtime",
                "--yes",
                "--format=json",
            )
        )
        == 1
    )
    malformed = _json(capsys)["error"]
    assert malformed["code"] == "operational_failure"
    assert "live or ambiguous" in malformed["message"]


@pytest.mark.parametrize(
    ("legacy", "replacement"),
    [
        (("agent", "add-project"), "qexp project register"),
        (("agent", "list-projects"), "qexp project list"),
        (("agent", "enable-project", "id"), "qexp project enable"),
        (("agent", "disable-project", "id"), "qexp project disable"),
        (("agent", "remove-project", "id"), "qexp project remove"),
        (("agent", "migrate-project", "id"), "qexp admin migrate agent"),
    ],
)
def test_retired_project_commands_are_nonexecuting_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy: tuple[str, ...],
    replacement: str,
) -> None:
    assert main(_invoke(tmp_path / "machine", *legacy)) == 2
    captured = capsys.readouterr()
    assert "QQTOOLS-COMPAT-0014" in captured.err
    assert replacement in captured.err
    assert not (tmp_path / "machine").exists()

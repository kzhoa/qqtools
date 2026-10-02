from __future__ import annotations

from qqtools.plugins.qexp import init_shared_root
from qqtools.plugins.qexp.runtime.group_discovery.probe import initial_group_service_probe_state, probe_group_service
from tests.helpers.qexp_discovery import isolated_group


def test_empty_legacy_project_has_closed_group_quiescence_proof(tmp_path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")

    evidence = probe_group_service(cfg.shared_root, initial_group_service_probe_state())

    assert evidence["state"] == "quiescent"
    assert evidence["candidate"] is None
    assert evidence["probe_state"]["mode"] == "legacy"


def test_legacy_group_is_returned_as_exact_service_candidate(tmp_path):
    cfg = isolated_group(tmp_path)

    evidence = probe_group_service(cfg.shared_root, initial_group_service_probe_state())

    assert evidence["state"] == "active"
    assert evidence["candidate"] == {"group": "experiment", "lane": "legacy", "generation": None}


def test_legacy_directory_change_invalidates_an_old_complete_continuation(tmp_path):
    cfg = init_shared_root(tmp_path / "project" / ".qexp", "gpu-1", runtime_root=tmp_path / "runtime")
    complete = probe_group_service(cfg.shared_root, initial_group_service_probe_state())
    # Mutate this exact Project after the prior proof was captured.
    from qqtools.plugins.qexp.commands.group import create_group

    create_group(cfg, "later")

    evidence = probe_group_service(cfg.shared_root, complete["probe_state"])

    assert evidence["state"] == "active"
    assert evidence["candidate"]["group"] == "later"

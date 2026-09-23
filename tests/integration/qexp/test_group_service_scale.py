"""Scale qualification for obligation-driven Group service discovery."""

from __future__ import annotations

import pytest

from qqtools.plugins.qexp.runtime.group_discovery import activation, locator
from qqtools.plugins.qexp.runtime.locks import group_writer_lock
from tests.helpers.qexp.group_service_qualification import (
    measure_locator_traversal,
    populate_settled_group_history,
    process_resources,
)
from tests.helpers.qexp_discovery import isolated_group

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.fixture(autouse=True)
def target_writer(monkeypatch):
    monkeypatch.setattr(activation, "__version__", "1.3.22")


def activate(cfg) -> None:
    for _ in range(20_000):
        if activation.advance_group_service_activation(cfg)["state"] == "active":
            return
    raise AssertionError("Group service activation did not converge")


@pytest.mark.parametrize(
    "history",
    [
        0,
        1_000,
        pytest.param(10_000, marks=pytest.mark.stress),
        pytest.param(100_000, marks=pytest.mark.stress),
    ],
)
def test_active_locator_work_is_independent_of_retained_group_history(tmp_path, history):
    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    populate_settled_group_history(cfg, history)
    before = process_resources()
    with group_writer_lock(cfg, "experiment"):
        locator.publish_group_locator_locked(cfg, "experiment", "control", "task_change")
    measured = measure_locator_traversal(cfg.shared_root, "control", "experiment")
    after = measured["resources"]
    assert measured["directory_reads"] <= locator.SHARD_COUNT + 1
    assert after["descriptors"] <= before["descriptors"] + 2
    assert after["threads"] == before["threads"]


@pytest.mark.stress
def test_one_hundred_thousand_locator_lifecycles_keep_process_resources_bounded(tmp_path):
    cfg = isolated_group(tmp_path, tail=0)
    activate(cfg)
    baseline = process_resources()
    samples = []
    with group_writer_lock(cfg, "experiment"):
        for cycle in range(100_000):
            record = locator.publish_group_locator_locked(cfg, "experiment", "maintenance", "metadata_cleanup")
            assert locator.acknowledge_group_locator_locked(
                cfg,
                "experiment",
                "maintenance",
                record["generation"],
                retirement_ready=lambda: True,
            )
            if cycle % 10_000 == 9_999:
                samples.append(process_resources())
    assert all(sample["descriptors"] <= baseline["descriptors"] + 2 for sample in samples)
    assert all(sample["threads"] == baseline["threads"] for sample in samples)
    assert max(sample["rss_kib"] for sample in samples) - min(sample["rss_kib"] for sample in samples) <= 32 * 1024

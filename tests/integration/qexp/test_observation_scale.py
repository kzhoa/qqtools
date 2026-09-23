"""Explicit retained-history matrix for indexed queries and seek completeness."""

import os
import time

import pytest

from qqtools.plugins.qexp import observer
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics
from tests.helpers.qexp.observation_scale import retained_tasks

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "history_count",
    [0, 129, pytest.param(1000, marks=pytest.mark.stress), pytest.param(10_000, marks=pytest.mark.stress)],
)
def test_retained_history_queries_remain_bounded_and_complete(tmp_path, monkeypatch, history_count, record_property):
    cfg, truth = retained_tasks(tmp_path, history_count)
    shapes = [{}, {"group": "absent"}]
    phases = sorted({phase for phase, _group in truth.values()})
    groups = sorted({group for _phase, group in truth.values()})
    shapes.extend({"phase": phase} for phase in phases)
    shapes.extend({"group": group} for group in groups)
    shapes.extend({"phase": phase, "group": group} for phase in phases for group in groups)
    reports = []

    def no_inventory(*args, **kwargs):
        raise AssertionError("indexed history query enumerated a directory")

    with monkeypatch.context() as guard:
        guard.setattr(os, "scandir", no_inventory)
        guard.setattr(os, "listdir", no_inventory)
        for filters in shapes:
            expected = sorted(
                identifier
                for identifier, (phase, group) in truth.items()
                if (not filters.get("phase") or filters["phase"] == phase)
                and (not filters.get("group") or filters["group"] == group)
            )
            seen = []
            cursor = None
            for page_number in range(len(truth) + 2):
                diagnostics = RuntimeDiagnostics()
                start = time.monotonic()
                with activate_diagnostics(diagnostics):
                    page = observer.list_tasks_page(cfg, page_size=50, cursor=cursor, **filters)
                elapsed = time.monotonic() - start
                counters = diagnostics.snapshot()["counters"]
                assert counters.get("observation.index.pages", 0) <= 1089
                assert counters.get("observation.index.bytes", 0) < 13 * 1024 * 1024
                assert len(page["items"]) <= 50
                seen.extend(item["task_id"] for item in page["items"])
                if page_number in (0, 1) or page["next_cursor"] is None:
                    reports.append({"filters": filters, "page": page_number, "seconds": elapsed, "counters": counters})
                if page["next_cursor"] is None:
                    break
                assert page["next_cursor"] != cursor
                cursor = page["next_cursor"]
            else:
                pytest.fail("history page traversal did not terminate")
            assert seen == expected
    record_property("history_count", history_count)
    record_property("warm_uncontrolled_cache_pages", reports)

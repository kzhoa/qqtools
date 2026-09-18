"""Identity-specific reservation checks avoid machine inventory amplification."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.paths import local_paths
from qqtools.plugins.qexp.runtime.resources.reservations import has_reservation
from qqtools.plugins.qexp.runtime.store import atomic_replace
from qqtools.plugins.qexp.runtime.work_budget import RuntimeDiagnostics, activate_diagnostics

pytestmark = [pytest.mark.integration, pytest.mark.qexp_fast_io]


@pytest.mark.parametrize("lane", ["active", "provisional", "cpu_active", "cpu_provisional"])
@pytest.mark.parametrize("is_expired", [False, True])
def test_lookup_checks_only_named_identity_and_preserves_usage_rules(
    tmp_path: Path, lane: str, is_expired: bool
) -> None:
    root = tmp_path / "runtime"
    paths = local_paths(root)
    for name in ("active", "provisional", "cpu_active", "cpu_provisional"):
        paths[name].mkdir(parents=True)
        for number in range(64):
            (paths[name] / f"unrelated-{number}.json").write_text("malformed unrelated history")
    atomic_replace(
        paths[lane] / "target.json",
        {"reservation": {"reservation_id": "target", "expires_at": "2000-01-01T00:00:00Z" if is_expired else None}},
    )
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics):
        assert has_reservation(root, "target") == (not is_expired or lane in {"active", "cpu_active"})
    counters = diagnostics.counters
    assert 1 <= counters["store.read_json.calls"] <= 4
    assert 1 <= counters["locks.acquire.calls"] <= 2
    assert counters["store.iter_json.calls"] == 0
    assert counters["store.atomic_replace.calls"] == 0
    assert (paths[lane] / "target.json").exists()


def test_missing_reservation_uses_four_point_reads(tmp_path: Path) -> None:
    diagnostics = RuntimeDiagnostics()
    with activate_diagnostics(diagnostics):
        assert not has_reservation(tmp_path / "runtime", "missing")
    assert diagnostics.counters["store.read_json.calls"] == 4
    assert diagnostics.counters["locks.acquire.calls"] == 2


@pytest.mark.parametrize("value", [{"reservation": {"reservation_id": "different"}}, {}])
def test_corrupt_matching_record_is_not_reported_as_free_capacity(tmp_path: Path, value: dict) -> None:
    root = tmp_path / "runtime"
    atomic_replace(local_paths(root)["active"] / "target.json", value)
    with pytest.raises((ValueError, KeyError)):
        has_reservation(root, "target")


@pytest.mark.parametrize("identity", ["", ".", "..", "../target", "/target"])
def test_lookup_rejects_non_identity_paths(tmp_path: Path, identity: str) -> None:
    with pytest.raises(ValueError, match="filename component"):
        has_reservation(tmp_path, identity)

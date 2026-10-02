"""Fixed identity reads do not initialize or traverse the responsibility ledger."""

from pathlib import Path

import pytest

from qqtools.plugins.qexp.runtime.responsibility_store import BUCKETS, FORMAT, Ledger, Unavailable, read_ledger_instance
from qqtools.plugins.qexp.runtime.store import atomic_replace

pytestmark = pytest.mark.integration


def test_fixed_identity_read_opens_only_existing_marker_without_ledger_construction(tmp_path, monkeypatch):
    root = tmp_path / "ledger"
    ledger = Ledger.create(root)
    marker = root / "marker"
    before = marker.read_bytes()
    original = Path.open

    def marker_only(path, *args, **kwargs):
        assert path == marker
        mode = args[0] if args else kwargs.get("mode", "r")
        assert "w" not in mode and "+" not in mode
        return original(path, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", marker_only)
        guarded.setattr(Ledger, "__init__", lambda *_: pytest.fail("identity read constructed ledger"))
        assert read_ledger_instance(root) == ledger.instance
    assert marker.read_bytes() == before


@pytest.mark.parametrize(
    "key,value", [("format", "unknown"), ("buckets", 1), ("instance", "unknown"), ("instance", True)]
)
def test_fixed_identity_read_rejects_invalid_marker_without_repair(tmp_path, key, value):
    root = tmp_path / "ledger"
    marker = root / "marker"
    atomic_replace(marker, {"format": FORMAT, "buckets": BUCKETS, "instance": "a" * 32, key: value})
    before = marker.read_bytes()
    with pytest.raises(Unavailable):
        read_ledger_instance(root)
    assert marker.read_bytes() == before


def test_fixed_identity_read_does_not_initialize_missing_root(tmp_path):
    root = tmp_path / "missing"
    with pytest.raises(Unavailable):
        read_ledger_instance(root)
    assert not root.exists()

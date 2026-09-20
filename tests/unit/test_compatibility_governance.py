from __future__ import annotations

from pathlib import Path

import pytest

from scripts.checks.check_compatibility_registry import RegistryError, Version, check_release, load_registry


def _write_registry(
    root: Path,
    *,
    item_id: str = "QQTOOLS-COMPAT-9001",
    component: str = "example.component",
    status: str = "compatibility_active",
    summary: str = "Temporary compatibility boundary for the example component.",
    introduced_in: str = "2.0.0",
    legacy_removed_in: str = "3.0.0",
    transition_purged_in: str = "4.0.0",
    extensions: str = "",
    next_id: int | None = None,
) -> Path:
    verification = root / "tests/test_example.py"
    verification.parent.mkdir(parents=True, exist_ok=True)
    verification.write_text("evidence\n", encoding="utf-8")
    if status in {"compatibility_active", "legacy_removed"}:
        marker_path = root / "src/example.py"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(f"# {item_id}\n", encoding="utf-8")
    registry = root / "docs/spec/compatibility-registry.toml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    item_number = int(item_id.rsplit("-", 1)[1])
    effective_next_id = next_id if next_id is not None else item_number + 1
    registry.write_text(
        f"""next_id = {effective_next_id}

[[items]]
id = "{item_id}"
component = "{component}"
kind = "public_api"
status = "{status}"
summary = "{summary}"
introduced_in = "{introduced_in}"
legacy_removed_in = "{legacy_removed_in}"
transition_purged_in = "{transition_purged_in}"
marker = "{item_id}"
owner = "example"
verification = ["tests/test_example.py"]
{extensions}
""",
        encoding="utf-8",
    )
    return registry


def _write_empty_registry(root: Path, *, next_id: int = 9002) -> Path:
    registry = root / "docs/spec/compatibility-registry.toml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(f"next_id = {next_id}\n", encoding="utf-8")
    return registry


def test_registry_validates_active_item_and_release_state(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)

    items = load_registry(registry, tmp_path)

    check_release(items, Version.parse("2.5.0", "target"))
    with pytest.raises(RegistryError, match="expected legacy_removed"):
        check_release(items, Version.parse("3.0.0", "target"))


def test_release_states_support_overlapping_windows(tmp_path: Path) -> None:
    first_registry = _write_registry(
        tmp_path / "first",
        status="legacy_removed",
        introduced_in="1.0.0",
        legacy_removed_in="2.0.0",
        transition_purged_in="3.0.0",
    )
    second_registry = _write_registry(
        tmp_path / "second",
        item_id="QQTOOLS-COMPAT-9002",
        status="legacy_removed",
        introduced_in="2.0.0",
        legacy_removed_in="3.0.0",
        transition_purged_in="4.0.0",
    )
    first = load_registry(first_registry, tmp_path / "first")[0]
    second = load_registry(second_registry, tmp_path / "second")[0]

    target = Version.parse("3.0.0", "target")

    assert first.expected_status(target) is None
    check_release((second,), target)


@pytest.mark.parametrize(
    ("introduced", "removed", "purged", "message"),
    [
        ("2", "3.0.0", "4.0.0", "exact X.Y.Z"),
        ("3.0.0", "3.0.0", "4.0.0", "must precede"),
        ("2.0.0", "4.0.0", "3.0.0", "must not precede"),
    ],
)
def test_registry_rejects_invalid_version_contract(
    tmp_path: Path,
    introduced: str,
    removed: str,
    purged: str,
    message: str,
) -> None:
    registry = _write_registry(
        tmp_path,
        introduced_in=introduced,
        legacy_removed_in=removed,
        transition_purged_in=purged,
    )

    with pytest.raises(RegistryError, match=message):
        load_registry(registry, tmp_path)


def test_registry_rejects_marker_for_planned_item(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path,
        status="planned",
    )
    marker = tmp_path / "src/example.py"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("# QQTOOLS-COMPAT-9001\n", encoding="utf-8")

    with pytest.raises(RegistryError, match="planned but its marker exists"):
        load_registry(registry, tmp_path)


def test_registry_requires_summary(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)
    registry.write_text(
        registry.read_text(encoding="utf-8").replace(
            'summary = "Temporary compatibility boundary for the example component."\n',
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="summary must be a non-empty string"):
        load_registry(registry, tmp_path)


def test_registry_rejects_schema_version(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)
    registry.write_text(
        "schema_version = 1\n" + registry.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="unknown field 'schema_version'"):
        load_registry(registry, tmp_path)


@pytest.mark.parametrize("field", ["decision_refs", "pitch_refs", "action_refs"])
def test_registry_rejects_private_or_action_reference_fields(tmp_path: Path, field: str) -> None:
    registry = _write_registry(tmp_path)
    registry.write_text(
        registry.read_text(encoding="utf-8") + f'{field} = ["docs/private/example.md"]\n',
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match=f"unknown field '{field}'"):
        load_registry(registry, tmp_path)


def test_registry_rejects_completed_as_a_persisted_status(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, status="completed")

    with pytest.raises(RegistryError, match="status must be one of"):
        load_registry(registry, tmp_path)


def test_registry_allows_no_unfinished_items(tmp_path: Path) -> None:
    registry = load_registry(_write_empty_registry(tmp_path), tmp_path)

    assert registry.items == ()
    assert registry.next_id == 9002


def test_registry_requires_next_id_above_every_item_id(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path, next_id=9001)

    with pytest.raises(RegistryError, match="lower than registry next_id"):
        load_registry(registry, tmp_path)


def test_registry_rejects_missing_verification_reference(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)
    (tmp_path / "tests/test_example.py").unlink()

    with pytest.raises(RegistryError, match="verification path does not exist"):
        load_registry(registry, tmp_path)


def test_registry_rejects_unknown_item_field(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)
    registry.write_text(
        registry.read_text(encoding="utf-8") + 'remove_in = "3.0.0"\n',
        encoding="utf-8",
    )

    with pytest.raises(RegistryError, match="unknown field 'remove_in'"):
        load_registry(registry, tmp_path)


def test_registry_accepts_chained_deadline_extension(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path,
        extensions="""
[[items.extensions]]
field = "legacy_removed_in"
from = "3.0.0"
to = "3.1.0"
approved_in = "2.5.0"
reason = "Deployment inventory needs one more release."
""",
    )

    item = load_registry(registry, tmp_path)[0]

    assert str(item.legacy_removed_in) == "3.1.0"


def test_registry_rejects_broken_deadline_extension_chain(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path,
        extensions="""
[[items.extensions]]
field = "legacy_removed_in"
from = "2.9.0"
to = "3.1.0"
approved_in = "2.5.0"
reason = "Deployment inventory needs one more release."
""",
    )

    with pytest.raises(RegistryError, match="previous effective legacy_removed_in"):
        load_registry(registry, tmp_path)


def test_repository_registry_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]

    items = load_registry(root / "docs/spec/compatibility-registry.toml", root)

    retired_ids = {
        "QQTOOLS-COMPAT-" + "0001",
        "QQTOOLS-COMPAT-" + "0002",
        "QQTOOLS-COMPAT-" + "0004",
    }
    assert items.next_id >= 5
    assert retired_ids.isdisjoint(item.item_id for item in items)


def test_planned_future_item_passes_an_earlier_release(tmp_path: Path) -> None:
    registry = _write_registry(
        tmp_path,
        status="planned",
        introduced_in="3.0.0",
        legacy_removed_in="4.0.0",
        transition_purged_in="5.0.0",
    )
    item = load_registry(registry, tmp_path)[0]

    check_release((item,), Version.parse("2.9.0", "target"))

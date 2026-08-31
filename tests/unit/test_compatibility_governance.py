from __future__ import annotations

from pathlib import Path

import pytest

from scripts.checks.check_compatibility_registry import (
    RegistryError,
    Version,
    check_release,
    check_registry_transition,
    load_registry,
)
from scripts import release_preflight


def _write_registry(
    root: Path,
    *,
    item_id: str = "QQTOOLS-COMPAT-9001",
    component: str = "example.component",
    status: str = "compatibility_active",
    introduced_in: str = "2.0.0",
    legacy_removed_in: str = "3.0.0",
    transition_purged_in: str = "4.0.0",
    extensions: str = "",
    next_id: int | None = None,
) -> Path:
    for path in (root / "docs/spec/decision.md", root / "tests/test_example.py"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n", encoding="utf-8")
    if status in {"compatibility_active", "legacy_removed"}:
        marker_path = root / "src/example.py"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(f"# {item_id}\n", encoding="utf-8")
    registry = root / "docs/spec/compatibility-registry.toml"
    item_number = int(item_id.rsplit("-", 1)[1])
    effective_next_id = next_id if next_id is not None else item_number + 1
    registry.write_text(
        f"""schema_version = 1
next_id = {effective_next_id}

[[items]]
id = "{item_id}"
component = "{component}"
kind = "public_api"
status = "{status}"
introduced_in = "{introduced_in}"
legacy_removed_in = "{legacy_removed_in}"
transition_purged_in = "{transition_purged_in}"
marker = "{item_id}"
owner = "example"
decision_refs = ["docs/spec/decision.md"]
verification = ["tests/test_example.py"]
{extensions}
""",
        encoding="utf-8",
    )
    return registry


def _write_empty_registry(root: Path, *, next_id: int = 9002) -> Path:
    registry = root / "docs/spec/compatibility-registry.toml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(f"schema_version = 1\nnext_id = {next_id}\n", encoding="utf-8")
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
    registry = _write_registry(tmp_path, status="planned")
    marker = tmp_path / "src/example.py"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("# QQTOOLS-COMPAT-9001\n", encoding="utf-8")

    with pytest.raises(RegistryError, match="planned but its marker exists"):
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


def test_registry_rejects_missing_decision_reference(tmp_path: Path) -> None:
    registry = _write_registry(tmp_path)
    (tmp_path / "docs/spec/decision.md").unlink()

    with pytest.raises(RegistryError, match="path does not exist"):
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
    decision = tmp_path / "docs/spec/extension.md"
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text("approved\n", encoding="utf-8")
    registry = _write_registry(
        tmp_path,
        extensions="""
[[items.extensions]]
field = "legacy_removed_in"
from = "3.0.0"
to = "3.1.0"
approved_in = "2.5.0"
reason = "Deployment inventory needs one more release."
decision_ref = "docs/spec/extension.md"
""",
    )

    item = load_registry(registry, tmp_path)[0]

    assert str(item.legacy_removed_in) == "3.1.0"


def test_registry_rejects_broken_deadline_extension_chain(tmp_path: Path) -> None:
    decision = tmp_path / "docs/spec/extension.md"
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text("approved\n", encoding="utf-8")
    registry = _write_registry(
        tmp_path,
        extensions="""
[[items.extensions]]
field = "legacy_removed_in"
from = "2.9.0"
to = "3.1.0"
approved_in = "2.5.0"
reason = "Deployment inventory needs one more release."
decision_ref = "docs/spec/extension.md"
""",
    )

    with pytest.raises(RegistryError, match="previous effective legacy_removed_in"):
        load_registry(registry, tmp_path)


def test_repository_registry_is_valid() -> None:
    root = Path(__file__).resolve().parents[2]

    items = load_registry(root / "docs/spec/compatibility-registry.toml", root)

    assert {
        "QQTOOLS-COMPAT-" + "0001",
        "QQTOOLS-COMPAT-" + "0002",
    } <= {item.item_id for item in items}


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


def test_registry_allows_removal_at_purge_version(tmp_path: Path) -> None:
    previous = load_registry(
        _write_registry(tmp_path, status="legacy_removed"),
        tmp_path,
    )
    (tmp_path / "src/example.py").unlink()
    current = load_registry(_write_empty_registry(tmp_path), tmp_path)

    check_registry_transition(
        current,
        previous,
        Version.parse("4.0.0", "target"),
        tmp_path,
    )


def test_registry_rejects_removal_before_purge_version(tmp_path: Path) -> None:
    previous = load_registry(
        _write_registry(tmp_path, status="legacy_removed"),
        tmp_path,
    )
    (tmp_path / "src/example.py").unlink()
    current = load_registry(_write_empty_registry(tmp_path), tmp_path)

    with pytest.raises(RegistryError, match="removed before transition_purged_in"):
        check_registry_transition(
            current,
            previous,
            Version.parse("3.9.0", "target"),
            tmp_path,
        )


def test_registry_rejects_retirement_while_marker_remains(tmp_path: Path) -> None:
    previous = load_registry(
        _write_registry(tmp_path, status="legacy_removed"),
        tmp_path,
    )
    current = load_registry(_write_empty_registry(tmp_path), tmp_path)

    with pytest.raises(RegistryError, match="retired but its marker remains"):
        check_registry_transition(
            current,
            previous,
            Version.parse("4.0.0", "target"),
            tmp_path,
        )


def test_registry_rejects_decreasing_next_id(tmp_path: Path) -> None:
    previous = load_registry(_write_registry(tmp_path), tmp_path)
    (tmp_path / "src/example.py").unlink()
    current = load_registry(_write_empty_registry(tmp_path, next_id=9001), tmp_path)

    with pytest.raises(RegistryError, match="next_id decreased"):
        check_registry_transition(
            current,
            previous,
            Version.parse("4.0.0", "target"),
            tmp_path,
        )


def test_registry_rejects_reusing_retired_id(tmp_path: Path) -> None:
    previous_path = _write_registry(tmp_path / "previous", next_id=9002)
    previous = load_registry(previous_path, tmp_path / "previous")
    current_path = _write_registry(
        tmp_path / "current",
        item_id="QQTOOLS-COMPAT-9000",
        next_id=9002,
    )
    current = load_registry(current_path, tmp_path / "current")

    with pytest.raises(RegistryError, match="cannot be reused"):
        check_registry_transition(
            current,
            previous,
            Version.parse("2.5.0", "target"),
            tmp_path / "current",
        )


def test_registry_bootstrap_has_no_previous_release_contract(tmp_path: Path) -> None:
    current = load_registry(_write_empty_registry(tmp_path), tmp_path)

    check_registry_transition(
        current,
        None,
        Version.parse("4.0.0", "target"),
        tmp_path,
    )


def test_preflight_runs_compatibility_gate_before_expensive_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(release_preflight, "_require_clean_head", lambda: order.append("clean"))
    monkeypatch.setattr(release_preflight, "_check_target_version", lambda target: order.append("target"))
    monkeypatch.setattr(release_preflight, "_check_compatibility", lambda target: order.append("compatibility"))
    monkeypatch.setattr(release_preflight, "_check_lazy_export_stubs", lambda: order.append("stubs"))
    monkeypatch.setattr(release_preflight, "_release_env", lambda: {})
    monkeypatch.setattr(
        release_preflight,
        "_run",
        lambda *args, **kwargs: order.append("run"),
    )
    monkeypatch.setattr(
        release_preflight,
        "_build_artifacts",
        lambda *args, **kwargs: Path("package.whl"),
    )

    assert release_preflight.main(["--target-version", "1.3.13"]) == 0
    assert order[:4] == ["clean", "target", "compatibility", "stubs"]


def test_preflight_stops_when_compatibility_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    monkeypatch.setattr(release_preflight, "_require_clean_head", lambda: order.append("clean"))
    monkeypatch.setattr(release_preflight, "_check_target_version", lambda target: order.append("target"))

    def fail_compatibility(target: Version) -> None:
        order.append("compatibility")
        raise RuntimeError("compatibility gate failed")

    monkeypatch.setattr(release_preflight, "_check_compatibility", fail_compatibility)
    monkeypatch.setattr(release_preflight, "_check_lazy_export_stubs", lambda: order.append("stubs"))

    with pytest.raises(RuntimeError, match="compatibility gate failed"):
        release_preflight.main(["--target-version", "1.3.13"])
    assert order == ["clean", "target", "compatibility"]


def test_preflight_rejects_non_future_target(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release_preflight,
        "_current_version",
        lambda: Version.parse("1.3.12", "current"),
    )

    with pytest.raises(RuntimeError, match="must be later"):
        release_preflight._check_target_version(Version.parse("1.3.12", "target"))


def test_preflight_rejects_non_exact_target_version() -> None:
    with pytest.raises(SystemExit) as exc_info:
        release_preflight.main(["--target-version", "1.3"])

    assert exc_info.value.code == 2

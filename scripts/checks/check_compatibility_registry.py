#!/usr/bin/env python3
"""Validate compatibility lifecycles and release obligations."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path("docs/spec/compatibility-registry.toml")
DEFAULT_VERSION_PATH = Path("src/qqtools/version.py")
MARKER_ROOTS = (Path("src"), Path("tests"), Path("scripts"))
ID_PATTERN = re.compile(r"QQTOOLS-COMPAT-[0-9]{4}")
VERSION_PATTERN = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
STATUSES = {"planned", "compatibility_active", "legacy_removed"}
KINDS = {
    "cli",
    "cli_and_local_state",
    "local_state",
    "persisted_schema",
    "public_api",
    "runtime_protocol",
}
EXTENDABLE_FIELDS = {"legacy_removed_in", "transition_purged_in"}
REGISTRY_FIELDS = {"schema_version", "next_id", "items"}
ITEM_FIELDS = {
    "id",
    "component",
    "kind",
    "status",
    "introduced_in",
    "legacy_removed_in",
    "transition_purged_in",
    "marker",
    "owner",
    "decision_refs",
    "verification",
    "extensions",
}
EXTENSION_FIELDS = {"field", "from", "to", "approved_in", "reason", "decision_ref"}


class RegistryError(ValueError):
    """Raised when the compatibility registry violates its contract."""


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, value: object, field: str) -> "Version":
        if not isinstance(value, str) or VERSION_PATTERN.fullmatch(value) is None:
            raise RegistryError(f"{field} must use an exact X.Y.Z version, got {value!r}.")
        return cls(*(int(part) for part in value.split(".")))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class CompatibilityItem:
    item_id: str
    component: str
    kind: str
    status: str
    introduced_in: Version
    legacy_removed_in: Version
    transition_purged_in: Version
    marker: str
    owner: str
    decision_refs: tuple[Path, ...]
    verification: tuple[Path, ...]

    def expected_status(self, target: Version) -> str | None:
        if target < self.introduced_in:
            return "planned"
        if target < self.legacy_removed_in:
            return "compatibility_active"
        if target < self.transition_purged_in:
            return "legacy_removed"
        return None


@dataclass(frozen=True)
class CompatibilityRegistry:
    next_id: int
    items: tuple[CompatibilityItem, ...]

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> CompatibilityItem:
        return self.items[index]


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{field} must be a non-empty string.")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise RegistryError(f"{field} contains unknown field {unknown[0]!r}.")


def _require_paths(value: object, field: str) -> tuple[Path, ...]:
    if not isinstance(value, list) or not value:
        raise RegistryError(f"{field} must be a non-empty array of repository-relative paths.")
    paths: list[Path] = []
    for index, raw_path in enumerate(value):
        path = Path(_require_string(raw_path, f"{field}[{index}]"))
        if path.is_absolute() or ".." in path.parts:
            raise RegistryError(f"{field}[{index}] must be a repository-relative path.")
        paths.append(path)
    return tuple(paths)


def _is_tracked_candidate(repo_root: Path, path: Path) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", str(path)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and str(path) in result.stdout.splitlines()


def _has_git_repository(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return Path(result.stdout.strip()).resolve() == repo_root.resolve()


def _validate_reference_paths(
    repo_root: Path,
    paths: Iterable[Path],
    field: str,
    *,
    require_tracked: bool,
) -> None:
    for path in paths:
        if not (repo_root / path).is_file():
            raise RegistryError(f"{field} path does not exist: {path}.")
        if require_tracked and not _is_tracked_candidate(repo_root, path):
            raise RegistryError(f"{field} path is ignored or unavailable to Git: {path}.")


def _apply_extensions(
    raw_extensions: object,
    item_id: str,
    deadlines: dict[str, Version],
    repo_root: Path,
    introduced_in: Version,
    *,
    validate_references: bool,
    require_tracked: bool,
) -> None:
    if raw_extensions is None:
        return
    if not isinstance(raw_extensions, list):
        raise RegistryError(f"{item_id}.extensions must be an array of tables.")
    for index, raw_extension in enumerate(raw_extensions):
        prefix = f"{item_id}.extensions[{index}]"
        if not isinstance(raw_extension, dict):
            raise RegistryError(f"{prefix} must be a table.")
        _reject_unknown(raw_extension, EXTENSION_FIELDS, prefix)
        field = _require_string(raw_extension.get("field"), f"{prefix}.field")
        if field not in EXTENDABLE_FIELDS:
            raise RegistryError(f"{prefix}.field cannot extend {field!r}.")
        old_version = Version.parse(raw_extension.get("from"), f"{prefix}.from")
        new_version = Version.parse(raw_extension.get("to"), f"{prefix}.to")
        approved_in = Version.parse(raw_extension.get("approved_in"), f"{prefix}.approved_in")
        _require_string(raw_extension.get("reason"), f"{prefix}.reason")
        decision_ref = Path(_require_string(raw_extension.get("decision_ref"), f"{prefix}.decision_ref"))
        if decision_ref.is_absolute() or ".." in decision_ref.parts:
            raise RegistryError(f"{prefix}.decision_ref must be repository-relative.")
        if validate_references:
            _validate_reference_paths(
                repo_root,
                (decision_ref,),
                f"{prefix}.decision_ref",
                require_tracked=require_tracked,
            )
        if deadlines[field] != old_version:
            raise RegistryError(
                f"{prefix}.from must equal the previous effective {field} " f"({deadlines[field]}), got {old_version}."
            )
        if new_version <= old_version:
            raise RegistryError(f"{prefix}.to must be later than {prefix}.from.")
        if not introduced_in <= approved_in <= old_version:
            raise RegistryError(f"{prefix}.approved_in must be between introduced_in and {prefix}.from.")
        deadlines[field] = new_version


def _parse_item(
    raw_item: object,
    index: int,
    repo_root: Path,
    *,
    validate_references: bool,
    require_tracked: bool,
) -> CompatibilityItem:
    if not isinstance(raw_item, dict):
        raise RegistryError(f"items[{index}] must be a table.")
    _reject_unknown(raw_item, ITEM_FIELDS, f"items[{index}]")
    item_id = _require_string(raw_item.get("id"), f"items[{index}].id")
    if ID_PATTERN.fullmatch(item_id) is None:
        raise RegistryError(f"items[{index}].id must match QQTOOLS-COMPAT-NNNN.")
    component = _require_string(raw_item.get("component"), f"{item_id}.component")
    kind = _require_string(raw_item.get("kind"), f"{item_id}.kind")
    if kind not in KINDS:
        raise RegistryError(f"{item_id}.kind must be one of {sorted(KINDS)}, got {kind!r}.")
    status = _require_string(raw_item.get("status"), f"{item_id}.status")
    if status not in STATUSES:
        raise RegistryError(f"{item_id}.status must be one of {sorted(STATUSES)}, got {status!r}.")
    introduced_in = Version.parse(raw_item.get("introduced_in"), f"{item_id}.introduced_in")
    deadlines = {
        "legacy_removed_in": Version.parse(raw_item.get("legacy_removed_in"), f"{item_id}.legacy_removed_in"),
        "transition_purged_in": Version.parse(raw_item.get("transition_purged_in"), f"{item_id}.transition_purged_in"),
    }
    if not introduced_in < deadlines["legacy_removed_in"]:
        raise RegistryError(f"{item_id}.introduced_in must precede legacy_removed_in.")
    if deadlines["transition_purged_in"] < deadlines["legacy_removed_in"]:
        raise RegistryError(f"{item_id}.transition_purged_in must not precede legacy_removed_in.")
    _apply_extensions(
        raw_item.get("extensions"),
        item_id,
        deadlines,
        repo_root,
        introduced_in,
        validate_references=validate_references,
        require_tracked=require_tracked,
    )
    if deadlines["transition_purged_in"] < deadlines["legacy_removed_in"]:
        raise RegistryError(f"{item_id} effective transition_purged_in must not precede legacy_removed_in.")
    marker = _require_string(raw_item.get("marker"), f"{item_id}.marker")
    if marker != item_id:
        raise RegistryError(f"{item_id}.marker must equal its compatibility ID.")
    owner = _require_string(raw_item.get("owner"), f"{item_id}.owner")
    decision_refs = _require_paths(raw_item.get("decision_refs"), f"{item_id}.decision_refs")
    verification = _require_paths(raw_item.get("verification"), f"{item_id}.verification")
    if validate_references:
        _validate_reference_paths(
            repo_root,
            decision_refs,
            f"{item_id}.decision_refs",
            require_tracked=require_tracked,
        )
        _validate_reference_paths(
            repo_root,
            verification,
            f"{item_id}.verification",
            require_tracked=require_tracked,
        )
    return CompatibilityItem(
        item_id=item_id,
        component=component,
        kind=kind,
        status=status,
        introduced_in=introduced_in,
        legacy_removed_in=deadlines["legacy_removed_in"],
        transition_purged_in=deadlines["transition_purged_in"],
        marker=marker,
        owner=owner,
        decision_refs=decision_refs,
        verification=verification,
    )


def _candidate_marker_paths(repo_root: Path) -> tuple[Path, ...]:
    if _has_git_repository(repo_root):
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *(str(path) for path in MARKER_ROOTS),
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            details = result.stderr.strip()
            raise RegistryError(f"could not list compatibility marker files: {details}")
        return tuple(Path(line) for line in result.stdout.splitlines() if line)
    paths = []
    for relative_root in MARKER_ROOTS:
        root = repo_root / relative_root
        if root.is_dir():
            paths.extend(path.relative_to(repo_root) for path in root.rglob("*") if path.is_file())
    return tuple(sorted(paths))


def _marker_files(repo_root: Path, markers: Iterable[str]) -> dict[str, tuple[Path, ...]]:
    marker_bytes = {marker: marker.encode("utf-8") for marker in markers}
    matches: dict[str, list[Path]] = {marker: [] for marker in marker_bytes}
    for relative_path in _candidate_marker_paths(repo_root):
        try:
            content = (repo_root / relative_path).read_bytes()
        except OSError as exc:
            raise RegistryError(f"could not inspect compatibility markers in {relative_path}: {exc}") from exc
        for marker, encoded_marker in marker_bytes.items():
            if encoded_marker in content:
                matches[marker].append(relative_path)
    return {marker: tuple(sorted(paths)) for marker, paths in matches.items()}


def _parse_registry(
    raw_registry: dict[str, Any],
    repo_root: Path,
    *,
    validate_repository: bool,
) -> CompatibilityRegistry:
    """Parse registry data and optionally validate working-tree evidence."""
    _reject_unknown(raw_registry, REGISTRY_FIELDS, "registry")
    if raw_registry.get("schema_version") != 1:
        raise RegistryError("compatibility registry schema_version must be 1.")
    next_id = raw_registry.get("next_id")
    if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id < 1:
        raise RegistryError("compatibility registry next_id must be a positive integer.")
    raw_items = raw_registry.get("items", [])
    if not isinstance(raw_items, list):
        raise RegistryError("compatibility registry items must be an array of tables.")
    require_tracked = validate_repository and _has_git_repository(repo_root)
    items = tuple(
        _parse_item(
            item,
            index,
            repo_root,
            validate_references=validate_repository,
            require_tracked=require_tracked,
        )
        for index, item in enumerate(raw_items)
    )
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise RegistryError("compatibility registry item IDs must be unique.")
    id_numbers = [int(item_id.rsplit("-", 1)[1]) for item_id in item_ids]
    if any(id_number >= next_id for id_number in id_numbers):
        raise RegistryError("every compatibility item ID must be lower than registry next_id.")
    components = [item.component for item in items]
    if len(components) != len(set(components)):
        raise RegistryError("compatibility registry components must be unique.")
    if validate_repository:
        marker_matches = _marker_files(repo_root, (item.marker for item in items))
        for item in items:
            marker_files = marker_matches[item.marker]
            if item.status == "planned" and marker_files:
                raise RegistryError(f"{item.item_id} is planned but its marker exists in {list(marker_files)}.")
            if item.status in {"compatibility_active", "legacy_removed"} and not marker_files:
                raise RegistryError(f"{item.item_id} status {item.status} requires a code/test marker.")
    return CompatibilityRegistry(next_id=next_id, items=items)


def load_registry(registry_path: Path, repo_root: Path) -> CompatibilityRegistry:
    try:
        raw_registry = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RegistryError(f"could not read compatibility registry {registry_path}: {exc}") from exc
    return _parse_registry(raw_registry, repo_root, validate_repository=True)


def _current_version(repo_root: Path) -> Version:
    version_path = repo_root / DEFAULT_VERSION_PATH
    try:
        version_source = version_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"could not read {DEFAULT_VERSION_PATH}: {exc}") from exc
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        version_source,
        re.MULTILINE,
    )
    if match is None:
        raise RegistryError(f"could not resolve __version__ from {DEFAULT_VERSION_PATH}.")
    return Version.parse(match.group(1), str(DEFAULT_VERSION_PATH))


def load_previous_registry(
    repo_root: Path,
    current_version: Version,
) -> CompatibilityRegistry | None:
    """Load the registry from the exact previous release tag, if governance existed then."""
    if not _has_git_repository(repo_root):
        return None
    tag = f"v{current_version}"
    tag_check = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if tag_check.returncode != 0:
        raise RegistryError(f"previous release tag {tag} does not exist.")
    registry_path = DEFAULT_REGISTRY.as_posix()
    tree_result = subprocess.run(
        ["git", "ls-tree", "--name-only", tag, "--", registry_path],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if tree_result.returncode != 0:
        details = tree_result.stderr.strip()
        raise RegistryError(f"could not inspect compatibility registry in {tag}: {details}")
    if registry_path not in tree_result.stdout.splitlines():
        return None
    result = subprocess.run(
        ["git", "show", f"{tag}:{registry_path}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip()
        raise RegistryError(f"could not read compatibility registry from {tag}: {details}")
    try:
        raw_registry = tomllib.loads(result.stdout)
    except tomllib.TOMLDecodeError as exc:
        raise RegistryError(f"could not parse compatibility registry from {tag}: {exc}") from exc
    return _parse_registry(raw_registry, repo_root, validate_repository=False)


def _expected_label(expected_status: str | None) -> str:
    return expected_status if expected_status is not None else "registry removal"


def release_plan(
    items: Iterable[CompatibilityItem],
    target: Version,
    previous: CompatibilityRegistry | None = None,
) -> list[str]:
    items = tuple(items)
    lines = [f"Compatibility plan for release {target}:"]
    for item in items:
        expected = item.expected_status(target)
        lines.append(f"- {item.item_id} ({item.component}): {item.status} -> expected " f"{_expected_label(expected)}")
    current_ids = {item.item_id for item in items}
    if previous is not None:
        for item in previous:
            if item.item_id not in current_ids:
                lines.append(
                    f"- {item.item_id} ({item.component}): absent -> expected "
                    f"{_expected_label(item.expected_status(target))}"
                )
    return lines


def check_release(items: Iterable[CompatibilityItem], target: Version) -> None:
    mismatches = []
    for item in items:
        expected = item.expected_status(target)
        if expected is None:
            mismatches.append(
                f"{item.item_id} ({item.component}) remains in the registry but must be removed " f"for {target}"
            )
        elif item.status != expected:
            mismatches.append(
                f"{item.item_id} ({item.component}) is {item.status}, expected {expected} " f"for {target}"
            )
    if mismatches:
        details = "\n".join(f"- {mismatch}" for mismatch in mismatches)
        raise RegistryError(f"compatibility release gate failed:\n{details}")


def check_registry_transition(
    current: CompatibilityRegistry,
    previous: CompatibilityRegistry | None,
    target: Version,
    repo_root: Path,
) -> None:
    """Protect retirement deadlines and the monotonic compatibility ID watermark."""
    if previous is None:
        return
    if current.next_id < previous.next_id:
        raise RegistryError(f"registry next_id decreased from {previous.next_id} to {current.next_id}.")
    previous_by_id = {item.item_id: item for item in previous}
    current_by_id = {item.item_id: item for item in current}
    reused_ids = [
        item.item_id
        for item in current
        if item.item_id not in previous_by_id and int(item.item_id.rsplit("-", 1)[1]) < previous.next_id
    ]
    if reused_ids:
        raise RegistryError(f"retired compatibility item ID cannot be reused: {reused_ids[0]}.")
    removed = [item for item in previous if item.item_id not in current_by_id]
    premature = [item for item in removed if target < item.transition_purged_in]
    if premature:
        item = premature[0]
        raise RegistryError(
            f"{item.item_id} was removed before transition_purged_in "
            f"{item.transition_purged_in}; target is {target}."
        )
    marker_matches = _marker_files(repo_root, (item.marker for item in removed))
    for item in removed:
        marker_files = marker_matches[item.marker]
        if marker_files:
            raise RegistryError(f"{item.item_id} was retired but its marker remains in {list(marker_files)}.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    for command in ("plan", "check"):
        action = commands.add_parser(command)
        action.add_argument("--release-version", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    registry_path = args.registry
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    try:
        registry = load_registry(registry_path, repo_root)
        if args.command == "validate":
            print(f"Compatibility registry is valid ({len(registry)} unfinished items).")
            return 0
        target = Version.parse(args.release_version, "--release-version")
        previous = load_previous_registry(repo_root, _current_version(repo_root))
        print("\n".join(release_plan(registry, target, previous)), flush=True)
        if args.command == "check":
            check_release(registry, target)
            check_registry_transition(registry, previous, target, repo_root)
            print("Compatibility release gate passed.")
        return 0
    except RegistryError as exc:
        print(f"compatibility: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

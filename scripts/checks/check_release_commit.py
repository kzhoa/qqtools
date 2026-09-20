#!/usr/bin/env python3
"""Classify and validate owner-only release metadata commits."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path

try:
    from scripts.checks.check_compatibility_registry import RegistryError, Version, check_release, load_registry
except ModuleNotFoundError:  # Direct script execution adds scripts/ rather than the repo root.
    from checks.check_compatibility_registry import RegistryError, Version, check_release, load_registry


REPO_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATH = Path("src/qqtools/version.py")
CHANGELOG_PATH = Path("CHANGELOG.md")
RELEASE_PATHS = frozenset((CHANGELOG_PATH.as_posix(), VERSION_PATH.as_posix()))
VERSION_ASSIGNMENT = "__version__"
CHANGELOG_HEADING = re.compile(r"^## v(?P<version>\d+\.\d+\.\d+)\s*$", re.MULTILINE)


class ReleaseCommitError(RuntimeError):
    """Raised when a release candidate violates the metadata contract."""


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a Git command in the configured repository and return its result."""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        command = " ".join(("git", *args))
        raise ReleaseCommitError(f"{command} failed: {details}")
    return result


def _git_text(*args: str) -> str:
    return _run_git(*args).stdout


def _resolve_ref(ref: str, *, label: str) -> str:
    """Resolve a ref to a commit SHA, rejecting zero and invalid refs."""
    value = ref.strip()
    if not value or set(value) == {"0"}:
        raise ReleaseCommitError(f"{label} is zero or empty; a valid commit ref is required.")
    try:
        resolved = _git_text("rev-parse", "--verify", "--end-of-options", value).strip()
        commit = _git_text("rev-parse", "--verify", "--end-of-options", f"{resolved}^{{commit}}").strip()
    except ReleaseCommitError as exc:
        raise ReleaseCommitError(f"{label} {value!r} is not a valid commit ref: {exc}") from exc
    if not commit:
        raise ReleaseCommitError(f"{label} {value!r} did not resolve to a commit.")
    return commit


def _require_ancestor(base: str, head: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, head],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseCommitError(f"base ref {base} is not an ancestor of head ref {head}.")


def _changed_paths(base: str, head: str) -> tuple[str, ...]:
    """Return paths changed between two commits."""
    output = _git_text("diff", "--name-only", "-z", base, head, "--")
    return tuple(path for path in output.split("\0") if path)


def _commit_parent(commit: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{commit}^"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _commit_paths(commit: str) -> tuple[str, ...]:
    """Return paths changed by a commit against its first parent."""
    parent = _commit_parent(commit)
    if parent is None:
        output = _git_text("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "-z", commit)
    else:
        output = _git_text("diff", "--name-only", "-z", parent, commit, "--")
    return tuple(path for path in output.split("\0") if path)


def _read_git_file(commit: str, path: Path) -> str:
    """Read a UTF-8 file from a committed tree."""
    try:
        return _git_text("show", f"{commit}:{path.as_posix()}")
    except ReleaseCommitError as exc:
        raise ReleaseCommitError(f"{path} is missing or unreadable in commit {commit}: {exc}") from exc


def _parse_version(source: str, field: str) -> Version:
    try:
        module = ast.parse(source, filename=field)
    except SyntaxError as exc:
        raise ReleaseCommitError(f"{field} is not valid Python: {exc}") from exc
    for node in module.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == VERSION_ASSIGNMENT:
                try:
                    value = ast.literal_eval(node.value)
                    return Version.parse(value, f"{field}::__version__")
                except (ValueError, TypeError, RegistryError) as exc:
                    raise ReleaseCommitError(f"{field}::__version__ is invalid: {exc}") from exc
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == VERSION_ASSIGNMENT
        ):
            try:
                value = ast.literal_eval(node.value)
                return Version.parse(value, f"{field}::__version__")
            except (ValueError, TypeError, RegistryError) as exc:
                raise ReleaseCommitError(f"{field}::__version__ is invalid: {exc}") from exc
    raise ReleaseCommitError(f"Could not resolve __version__ from {field}.")


def _version_at(commit: str) -> Version:
    return _parse_version(_read_git_file(commit, VERSION_PATH), f"{VERSION_PATH} at {commit}")


def _changelog_body(changelog: str, target: Version, field: str) -> str:
    """Return the body of the exact target changelog section."""
    heading = f"## v{target}"
    match = next(
        (candidate for candidate in CHANGELOG_HEADING.finditer(changelog) if candidate.group("version") == str(target)),
        None,
    )
    if match is None:
        raise ReleaseCommitError(f"{field} is missing the required {heading} section.")
    section_start = match.end()
    next_heading = re.search(r"^##\s+", changelog[section_start:], re.MULTILINE)
    section_end = section_start + next_heading.start() if next_heading else len(changelog)
    body = changelog[section_start:section_end].strip()
    if not body:
        raise ReleaseCommitError(f"{field} section {heading} must contain release notes.")
    return body


def _check_worktree_metadata(head: str, target: Version) -> None:
    """Ensure checked-out metadata is the exact candidate tree and target version."""
    current_head = _resolve_ref("HEAD", label="checked-out HEAD")
    if current_head != head:
        raise ReleaseCommitError(f"checked-out HEAD {current_head} does not match requested head {head}.")

    version_path = REPO_ROOT / VERSION_PATH
    changelog_path = REPO_ROOT / CHANGELOG_PATH
    try:
        current_version_source = version_path.read_text(encoding="utf-8")
        current_changelog = changelog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseCommitError(f"release metadata is unavailable in the checked-out tree: {exc}") from exc
    current_version = _parse_version(current_version_source, str(VERSION_PATH))
    if current_version != target:
        raise ReleaseCommitError(f"checked-out source version {current_version} does not match release {target}.")
    _changelog_body(current_changelog, target, str(CHANGELOG_PATH))

    for path, current in ((VERSION_PATH, current_version_source), (CHANGELOG_PATH, current_changelog)):
        committed = _read_git_file(head, path)
        if current != committed:
            raise ReleaseCommitError(f"checked-out {path} does not match head commit {head}.")


def _validate_initial_release(commit: str, parent: str, target: Version, paths: tuple[str, ...]) -> None:
    if set(paths) != RELEASE_PATHS:
        raise ReleaseCommitError(
            f"initial release commit {commit} must change both {CHANGELOG_PATH} and {VERSION_PATH}; "
            f"changed: {sorted(paths)}"
        )
    parent_version = _version_at(parent)
    commit_version = _version_at(commit)
    if commit_version != target:
        raise ReleaseCommitError(f"release commit {commit} has version {commit_version}, expected {target}.")
    if commit_version <= parent_version:
        raise ReleaseCommitError(f"release {target} must bump the parent version {parent_version}.")
    _changelog_body(_read_git_file(commit, CHANGELOG_PATH), target, f"{CHANGELOG_PATH} at {commit}")


def _validate_release_series(base: str, head: str, target: Version) -> None:
    """Walk first-parent metadata commits until a valid bump for ``target`` is found."""
    _require_ancestor(base, head)
    first_parent_history = _git_text("rev-list", "--first-parent", head).splitlines()
    try:
        base_index = first_parent_history.index(base)
    except ValueError as exc:
        raise ReleaseCommitError(f"base ref {base} is not on the first-parent history of head ref {head}.") from exc

    # Validate every commit introduced by the push before looking behind its
    # base for an earlier release bump. This prevents a final valid bump from
    # hiding a malformed metadata-only commit in the same batched push.
    pushed_commits = first_parent_history[:base_index]
    historical_commits = first_parent_history[base_index:]

    found_pushed_bump = False
    for commit in reversed(pushed_commits):
        parent = _commit_parent(commit)
        if parent is None:
            raise ReleaseCommitError(f"release series for {target} has no initial parent bump.")
        paths = _commit_paths(commit)
        if not paths or not set(paths).issubset(RELEASE_PATHS):
            raise ReleaseCommitError(
                f"release series contains non-metadata commit {commit}; changed paths: {sorted(paths)}"
            )
        commit_version = _version_at(commit)
        parent_version = _version_at(parent)
        if commit_version == target and commit_version > parent_version:
            _validate_initial_release(commit, parent, target, paths)
            found_pushed_bump = True
        elif commit_version == target and commit_version == parent_version:
            continue
        else:
            raise ReleaseCommitError(
                f"release series commit {commit} has version {commit_version}; expected a correction for {target} "
                "or an initial version bump."
            )
    if found_pushed_bump:
        return

    for commit in historical_commits:
        parent = _commit_parent(commit)
        if parent is None:
            raise ReleaseCommitError(f"release series for {target} has no initial parent bump.")
        paths = _commit_paths(commit)
        if not paths or not set(paths).issubset(RELEASE_PATHS):
            raise ReleaseCommitError(
                f"release series contains non-metadata commit {commit}; changed paths: {sorted(paths)}"
            )
        commit_version = _version_at(commit)
        parent_version = _version_at(parent)
        if commit_version == target and commit_version > parent_version:
            _validate_initial_release(commit, parent, target, paths)
            return
        if commit_version != target or commit_version != parent_version:
            raise ReleaseCommitError(
                f"release series commit {commit} has version {commit_version}; expected a correction for {target} "
                "or an initial version bump."
            )
    raise ReleaseCommitError(f"release series does not reach an initial bump for {target}.")


def _check_compatibility(target: Version) -> None:
    registry_path = REPO_ROOT / "docs/spec/compatibility-registry.toml"
    try:
        registry = load_registry(registry_path, REPO_ROOT)
        check_release(registry, target)
    except RegistryError as exc:
        raise ReleaseCommitError(f"compatibility release gate failed: {exc}") from exc


def _imported_names(module: ast.Module) -> set[str]:
    """Return names statically re-exported by a stub module."""
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
    return names


def _lazy_exported_names(module: ast.Module) -> set[str]:
    """Return names registered through lazy-export declarations."""
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name) or node.value.func.id not in {"lazy_export", "_lazy_export"}:
            continue
        if len(node.value.args) < 2:
            raise ReleaseCommitError(f"{node.value.func.id} requires at least one export name.")
        for argument in node.value.args[1:]:
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                raise ReleaseCommitError(
                    f"{node.value.func.id} arguments must be literal export names for preflight validation."
                )
            names.add(argument.value)
    return names


def _lazy_imported_names(module: ast.Module) -> set[str]:
    """Return names assigned to LazyImport proxies."""
    names: set[str] = set()
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name) or value.func.id != "LazyImport":
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _runtime_exported_names(module: ast.Module) -> set[str]:
    """Return public names made available by a package initializer."""
    names = _imported_names(module) | _lazy_exported_names(module) | _lazy_imported_names(module)
    for getattr_node in (
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
    ):
        for node in ast.walk(getattr_node):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1 or len(node.comparators) != 1:
                continue
            if not isinstance(node.ops[0], ast.Eq):
                continue
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Name) and left.id == "name" and isinstance(right, ast.Constant):
                if isinstance(right.value, str):
                    names.add(right.value)
    return names


def _check_lazy_export_stubs() -> None:
    """Ensure package stubs match the public lazy-export surface."""
    packages = (
        ("qqtools", Path("src/qqtools/__init__.py")),
        ("qqtools.plugins.qpipeline", Path("src/qqtools/plugins/qpipeline/__init__.py")),
    )
    for package_name, relative_init in packages:
        init_path = REPO_ROOT / relative_init
        stub_path = init_path.with_suffix(".pyi")
        if not stub_path.is_file():
            raise ReleaseCommitError(f"Missing IDE stub for {package_name}: {stub_path.relative_to(REPO_ROOT)}")
        try:
            runtime_module = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
            stub_module = ast.parse(stub_path.read_text(encoding="utf-8"), filename=str(stub_path))
        except (OSError, SyntaxError) as exc:
            raise ReleaseCommitError(f"Could not inspect {package_name} runtime/stub exports: {exc}") from exc
        runtime_exports = _runtime_exported_names(runtime_module)
        lazy_exports = _lazy_exported_names(runtime_module) | _lazy_imported_names(runtime_module)
        stub_exports = _imported_names(stub_module)
        missing_lazy_exports = lazy_exports - stub_exports
        stale_stub_exports = stub_exports - runtime_exports
        if missing_lazy_exports or stale_stub_exports:
            details = []
            if missing_lazy_exports:
                details.append(f"missing lazy exports: {sorted(missing_lazy_exports)}")
            if stale_stub_exports:
                details.append(f"stale stub exports: {sorted(stale_stub_exports)}")
            raise ReleaseCommitError(f"{package_name} stub drift: {'; '.join(details)}")


def _classify(base_ref: str, head_ref: str, github_output: Path) -> int:
    base = _resolve_ref(base_ref, label="base ref")
    head = _resolve_ref(head_ref, label="head ref")
    _require_ancestor(base, head)
    changed = _changed_paths(base, head)
    profile = "release" if set(changed).issubset(RELEASE_PATHS) else "feature"
    output_path = github_output if github_output.is_absolute() else REPO_ROOT / github_output
    with output_path.open("a", encoding="utf-8") as output:
        output.write(f"profile={profile}\n")
        output.write(f"base_ref={base}\n")
    return 0


def _validate(base_ref: str, head_ref: str, actor: str) -> int:
    if actor != "kzhoa":
        raise ReleaseCommitError(f"release validation requires owner actor kzhoa; got {actor!r}.")
    base = _resolve_ref(base_ref, label="base ref")
    head = _resolve_ref(head_ref, label="head ref")
    _require_ancestor(base, head)
    target = _version_at(head)
    _check_worktree_metadata(head, target)
    _validate_release_series(base, head, target)
    _check_compatibility(target)
    _check_lazy_export_stubs()
    print(f"Release metadata validation passed for {target}.")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    classify = commands.add_parser("classify")
    classify.add_argument("--base-ref", required=True)
    classify.add_argument("--head-ref", required=True)
    classify.add_argument("--github-output", type=Path, required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--base-ref", required=True)
    validate.add_argument("--head-ref", required=True)
    validate.add_argument("--actor", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "classify":
        return _classify(args.base_ref, args.head_ref, args.github_output)
    return _validate(args.base_ref, args.head_ref, args.actor)


if __name__ == "__main__":
    raise SystemExit(main())

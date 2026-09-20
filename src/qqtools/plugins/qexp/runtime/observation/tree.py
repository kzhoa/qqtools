"""Bounded, ordered Task-ID indexes for local observation projections."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..records import TASK_ID_PATTERN
from ..store import atomic_replace
from ..work_budget import diagnostic_increment

_VERSION = 1
_MAX_NODE_BYTES = 64 * 1024
_MAX_LEAF_KEYS = 128
_MAX_CHILDREN = 65
_MAX_TASK_ID_LENGTH = 250

_LEAF_FIELDS = frozenset({"version", "prefix", "keys"})
_INTERNAL_FIELDS = frozenset({"version", "prefix", "children", "terminal"})


@dataclass(frozen=True, slots=True)
class TreeBatch:
    """A bounded ordered batch read from one observation index partition."""

    keys: tuple[str, ...]
    exhausted: bool
    pages: int
    bytes_read: int


@dataclass(frozen=True, slots=True)
class _Node:
    prefix: str
    keys: tuple[str, ...] | None = None
    children: tuple[str, ...] | None = None
    terminal: bool | None = None

    @property
    def is_leaf(self) -> bool:
        return self.keys is not None

    def to_record(self) -> dict[str, Any]:
        if self.is_leaf:
            return {"version": _VERSION, "prefix": self.prefix, "keys": list(self.keys or ())}
        return {
            "version": _VERSION,
            "prefix": self.prefix,
            "children": list(self.children or ()),
            "terminal": self.terminal,
        }


@dataclass(frozen=True, slots=True)
class _BuiltNode:
    node: _Node
    children: tuple["_BuiltNode", ...] = ()


@dataclass(slots=True)
class _ReadMeter:
    pages: int = 0
    bytes_read: int = 0


class _BudgetExceeded(Exception):
    """Internal signal for a candidate read that reached its I/O budget."""


@dataclass(frozen=True, slots=True)
class _DiscardResult:
    changed: bool
    empty: bool


def _partition_value(value: str | None, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{label} must be a string or None.")
    return value or None


def partition_key(phase: str | None, group: str | None) -> str:
    """Return the canonical digest used for a phase/group partition."""
    normalized = [_partition_value(phase, "phase"), _partition_value(group, "group")]
    encoded = json.dumps(normalized, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_task_id(value: str, label: str = "task_id") -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _MAX_TASK_ID_LENGTH
        or TASK_ID_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_' and '-', and be 1..{_MAX_TASK_ID_LENGTH} characters."
        )
    return value


def _validate_prefix(value: str, label: str = "prefix", *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or len(value) > _MAX_TASK_ID_LENGTH:
        raise ValueError(f"{label} is not a valid Task-ID prefix.")
    if not value:
        if allow_empty:
            return value
        raise ValueError(f"{label} must not be empty.")
    if TASK_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is not a valid Task-ID prefix.")
    return value


def _common_prefix(values: tuple[str, ...] | list[str]) -> str:
    if not values:
        return ""
    prefix = values[0]
    for value in values[1:]:
        limit = min(len(prefix), len(value))
        index = 0
        while index < limit and prefix[index] == value[index]:
            index += 1
        prefix = prefix[:index]
        if not prefix:
            break
    return prefix


def _no_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}.")
        result[key] = value
    return result


class IndexTree:
    """Persist an ordered radix tree for one observation filter partition."""

    def __init__(self, base: Path, phase: str | None = None, group: str | None = None) -> None:
        self._base = Path(base)
        self._phase = _partition_value(phase, "phase")
        self._group = _partition_value(group, "group")
        self._partition = self._base / "partitions" / partition_key(self._phase, self._group)

    @staticmethod
    def partition_key(phase: str | None, group: str | None) -> str:
        """Return the canonical phase/group partition digest."""
        return partition_key(phase, group)

    @property
    def root_path(self) -> Path:
        """Return this partition's root node path."""
        return self._partition / "root.json"

    def ensure_root(self) -> None:
        """Create a valid explicit empty root when this partition has none."""
        self._ensure_partition_directory()
        root = self._load_root()
        if root is None:
            self._persist_node(_Node(prefix="", keys=()))

    def add(self, task_id: str) -> None:
        """Insert one Task ID, preserving unique lexical ordering."""
        task_id = _validate_task_id(task_id)
        self._ensure_partition_directory()
        root = self._load_root()
        if root is None:
            self._persist_node(_Node(prefix="", keys=(task_id,)))
            return
        self._add_at(root, task_id, depth=0)

    def discard(self, task_id: str) -> None:
        """Remove one Task ID when present and prune empty nodes."""
        task_id = _validate_task_id(task_id)
        root = self._load_root()
        if root is None:
            return
        self._discard_at(root, task_id, depth=0)

    def contains(self, task_id: str) -> bool:
        """Return whether a Task ID is present using direct bounded descent."""
        task_id = _validate_task_id(task_id)
        node = self._load_root()
        if node is None:
            return False
        depth = 0
        while True:
            if depth > _MAX_TASK_ID_LENGTH + 1:
                raise ValueError("observation index depth exceeds its bounded limit.")
            if node.is_leaf:
                keys = node.keys or ()
                position = bisect_right(keys, task_id)
                return position > 0 and keys[position - 1] == task_id
            if task_id == node.prefix:
                return bool(node.terminal)
            if not task_id.startswith(node.prefix):
                return False
            child_prefix = self._matching_child(node, task_id)
            if child_prefix is None or not task_id.startswith(child_prefix):
                return False
            node = self._load_required(child_prefix)
            depth += 1

    def candidates(
        self,
        after: str | None,
        *,
        max_candidates: int = 256,
        max_pages: int = 1024,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> TreeBatch:
        """Return the next ordered IDs after ``after`` under explicit budgets."""
        if after is not None:
            after = _validate_task_id(after, "after")
        _validate_limit(max_candidates, "max_candidates")
        _validate_limit(max_pages, "max_pages")
        _validate_limit(max_bytes, "max_bytes")

        meter = _ReadMeter()
        try:
            root = self._load_root(meter=meter, max_pages=max_pages, max_bytes=max_bytes)
        except _BudgetExceeded as exc:
            raise ValueError(str(exc)) from exc
        if root is None:
            return TreeBatch(keys=(), exhausted=True, pages=meter.pages, bytes_read=meter.bytes_read)

        keys: list[str] = []
        try:
            exhausted = self._collect(
                root,
                after,
                keys,
                meter,
                max_candidates=max_candidates,
                max_pages=max_pages,
                max_bytes=max_bytes,
                depth=0,
            )
        except _BudgetExceeded as exc:
            if not keys:
                raise ValueError(str(exc)) from exc
            exhausted = False
        return TreeBatch(keys=tuple(keys), exhausted=exhausted, pages=meter.pages, bytes_read=meter.bytes_read)

    def _collect(
        self,
        node: _Node,
        after: str | None,
        output: list[str],
        meter: _ReadMeter,
        *,
        max_candidates: int,
        max_pages: int,
        max_bytes: int,
        depth: int,
    ) -> bool:
        if depth > _MAX_TASK_ID_LENGTH + 1:
            raise ValueError("observation index depth exceeds its bounded limit.")
        if after is not None and not _prefix_may_have_keys(node.prefix, after):
            return True
        if node.is_leaf:
            keys = node.keys or ()
            start = 0 if after is None else bisect_right(keys, after)
            for key in keys[start:]:
                output.append(key)
                if len(output) >= max_candidates:
                    return False
            return True

        if node.terminal and (after is None or node.prefix > after):
            output.append(node.prefix)
            if len(output) >= max_candidates:
                return False

        for child_prefix in node.children or ():
            if after is not None and not _prefix_may_have_keys(child_prefix, after):
                continue
            child = self._load_for_candidates(
                child_prefix,
                meter,
                max_pages=max_pages,
                max_bytes=max_bytes,
            )
            if not self._collect(
                child,
                after,
                output,
                meter,
                max_candidates=max_candidates,
                max_pages=max_pages,
                max_bytes=max_bytes,
                depth=depth + 1,
            ):
                return False
        return True

    def _add_at(self, node: _Node, task_id: str, *, depth: int) -> bool:
        if depth > _MAX_TASK_ID_LENGTH + 1:
            raise ValueError("observation index depth exceeds its bounded limit.")
        if node.is_leaf:
            keys = node.keys or ()
            position = bisect_right(keys, task_id)
            if position > 0 and keys[position - 1] == task_id:
                return False
            updated = list(keys)
            updated.insert(position, task_id)
            if len(updated) <= _MAX_LEAF_KEYS:
                self._persist_node(_Node(prefix=node.prefix, keys=tuple(updated)))
                return True
            built = self._build_split(node.prefix, tuple(updated))
            self._persist_built(built)
            return True

        if task_id == node.prefix:
            if node.terminal:
                return False
            self._persist_node(_Node(prefix=node.prefix, children=node.children or (), terminal=True))
            return True
        if not task_id.startswith(node.prefix):
            raise ValueError("observation index prefix invariant is violated.")

        child_prefix = self._matching_child(node, task_id)
        if child_prefix is None:
            if len(node.children or ()) >= _MAX_CHILDREN:
                raise ValueError("observation index has too many radix children.")
            child = _Node(prefix=task_id, keys=(task_id,))
            self._persist_node(child)
            children = tuple(sorted((*node.children, task_id))) if node.children else (task_id,)
            self._persist_node(_Node(prefix=node.prefix, children=children, terminal=bool(node.terminal)))
            return True

        if task_id.startswith(child_prefix):
            child = self._load_required(child_prefix)
            changed = self._add_at(child, task_id, depth=depth + 1)
            return changed

        common = _common_prefix((child_prefix, task_id))
        if common == node.prefix or common == child_prefix:
            raise ValueError("observation index child prefix invariant is violated.")
        bridge_children = [child_prefix]
        if task_id != common:
            new_leaf = _Node(prefix=task_id, keys=(task_id,))
            self._persist_node(new_leaf)
            bridge_children.append(task_id)
        bridge = _Node(prefix=common, children=tuple(sorted(bridge_children)), terminal=task_id == common)
        self._persist_node(bridge)
        children = tuple(common if item == child_prefix else item for item in node.children or ())
        self._persist_node(_Node(prefix=node.prefix, children=tuple(sorted(children)), terminal=bool(node.terminal)))
        return True

    def _discard_at(self, node: _Node, task_id: str, *, depth: int) -> _DiscardResult:
        if depth > _MAX_TASK_ID_LENGTH + 1:
            raise ValueError("observation index depth exceeds its bounded limit.")
        if node.is_leaf:
            keys = node.keys or ()
            position = bisect_right(keys, task_id)
            if position == 0 or keys[position - 1] != task_id:
                return _DiscardResult(changed=False, empty=False)
            remaining = keys[: position - 1] + keys[position:]
            if remaining:
                self._persist_node(_Node(prefix=node.prefix, keys=remaining))
                return _DiscardResult(changed=True, empty=False)
            self._unlink_node(node.prefix)
            return _DiscardResult(changed=True, empty=True)

        if task_id == node.prefix:
            if not node.terminal:
                return _DiscardResult(changed=False, empty=False)
            children = node.children or ()
            if children:
                self._persist_node(_Node(prefix=node.prefix, children=children, terminal=False))
                return _DiscardResult(changed=True, empty=False)
            self._unlink_node(node.prefix)
            return _DiscardResult(changed=True, empty=True)
        if not task_id.startswith(node.prefix):
            raise ValueError("observation index prefix invariant is violated.")

        child_prefix = self._matching_child(node, task_id)
        if child_prefix is None or not task_id.startswith(child_prefix):
            return _DiscardResult(changed=False, empty=False)
        child = self._load_required(child_prefix)
        result = self._discard_at(child, task_id, depth=depth + 1)
        if not result.changed:
            return result
        children = node.children or ()
        if result.empty:
            children = tuple(item for item in children if item != child_prefix)
        if children or node.terminal:
            self._persist_node(_Node(prefix=node.prefix, children=children, terminal=bool(node.terminal)))
            return _DiscardResult(changed=True, empty=False)
        self._unlink_node(node.prefix)
        return _DiscardResult(changed=True, empty=True)

    def _matching_child(self, node: _Node, task_id: str) -> str | None:
        children = node.children or ()
        if len(task_id) <= len(node.prefix):
            return None
        next_character = task_id[len(node.prefix)]
        for child_prefix in children:
            if child_prefix[len(node.prefix)] == next_character:
                return child_prefix
        return None

    def _build_split(self, prefix: str, keys: tuple[str, ...]) -> _BuiltNode:
        common = _common_prefix(keys)
        if not common.startswith(prefix):
            raise ValueError("observation index split prefix invariant is violated.")
        if common != prefix:
            child = self._build_branch(common, keys)
            return _BuiltNode(_Node(prefix=prefix, children=(common,), terminal=False), (child,))
        return self._build_branch(prefix, keys)

    def _build_branch(self, prefix: str, keys: tuple[str, ...]) -> _BuiltNode:
        if len(keys) <= _MAX_LEAF_KEYS:
            return _BuiltNode(_Node(prefix=_common_prefix(keys), keys=tuple(sorted(keys))))
        common = _common_prefix(keys)
        if common != prefix:
            return self._build_split(prefix, keys)

        terminal = prefix in keys
        groups: dict[str, list[str]] = {}
        for key in keys:
            if key == prefix:
                continue
            if len(key) <= len(prefix):
                raise ValueError("observation index split key is not below its prefix.")
            groups.setdefault(key[len(prefix)], []).append(key)
        built_children: list[_BuiltNode] = []
        for group in groups.values():
            group_keys = tuple(sorted(group))
            group_prefix = _common_prefix(group_keys)
            if len(group_keys) <= _MAX_LEAF_KEYS:
                built_children.append(_BuiltNode(_Node(prefix=group_prefix, keys=group_keys)))
            else:
                built_children.append(self._build_split(group_prefix, group_keys))
        built_children.sort(key=lambda item: item.node.prefix)
        child_prefixes = tuple(item.node.prefix for item in built_children)
        node = _Node(prefix=prefix, children=child_prefixes, terminal=terminal)
        return _BuiltNode(node, tuple(built_children))

    def _persist_built(self, built: _BuiltNode) -> None:
        for child in built.children:
            self._persist_built(child)
        self._persist_node(built.node)

    def _load_root(
        self,
        *,
        meter: _ReadMeter | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> _Node | None:
        if not self._partition_exists():
            return None
        return self._load_node(
            "",
            allow_missing=True,
            meter=meter,
            max_pages=max_pages,
            max_bytes=max_bytes,
        )

    def _load_required(self, prefix: str) -> _Node:
        node = self._load_node(prefix, allow_missing=False)
        if node is None:
            raise ValueError(f"observation index node is missing for prefix {prefix!r}.")
        return node

    def _load_for_candidates(
        self,
        prefix: str,
        meter: _ReadMeter,
        *,
        max_pages: int,
        max_bytes: int,
    ) -> _Node:
        node = self._load_node(
            prefix,
            allow_missing=False,
            meter=meter,
            max_pages=max_pages,
            max_bytes=max_bytes,
        )
        if node is None:
            raise ValueError(f"observation index node is missing for prefix {prefix!r}.")
        return node

    def _load_node(
        self,
        prefix: str,
        *,
        allow_missing: bool,
        meter: _ReadMeter | None = None,
        max_pages: int | None = None,
        max_bytes: int | None = None,
    ) -> _Node | None:
        path = self._node_path(prefix)
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            if allow_missing:
                return None
            raise ValueError(f"observation index node is missing for prefix {prefix!r}: {path}")
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"observation index node must not be a symlink: {path}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"observation index node must be a regular file: {path}")
        size = metadata.st_size
        if size > _MAX_NODE_BYTES:
            raise ValueError(f"observation index node exceeds its {_MAX_NODE_BYTES}-byte limit: {path}")
        if meter is not None:
            if max_pages is None or max_bytes is None:
                raise ValueError("candidate read budgets are incomplete.")
            if meter.pages >= max_pages or meter.bytes_read + size > max_bytes:
                raise _BudgetExceeded("observation index budget exhausted before yielding a candidate.")
        with path.open("rb") as handle:
            encoded = handle.read(size)
        if meter is not None:
            meter.pages += 1
            meter.bytes_read += len(encoded)
        diagnostic_increment("observation.index.pages")
        diagnostic_increment("observation.index.bytes", len(encoded))
        if len(encoded) != size:
            raise OSError(f"observation index node changed while being read: {path}")
        try:
            value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_no_duplicate_object_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid observation index JSON at {path}") from exc
        return self._validate_node(value, prefix, path)

    def _validate_node(self, value: Any, expected_prefix: str, path: Path) -> _Node:
        if not isinstance(value, dict):
            raise ValueError(f"invalid observation index node at {path}: expected an object.")
        fields = frozenset(value)
        if fields not in {_LEAF_FIELDS, _INTERNAL_FIELDS}:
            raise ValueError(f"invalid observation index node fields at {path}.")
        if type(value.get("version")) is not int or value["version"] != _VERSION:
            raise ValueError(f"invalid observation index node version at {path}.")
        prefix = value.get("prefix")
        _validate_prefix(prefix)
        if prefix != expected_prefix:
            raise ValueError(f"observation index node prefix does not match its path at {path}.")

        if fields == _LEAF_FIELDS:
            keys = value.get("keys")
            if not isinstance(keys, list) or len(keys) > _MAX_LEAF_KEYS:
                raise ValueError(f"invalid observation index leaf keys at {path}.")
            normalized = tuple(keys)
            for key in normalized:
                _validate_task_id(key, "index key")
                if not key.startswith(prefix):
                    raise ValueError(f"observation index key does not match its prefix at {path}.")
            if tuple(sorted(normalized)) != normalized or len(set(normalized)) != len(normalized):
                raise ValueError(f"observation index leaf keys are not sorted and unique at {path}.")
            if not normalized and prefix:
                raise ValueError(f"non-root observation index leaves must contain a key at {path}.")
            return _Node(prefix=prefix, keys=normalized)

        children = value.get("children")
        terminal = value.get("terminal")
        if not isinstance(children, list) or len(children) > _MAX_CHILDREN or type(terminal) is not bool:
            raise ValueError(f"invalid observation index internal node at {path}.")
        if terminal and not prefix:
            raise ValueError(f"the root observation index node cannot be terminal at {path}.")
        normalized_children = tuple(children)
        for child in normalized_children:
            _validate_prefix(child, "child prefix", allow_empty=False)
            if not child.startswith(prefix) or len(child) <= len(prefix):
                raise ValueError(f"observation index child does not extend its parent at {path}.")
        if tuple(sorted(normalized_children)) != normalized_children or len(set(normalized_children)) != len(
            normalized_children
        ):
            raise ValueError(f"observation index children are not sorted and unique at {path}.")
        if len({child[len(prefix)] for child in normalized_children}) != len(normalized_children):
            raise ValueError(f"observation index children do not have distinct radix characters at {path}.")
        if not normalized_children and not terminal:
            raise ValueError(f"empty non-terminal observation index node at {path}.")
        return _Node(prefix=prefix, children=normalized_children, terminal=terminal)

    def _persist_node(self, node: _Node) -> None:
        record = node.to_record()
        encoded_size = len(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
        if encoded_size > _MAX_NODE_BYTES:
            raise ValueError(f"observation index node exceeds its {_MAX_NODE_BYTES}-byte limit.")
        path = self._node_path(node.prefix)
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            metadata = None
        if metadata is not None and (stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode)):
            raise ValueError(f"observation index node must be a regular file: {path}")
        atomic_replace(path, record)

    def _unlink_node(self, prefix: str) -> None:
        path = self._node_path(prefix)
        try:
            metadata = os.lstat(path)
        except FileNotFoundError as exc:
            raise ValueError(f"observation index node is missing for prefix {prefix!r}: {path}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"observation index node must be a regular file: {path}")
        os.unlink(path)
        _fsync_directory(path.parent)

    def _node_path(self, prefix: str) -> Path:
        if not prefix:
            return self._partition / "root.json"
        digest = hashlib.sha256(prefix.encode("ascii")).hexdigest()
        return self._partition / f"{digest}.json"

    def _partition_exists(self) -> bool:
        try:
            metadata = os.lstat(self._partition)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"observation index partition must not be a symlink: {self._partition}")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"observation index partition must be a directory: {self._partition}")
        return True

    def _ensure_partition_directory(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)
        _ensure_directory(self._base)
        partitions = self._base / "partitions"
        created_partitions = _ensure_directory(partitions)
        if created_partitions:
            _fsync_directory(self._base)
        created_partition = _ensure_directory(self._partition)
        if created_partition:
            _fsync_directory(partitions)


def _validate_limit(value: int, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")


def _prefix_may_have_keys(prefix: str, after: str) -> bool:
    return prefix > after or after.startswith(prefix)


def _ensure_directory(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        path.mkdir()
        return True
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"observation index directory must be a real directory: {path}")
    return False


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

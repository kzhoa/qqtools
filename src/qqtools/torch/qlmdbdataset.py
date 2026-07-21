from __future__ import annotations

import bisect
import copy
import multiprocessing
import os
import pickle
import uuid
from collections.abc import Callable
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch

# PyTorch declares filelock as a runtime dependency, so it is available with torch.
from filelock import FileLock

from ..qimport import LazyImport
from .ddp.qbalancedsampler import BalancedBatchSampler
from .qbalanceddataset import _BalancedDatasetProvider
from .qdataset import qDictDataloader, qDictDataset

lmdb = LazyImport("lmdb")
tqdm = LazyImport("tqdm", "tqdm")

__all__ = ["qLmdbDataset", "qLmdbDatasetBase"]


class _FileLockWriteGuard:
    """Serialize idempotent artifact writes across processes using a filesystem lock.

    The readiness predicate is checked before and after locking. The writer must make its
    outputs atomically visible before the predicate can return ``True``.
    """

    def __init__(
        self,
        lock_path: str | Path,
        is_ready: Callable[[], bool],
        *,
        timeout: float = -1,
    ) -> None:
        self.lock_path = Path(lock_path)
        self.is_ready = is_ready
        self.timeout = float(timeout)

    def ensure(self, writer: Callable[[], Any]) -> bool:
        """Run ``writer`` once and return whether this process performed the write."""
        if self.is_ready():
            return False

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path), timeout=self.timeout):
            if self.is_ready():
                return False
            writer()
            if not self.is_ready():
                raise RuntimeError(
                    "Guarded writer completed, but the readiness check is still false: "
                    f"{self.lock_path}"
                )
        return True


class qLmdbDatasetBase(qDictDataset):
    """Generic read-only dataset for one or more LMDB shards.

    The default storage contract matches the production datasets extracted into this class:
    samples use contiguous ASCII integer keys (``b"0"``, ``b"1"``, ...), and an optional
    pickled integer at ``b"length"`` records the shard length. Subclasses normally only declare
    :attr:`lmdb_files`; custom payload formats can override :meth:`parse_value`.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        sort_lmdb_files: bool = True,
    ) -> None:
        self.root = Path(root).expanduser() if root is not None else None
        self.sort_lmdb_files = bool(sort_lmdb_files)
        self._source_lmdb_paths = tuple(self._resolve_lmdb_paths())
        self._lmdb_paths = self._source_lmdb_paths
        self._shard_lengths: tuple[int, ...] | None = None
        self._cumulative_sizes: tuple[int, ...] | None = None
        self._environments: list[Any] | None = None
        self._transactions: list[Any] | None = None
        self._storage_pid: int | None = None
        super().__init__(root=self.root)

    @property
    def lmdb_files(self) -> list[str | Path]:
        """Return dataset-local LMDB paths relative to ``root``."""
        raise NotImplementedError(
            "`lmdb_files` is not defined. Override `lmdb_files` or `lmdb_paths()`."
        )

    def lmdb_paths(self) -> list[Path]:
        """Resolve and optionally sort :attr:`lmdb_files` under ``root``."""
        if self.root is None:
            raise ValueError("`root` is required when using the default `lmdb_paths()`.")
        files = list(self.lmdb_files)
        if self.sort_lmdb_files:
            files = sorted(files, key=lambda item: str(item))
        return [self.root / Path(name) for name in files]

    @staticmethod
    def parse_value(blob: bytes) -> Any:
        """Decode a stored value with pickle.

        Args:
            blob: Raw LMDB value bytes.

        Returns:
            The decoded Python value.

        Raises:
            RuntimeError: If the payload is not pickle encoded.
        """
        try:
            return pickle.loads(blob)
        except Exception as exc:
            raise RuntimeError(
                "Failed to decode LMDB payload with the default pickle loader. "
                "Override `parse_value()` if this dataset uses another serialization format."
            ) from exc

    def get(self, idx: int) -> Any:
        """Read and decode one dataset sample."""
        return self.parse_value(self._load_raw_blob(idx))

    def len(self) -> int:
        """Return the total number of samples across all shards."""
        self._ensure_layout_loaded()
        return self._cumulative_sizes[-1] if self._cumulative_sizes else 0

    def close(self) -> None:
        """Release runtime LMDB transactions and environments owned by this process."""
        transactions = self._transactions or []
        environments = self._environments or []
        self._transactions = None
        self._environments = None
        self._storage_pid = None

        for transaction in transactions:
            try:
                transaction.abort()
            except Exception:
                pass
        for environment in environments:
            try:
                environment.close()
            except Exception:
                pass

    def __copy__(self) -> Self:
        dataset = type(self).__new__(type(self))
        dataset.__dict__ = self.__dict__.copy()
        dataset._environments = None
        dataset._transactions = None
        dataset._storage_pid = None
        if hasattr(self, "_balance"):
            dataset._balance = copy.copy(self._balance)
            dataset._balance.host = dataset
            hook = dataset._balance._get_sample_cost
            if getattr(hook, "__self__", None) is self:
                dataset._balance._get_sample_cost = getattr(dataset, hook.__name__)
        if hasattr(self, "_effective_costs_cache"):
            dataset._effective_costs_cache = None
            dataset._effective_order_cache = None
        return dataset

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_environments"] = None
        state["_transactions"] = None
        state["_storage_pid"] = None
        return state

    def __del__(self) -> None:
        if hasattr(self, "_environments"):
            self.close()

    def _resolve_lmdb_paths(self) -> list[Path]:
        paths = [Path(path).expanduser().absolute() for path in self.lmdb_paths()]
        if not paths:
            raise ValueError("At least one LMDB path is required.")
        missing_paths = [path for path in paths if not path.exists()]
        if missing_paths:
            formatted = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Missing LMDB input(s): {formatted}")
        return paths

    def _ensure_layout_loaded(self) -> None:
        if self._shard_lengths is not None:
            return

        lengths = tuple(self._read_lmdb_length(path) for path in self._lmdb_paths)
        cumulative_sizes = []
        running_total = 0
        for length in lengths:
            running_total += length
            cumulative_sizes.append(running_total)
        self._shard_lengths = lengths
        self._cumulative_sizes = tuple(cumulative_sizes)

    def _read_lmdb_length(self, path: Path) -> int:
        environment = self._open_readonly_environment(path)
        try:
            with environment.begin(write=False) as transaction:
                raw_length = transaction.get(b"length")
                if raw_length is not None:
                    try:
                        decoded_length = pickle.loads(raw_length)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to decode LMDB length metadata in {path}"
                        ) from exc
                    if isinstance(decoded_length, (bool, np.bool_)) or not isinstance(
                        decoded_length,
                        (int, np.integer),
                    ):
                        raise RuntimeError(
                            f"LMDB length metadata must be an integer in {path}, "
                            f"got {decoded_length!r}"
                        )
                    length = int(decoded_length)
                else:
                    length = sum(1 for key, _ in transaction.cursor() if key.isdigit())
        finally:
            environment.close()

        if length < 0:
            raise RuntimeError(f"LMDB length must be non-negative, got {length} in {path}")
        return length

    def _ensure_storage_open(self) -> None:
        current_pid = os.getpid()
        if self._storage_pid == current_pid and self._transactions is not None:
            return
        if self._transactions is not None or self._environments is not None:
            self.close()

        environments = []
        transactions = []
        try:
            for path in self._lmdb_paths:
                environment = self._open_readonly_environment(path)
                environments.append(environment)
                transactions.append(environment.begin(write=False))
        except Exception:
            for environment in environments:
                environment.close()
            raise

        self._environments = environments
        self._transactions = transactions
        self._storage_pid = current_pid

    @staticmethod
    def _open_readonly_environment(path: Path):
        return lmdb.open(
            str(path),
            subdir=path.is_dir(),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=128,
        )

    def _resolve_true_idx(self, idx: int) -> tuple[int, int]:
        self._ensure_layout_loaded()
        total = self._cumulative_sizes[-1] if self._cumulative_sizes else 0
        idx = int(idx)
        if idx < 0:
            idx += total
        if idx < 0 or idx >= total:
            raise IndexError(f"Sample index out of range: {idx} for dataset of length {total}")

        shard_idx = bisect.bisect_right(self._cumulative_sizes, idx)
        shard_start = 0 if shard_idx == 0 else self._cumulative_sizes[shard_idx - 1]
        return shard_idx, idx - shard_start

    def _load_raw_blob(self, idx: int) -> bytes:
        shard_idx, local_idx = self._resolve_true_idx(idx)
        self._ensure_storage_open()
        key = str(local_idx).encode("ascii")
        blob = self._transactions[shard_idx].get(key)
        if blob is None:
            path = self._lmdb_paths[shard_idx]
            raise IndexError(
                f"Missing LMDB sample key {key!r} at local index {local_idx} in {path}"
            )
        return blob

    def _set_lmdb_paths(self, paths: tuple[Path, ...]) -> None:
        self.close()
        self._lmdb_paths = paths
        self._shard_lengths = None
        self._cumulative_sizes = None


class qLmdbDataset(qLmdbDatasetBase):
    """LMDB dataset with optional balance assets and rewritten materialization.

    Implement :attr:`lmdb_files` for plain reading. When balance or rewrite is enabled, also
    implement ``get_sample_cost(sample)`` on the subclass.
    """

    rewrite_file_name = "balance_rewrite.lmdb"

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        is_graph: bool = False,
        enable_balance: bool = False,
        enable_rewrite: bool = False,
        balance_seed: int = 0,
        balance_strategy: str = "v3",
        sort_lmdb_files: bool = True,
    ) -> None:
        self.root = Path(root).expanduser() if root is not None else None
        self.is_graph = bool(is_graph)
        self.enable_balance = bool(enable_balance)
        self.enable_rewrite = bool(enable_rewrite)
        self.balance_seed = int(balance_seed)
        self.balance_strategy = str(balance_strategy)
        self._effective_costs_cache: np.ndarray | None = None
        self._effective_order_cache: np.ndarray | None = None
        self._balance = _BalancedDatasetProvider(
            host=self,
            enabled=(self.enable_balance or self.enable_rewrite),
            is_graph=self.is_graph,
            balance_seed=self.balance_seed,
            balance_strategy=self.balance_strategy,
        )
        super().__init__(root=self.root, sort_lmdb_files=sort_lmdb_files)
        if self.enable_rewrite and self.rewrite_path.exists():
            self._activate_rewrite()

    @property
    def processed_file_names(self) -> list[str]:
        names = list(self._balance.processed_file_names())
        if self.enable_rewrite:
            names.append(self.rewrite_file_name)
        return names

    @property
    def rewrite_path(self) -> Path:
        if self.processed_dir is None:
            raise RuntimeError("LMDB rewrite requires the dataset to define `root`.")
        return Path(self.processed_dir) / self.rewrite_file_name

    @property
    def _process_lock_path(self) -> Path:
        if self.processed_dir is None:
            raise RuntimeError("LMDB artifact processing requires the dataset to define `root`.")
        return Path(self.processed_dir) / ".qlmdbdataset.process.lock"

    @property
    def sample_costs(self) -> np.ndarray:
        """Return read-only costs aligned with the effective dataset index space."""
        if self._effective_costs_cache is None:
            costs = self._balance.sample_costs()
            if self._is_rewrite_active:
                costs = np.ascontiguousarray(costs[self._balance.sample_order()])
            if self._indices is not None:
                indices = np.asarray(list(self._indices), dtype=np.int64)
                costs = np.ascontiguousarray(costs[indices])
            costs.setflags(write=False)
            self._effective_costs_cache = costs
        return self._effective_costs_cache

    @property
    def sample_order(self) -> np.ndarray:
        """Return the read-only stable order for the effective dataset index space."""
        if self._effective_order_cache is None:
            if self._is_rewrite_active:
                materialization_order = self._balance.sample_order()
                full_order = np.arange(materialization_order.shape[0], dtype=np.int64)
            else:
                full_order = self._balance.sample_order()
            if self._indices is None:
                order = full_order
            else:
                indices = np.asarray(list(self._indices), dtype=np.int64)
                global_positions = np.empty(full_order.shape[0], dtype=np.int64)
                global_positions[full_order] = np.arange(full_order.shape[0], dtype=np.int64)
                order = np.argsort(global_positions[indices], kind="stable").astype(np.int64)
                order = np.ascontiguousarray(order)
            order.setflags(write=False)
            self._effective_order_cache = order
        return self._effective_order_cache

    def process(self) -> None:
        guard = _FileLockWriteGuard(
            self._process_lock_path,
            self.processed_files_exist,
        )
        guard.ensure(self._write_processed_artifacts)

    def _write_processed_artifacts(self) -> None:
        self._balance.process()
        if self.enable_rewrite:
            self._materialize_rewrite_if_needed()

    def to_dataloader(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        multiprocessing_context: str | BaseContext | None = None,
        collate_fn: Callable[[list[Any]], Any] | None = None,
        is_graph: bool | None = None,
    ) -> torch.utils.data.DataLoader:
        """Build a plain or balance-aware dataloader.

        Args:
            batch_size: Number of samples assigned to each rank-local batch.
            shuffle: Whether to generate a seed- and epoch-dependent balanced order.
            drop_last: Whether to drop an incomplete global batch window.
            num_workers: Number of dataloader worker processes.
            pin_memory: Whether dataloader workers should pin returned tensors.
            persistent_workers: Whether workers should remain alive between epochs.
            multiprocessing_context: Optional explicit worker start context. When omitted and
                workers are enabled, ``forkserver`` is preferred with ``spawn`` as fallback.
            collate_fn: Optional caller-provided batch collation function.
            is_graph: Optional per-loader override for graph collation.

        Returns:
            A configured PyTorch dataloader.

        Raises:
            ValueError: If batch or worker settings are invalid.
        """
        batch_size = int(batch_size)
        num_workers = int(num_workers)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")
        if persistent_workers and num_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")
        worker_context = self._resolve_multiprocessing_context(
            num_workers,
            multiprocessing_context,
        )

        graph_mode = self.is_graph if is_graph is None else bool(is_graph)
        if not self._balance.enabled:
            return qDictDataloader(
                dataset=self,
                batch_size=batch_size,
                shuffle=bool(shuffle),
                drop_last=bool(drop_last),
                num_workers=num_workers,
                pin_memory=bool(pin_memory),
                persistent_workers=bool(persistent_workers),
                multiprocessing_context=worker_context,
                collate_fn=collate_fn,
                is_graph=graph_mode,
            )

        batch_sampler = BalancedBatchSampler(
            self.sample_costs,
            batch_size=batch_size,
            shuffle=bool(shuffle),
            seed=self.balance_seed,
            drop_last=bool(drop_last),
            sample_order=None if shuffle else self.sample_order,
            strategy=self.balance_strategy,
        )
        return qDictDataloader(
            dataset=self,
            batch_size=None,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
            persistent_workers=bool(persistent_workers),
            multiprocessing_context=worker_context,
            collate_fn=collate_fn,
            is_graph=graph_mode,
        )

    @staticmethod
    def _resolve_multiprocessing_context(
        num_workers: int,
        context: str | BaseContext | None,
    ) -> str | BaseContext | None:
        if num_workers == 0:
            if context is not None:
                raise ValueError(
                    "multiprocessing_context requires num_workers > 0"
                )
            return None
        if context is not None:
            return context

        available_methods = multiprocessing.get_all_start_methods()
        if "forkserver" in available_methods:
            return "forkserver"
        if "spawn" in available_methods:
            return "spawn"
        return None

    @property
    def _is_rewrite_active(self) -> bool:
        return self.enable_rewrite and self._lmdb_paths == (self.rewrite_path,)

    def _materialize_rewrite_if_needed(self) -> None:
        if self.rewrite_path.exists():
            self._activate_rewrite()
            return

        order = self._balance.sample_order()
        target_path = self.rewrite_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = target_path.with_name(
            f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        temporary_lock_path = Path(f"{temporary_path}-lock")
        map_size = self._estimate_rewrite_map_size()
        environment = None
        transaction = None
        try:
            environment = lmdb.open(
                str(temporary_path),
                subdir=False,
                map_size=map_size,
                readonly=False,
                lock=True,
                readahead=False,
                meminit=False,
                max_readers=1,
            )
            transaction = environment.begin(write=True)
            progress = tqdm(
                enumerate(order),
                total=int(order.shape[0]),
                desc="Rewrite LMDB",
                unit="sample",
                disable=None,
            )
            for new_idx, source_idx in progress:
                transaction.put(
                    str(new_idx).encode("ascii"),
                    self._load_raw_blob(int(source_idx)),
                )
                if (new_idx + 1) % 1000 == 0:
                    transaction.commit()
                    transaction = environment.begin(write=True)
            transaction.put(b"length", pickle.dumps(int(order.shape[0])))
            transaction.commit()
            transaction = None
            environment.sync()
            environment.close()
            environment = None
            self.close()

            if target_path.exists():
                raise FileExistsError(f"LMDB rewrite already exists: {target_path}")
            temporary_path.replace(target_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to materialize rewritten LMDB at {target_path}") from exc
        finally:
            if transaction is not None:
                transaction.abort()
            if environment is not None:
                environment.close()
            self.close()
            temporary_path.unlink(missing_ok=True)
            temporary_lock_path.unlink(missing_ok=True)

        self._activate_rewrite()

    def _estimate_rewrite_map_size(self) -> int:
        source_size = 0
        for path in self._source_lmdb_paths:
            data_path = path / "data.mdb" if path.is_dir() else path
            source_size += data_path.stat().st_size
        return max(1 << 30, int(source_size * 1.5) + (64 << 20))

    def _activate_rewrite(self) -> None:
        self._set_lmdb_paths((self.rewrite_path,))
        self._effective_costs_cache = None
        self._effective_order_cache = None

import csv
import json
import os
import queue
import tempfile
import time
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Union

from ..events import ProgressEventContext
from ..events.types import _EvalEndInternalContext
from .common import _is_periodic_trigger
from .types import RunConfig


RowAdapter = Callable[[Mapping[str, Any]], Dict[str, Any]]


class SheetLogger:
    """Append structured rows to CSV or JSONL, expanding CSV schema as new keys arrive.

    ``columns`` supplies the initial CSV schema.  It is not a whitelist: a newly observed
    key is appended to the active schema and all existing rows are atomically rewritten with
    an empty value for that column.  ``close()`` commits an empty CSV header when no row was
    written; ``abort()`` commits rows already accepted by ``write()`` without making an
    otherwise empty-run commit.
    """

    def __init__(
        self,
        file_path: Union[Path, str],
        columns: List[str],
        format: str = "csv",
        max_size: int = 10 * 1024 * 1024,
        buffer_size: int = 1,
        async_write: bool = False,
        recover: bool = True,
        row_adapter: Optional[RowAdapter] = None,
    ) -> None:
        self.file_path = Path(file_path).resolve()
        self.columns = self._validate_columns(columns)
        self.format = "jsonl" if format.lower() == "json" else format.lower()
        if self.format not in {"csv", "jsonl"}:
            raise ValueError("format must be 'csv', 'json', or 'jsonl'")

        self.max_size = max_size
        self.buffer_size = max(1, buffer_size)
        self.async_write = async_write
        self.recover = recover
        self.row_adapter = row_adapter
        self._lock = Lock()
        self._state_lock = Lock()
        self._failure: Optional[BaseException] = None
        self._closed = False
        self._aborted = False
        self._has_committed_row = False

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.format == "csv" and recover and self.file_path.exists() and self.file_path.stat().st_size:
            self._recover_csv()

        if async_write:
            self._queue: queue.Queue[Optional[Mapping[str, Any]]] = queue.Queue()
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()

    @staticmethod
    def _validate_columns(columns: List[str]) -> List[str]:
        if columns is None:
            raise ValueError("columns is required")
        result = list(columns)
        if any(not isinstance(column, str) or not column for column in result):
            raise ValueError("columns must contain non-empty strings")
        if len(set(result)) != len(result):
            raise ValueError("columns must be unique")
        return result

    def _raise_if_unavailable(self) -> None:
        if self._failure is not None:
            raise RuntimeError("SheetLogger is in a failed state") from self._failure
        if self._closed or self._aborted:
            raise RuntimeError("SheetLogger is closed")

    def _set_failure(self, error: BaseException) -> None:
        if self._failure is None:
            self._failure = error

    def write(self, data: Mapping[str, Any]) -> None:
        """Persist one row, or queue it for serial asynchronous persistence."""
        row = dict(data)
        if self.async_write:
            with self._state_lock:
                self._raise_if_unavailable()
                self._queue.put(row)
            return
        with self._state_lock:
            self._raise_if_unavailable()
        with self._lock:
            try:
                self._write_row(row)
            except BaseException as error:
                self._set_failure(error)
                raise

    def _adapt_row(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        return dict(self.row_adapter(row)) if self.row_adapter is not None else dict(row)

    def _write_row(self, raw_row: Mapping[str, Any]) -> None:
        row = self._adapt_row(raw_row)
        if self.format == "jsonl":
            mode = "a" if self.recover or self._has_committed_row else "w"
            with open(self.file_path, mode, encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
                file.flush()
            self._has_committed_row = True
            self._rotate_if_needed()
            return

        new_columns = [key for key in row if key not in self.columns]
        if not self.file_path.exists() or self.file_path.stat().st_size == 0 or (
            not self.recover and not self._has_committed_row
        ):
            self._replace_csv(self.columns + new_columns, pending_row=row, source_path=None)
        elif new_columns:
            self._replace_csv(self.columns + new_columns, pending_row=row, source_path=self.file_path)
        else:
            self._append_csv(row)
        self._has_committed_row = True
        self._rotate_if_needed()

    def _append_csv(self, row: Mapping[str, Any]) -> None:
        original_size = self.file_path.stat().st_size
        try:
            with open(self.file_path, "a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([row.get(column, "") for column in self.columns])
                file.flush()
        except BaseException as write_error:
            try:
                with open(self.file_path, "r+b") as file:
                    file.truncate(original_size)
            except BaseException as rollback_error:
                raise RuntimeError("CSV append and rollback both failed") from rollback_error
            raise write_error

    def _recover_csv(self) -> None:
        header, rows = self._read_csv(self.file_path)
        missing = [column for column in self.columns if column not in header]
        self.columns = header
        if missing:
            self._replace_csv(header + missing, source_path=self.file_path, known_rows=rows)

    @staticmethod
    def _read_csv(path: Path) -> tuple[List[str], List[List[str]]]:
        with open(path, newline="", encoding="utf-8") as file:
            rows = list(csv.reader(file))
        while rows and not any(rows[-1]):
            rows.pop()
        if not rows:
            raise ValueError(f"CSV file has no header: {path}")
        header = rows[0]
        if not header or any(not column for column in header) or len(set(header)) != len(header):
            raise ValueError(f"CSV header must contain unique, non-empty columns: {path}")
        data_rows = rows[1:]
        if any(len(row) != len(header) for row in data_rows if any(row)):
            raise ValueError(f"CSV row width does not match header: {path}")
        return header, [row for row in data_rows if any(row)]

    def _replace_csv(
        self,
        new_columns: List[str],
        pending_row: Optional[Mapping[str, Any]] = None,
        source_path: Optional[Path] = None,
        known_rows: Optional[List[List[str]]] = None,
    ) -> None:
        old_columns: List[str] = []
        old_rows: List[List[str]] = []
        if source_path is not None:
            old_columns, old_rows = self._read_csv(source_path) if known_rows is None else (self.columns, known_rows)
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{self.file_path.name}.", dir=self.file_path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(new_columns)
                for old_row in old_rows:
                    old_data = dict(zip(old_columns, old_row))
                    writer.writerow([old_data.get(column, "") for column in new_columns])
                if pending_row is not None:
                    writer.writerow([pending_row.get(column, "") for column in new_columns])
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.file_path)
            self.columns = new_columns
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def _materialize_empty_csv(self) -> None:
        if self.format == "csv" and not self._has_committed_row:
            self._replace_csv(self.columns, source_path=None)

    def _rotate_if_needed(self) -> None:
        if not self.file_path.exists() or self.file_path.stat().st_size <= self.max_size:
            return
        timestamp = time.strftime("%Y%m%d%H%M%S")
        candidate = self.file_path.with_name(f"{self.file_path.name}.{timestamp}")
        suffix = 1
        while candidate.exists():
            candidate = self.file_path.with_name(f"{self.file_path.name}.{timestamp}.{suffix}")
            suffix += 1
        os.replace(self.file_path, candidate)
        if self.format == "csv":
            self._replace_csv(self.columns, source_path=None)

    def _worker(self) -> None:
        while True:
            raw_row = self._queue.get()
            try:
                if raw_row is None:
                    if not self._aborted and self._failure is None:
                        with self._lock:
                            self._materialize_empty_csv()
                    return
                # ``abort()`` prevents new writes, but accepted queue items remain part of
                # the run and must be persisted before the worker exits.
                if self._failure is None:
                    with self._lock:
                        self._write_row(raw_row)
            except BaseException as error:
                self._set_failure(error)
            finally:
                self._queue.task_done()

    def _stop_async(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._thread.join()

    def close(self) -> None:
        """Commit all accepted rows and, for an empty CSV run, its initial header."""
        if self.async_write:
            with self._state_lock:
                if self._closed:
                    if self._failure is not None:
                        raise RuntimeError("SheetLogger close failed") from self._failure
                    return
                if self._aborted:
                    raise RuntimeError("SheetLogger was aborted")
                self._closed = True
            self._stop_async()
        else:
            with self._state_lock:
                if self._closed:
                    if self._failure is not None:
                        raise RuntimeError("SheetLogger close failed") from self._failure
                    return
                if self._aborted:
                    raise RuntimeError("SheetLogger was aborted")
                self._closed = True
            with self._lock:
                try:
                    self._materialize_empty_csv()
                except BaseException as error:
                    self._set_failure(error)
        if self._failure is not None:
            raise RuntimeError("SheetLogger close failed") from self._failure

    def abort(self) -> None:
        """Reject future rows while committing rows already accepted by ``write()``."""
        with self._state_lock:
            if self._closed or self._aborted:
                if self._failure is not None:
                    raise RuntimeError("SheetLogger abort failed") from self._failure
                return
            self._aborted = True
        if self.async_write:
            self._stop_async()
        if self._failure is not None:
            raise RuntimeError("SheetLogger abort failed") from self._failure


def adapt_qpipeline_metric_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate qpipeline metric names and normalize scalar library values."""
    adapted: Dict[str, Any] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key or key != key.strip() or "\x00" in key or "\n" in key or "\r" in key:
            raise ValueError(f"Invalid metric key: {key!r}")
        if isinstance(value, (Mapping, list, tuple, set)):
            raise TypeError(f"Metric {key!r} must be a scalar, string, or None")
        if hasattr(value, "numel") and value.numel() != 1:
            raise TypeError(f"Metric {key!r} must be scalar")
        if hasattr(value, "shape") and getattr(value, "shape", ()) not in ((), None) and hasattr(value, "size"):
            if value.size != 1:
                raise TypeError(f"Metric {key!r} must be scalar")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            item = getattr(value, "item", None)
            if callable(item):
                value = item()
            else:
                raise TypeError(f"Metric {key!r} must be a scalar, string, or None")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"Metric {key!r} did not normalize to a scalar")
        adapted[key] = value
    return adapted


class SheetLoggerListener:
    """Translate qpipeline event metrics into SheetLogger rows."""

    _reserved_keys = {"epoch", "global_step"}

    def __init__(
        self,
        sheet_logger: SheetLogger,
        run_config: RunConfig,
        log_granularity: List[str],
        logger: Optional[Any] = None,
    ) -> None:
        self.sheet_logger = sheet_logger
        self.config = run_config
        self.log_granularity = log_granularity
        self.logger = logger

    def _warn(self, message: str) -> None:
        if self.logger is not None:
            self.logger.warning(message)
        else:
            warnings.warn(message)

    @staticmethod
    def _eval_metrics(context: _EvalEndInternalContext) -> Dict[str, Any]:
        evaluation = getattr(context, "evaluation", None)
        flat_metrics = getattr(evaluation, "flat_metrics", None)
        if flat_metrics is not None:
            return dict(flat_metrics)
        return dict(getattr(context, "eval_results", None) or {})

    def _prepare_data(self, context: Any, mode: str) -> Dict[str, Any]:
        state = context.runner.run_state
        data = {"epoch": state.epoch, "global_step": state.global_step}
        if mode == "eval":
            source_metrics = self._eval_metrics(context)
            if not source_metrics:
                for key in ("current_val_metric", "current_test_metric", "current_train_metric", "current_train_loss"):
                    value = getattr(state, key, None)
                    if value is not None:
                        source_metrics[key.replace("current_", "")] = value
        else:
            source_metrics = dict(getattr(context, "batch_metrics", None) or {})
            source_metrics = {
                key if key.startswith("train_") else f"train_{key}": value
                for key, value in source_metrics.items()
            }
        conflicts = self._reserved_keys.intersection(source_metrics)
        if conflicts:
            raise ValueError(f"{mode} metrics cannot override reserved fields: {sorted(conflicts)}")
        data.update(source_metrics)
        return data

    def on_eval_end(self, context: _EvalEndInternalContext) -> None:
        self.sheet_logger.write(self._prepare_data(context, mode="eval"))

    def on_train_batch_end(self, context: ProgressEventContext) -> None:
        if "eval" in self.log_granularity:
            is_epoch_end = context.batch_idx is not None and context.total_batches is not None and context.batch_idx == context.total_batches - 1
            if _is_periodic_trigger(
                run_mode=self.config.run_mode,
                interval=self.config.eval_interval,
                global_step=context.runner.run_state.global_step,
                epoch=context.runner.run_state.epoch,
                is_epoch_end=is_epoch_end,
            ):
                return
        self.sheet_logger.write(self._prepare_data(context, mode="batch"))

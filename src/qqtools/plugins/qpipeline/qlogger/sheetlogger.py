import csv
import json
import os
import queue
import warnings
from threading import Thread
from typing import Dict, List


class SheetLogger:
    def __init__(
        self,
        file_path: str,
        columns: List[str],
        format: str = "csv",  # "csv" or "json"
        max_size: int = 10 * 1024 * 1024,  # 10MB
        buffer_size: int = 1,
        async_write: bool = False,
    ):
        """
        :param file_path:   "logs/training.csv"
        :param columns:   ["epoch", "loss", "train_mae", "val_mae"]
        :param delimiter:
        :param buffer_size:
        """
        self.file_path = os.path.abspath(file_path)
        self.columns = columns
        self.format = format.lower()
        self.max_size = max_size
        self.buffer_size = buffer_size
        self.async_write = async_write

        assert columns is not None
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(file_path) and format == "csv":
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)

        if async_write:
            warnings.warn(" async_write is Depreciated")
            self._buffer = []
            self._queue = queue.Queue()
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()

    def write(self, data: Dict[str, float]):
        """write data to file"""

        extra_keys = set(data.keys()) - set(self.columns)
        if extra_keys:
            raise ValueError(f"Unexpected keys: {extra_keys}")

        if self.async_write:
            self._queue.put(data)
        else:
            self.sync_write_data(data)

    def sync_write_data(self, data: Dict[str, float]):
        """sync write data to file"""
        if self.format == "csv":
            with open(self.file_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([data.get(col, "") for col in self.columns])
        elif self.format == "json":
            with open(self.file_path, "a") as f:
                f.write(json.dumps(data) + "\n")

        self._rotate_if_needed()

    def _worker(self):
        """async write data to file"""
        while True:
            data = self._queue.get()
            self._buffer.append(data)
            if len(self._buffer) >= self.buffer_size:
                self._flush_buffer()
            self._queue.task_done()

    def _flush_buffer(self):
        """write buffer to file"""
        if not self._buffer:
            return

        if self.format == "csv":
            with open(self.file_path, "a", newline="") as f:
                writer = csv.writer(f)
                for data in self._buffer:
                    row = [data.get(col, "") for col in self.columns]
                    writer.writerow(row)
        elif self.format == "jsonl":
            with open(self.file_path, "a") as f:
                for data in self._buffer:
                    f.write(json.dumps(data) + "\n")

        self._buffer.clear()
        self._rotate_if_needed()

    def _rotate_if_needed(self):
        """"""
        if os.path.getsize(self.file_path) <= self.max_size:
            return

        base_path = self.file_path
        existing_files = []
        for f in os.listdir(os.path.dirname(base_path)):
            if f.startswith(os.path.basename(base_path) + "."):
                try:
                    # extract suffix (e.g. "training.csv.1" → 1)
                    num = int(f.split(".")[-1])
                    existing_files.append(num)
                except ValueError:
                    pass

        next_num = max(existing_files) + 1 if existing_files else 1

        for num in sorted(existing_files, reverse=True):
            old_name = f"{base_path}.{num}"
            new_name = f"{base_path}.{num + 1}"
            os.rename(old_name, new_name)

        os.rename(base_path, f"{base_path}.{next_num}")

        if self.format == "csv":
            with open(base_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)

    def close(self):
        if hasattr(self, "_queue"):
            self._queue.join()  # await all tasks
            if self._buffer:
                self._flush_buffer()

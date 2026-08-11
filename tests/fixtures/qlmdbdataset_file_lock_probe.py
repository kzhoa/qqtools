import multiprocessing
import pickle
import sys
import time
from pathlib import Path

import lmdb

from qqtools.torch.qlmdbdataset import _FileLockWriteGuard, qLmdbDataset


class _RewriteDataset(qLmdbDataset):
    @property
    def lmdb_files(self):
        return ["raw/data.lmdb"]

    def get_sample_cost(self, idx):
        return self.get(idx)["cost"]

    def _write_processed_artifacts(self) -> None:
        writer_log_path = Path(self.root) / "rewrite_writers.log"
        with writer_log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{multiprocessing.current_process().pid}\n")
        time.sleep(0.2)
        super()._write_processed_artifacts()


def _write_artifact(root: str, barrier) -> None:
    root_path = Path(root)
    artifact_path = root_path / "artifact.ready"
    writer_log_path = root_path / "writers.log"
    guard = _FileLockWriteGuard(
        root_path / "artifact.lock",
        artifact_path.exists,
    )

    def writer() -> None:
        with writer_log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{multiprocessing.current_process().pid}\n")
        time.sleep(0.2)
        artifact_path.write_text("ready", encoding="utf-8")

    guard.ensure(writer)
    barrier.wait()

    dataset = _RewriteDataset(root=root_path, enable_rewrite=True)
    assert sorted(dataset[idx]["id"] for idx in range(len(dataset))) == [0, 1]
    dataset.close()


def _prepare_lmdb(root: Path) -> None:
    path = root / "raw" / "data.lmdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = lmdb.open(str(path), subdir=False, map_size=1 << 24)
    try:
        with environment.begin(write=True) as transaction:
            transaction.put(b"0", pickle.dumps({"id": 0, "cost": 1.0}))
            transaction.put(b"1", pickle.dumps({"id": 1, "cost": 2.0}))
            transaction.put(b"length", pickle.dumps(2))
    finally:
        environment.close()


def main(start_method: str, root: str) -> None:
    context = multiprocessing.get_context(start_method)
    root_path = Path(root)
    _prepare_lmdb(root_path)
    barrier = context.Barrier(2)
    processes = [
        context.Process(target=_write_artifact, args=(root, barrier))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert root_path.joinpath("artifact.ready").read_text(encoding="utf-8") == "ready"
    writer_lines = root_path.joinpath("writers.log").read_text(encoding="utf-8").splitlines()
    assert len(writer_lines) == 1
    rewrite_writer_lines = (
        root_path.joinpath("rewrite_writers.log")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rewrite_writer_lines) == 1


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Share parent imports while preserving fresh workers for both process contracts."""

import multiprocessing
import sys

from qdataset_graph_worker_probe import main as check_graph_worker
from qlmdbdataset_file_lock_probe import main as check_file_lock


def main(start_method: str, root: str) -> None:
    if start_method == "forkserver":
        multiprocessing.set_forkserver_preload(["__main__", "torch"])
    check_graph_worker(start_method)
    check_file_lock(start_method, root)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

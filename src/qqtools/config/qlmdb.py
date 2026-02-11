from contextlib import contextmanager

from ..qimport import LazyImport

lmdb = LazyImport("lmdb")
tqdm = LazyImport("tqdm", "tqdm")

MAP_SIZE = 100 * 1024 * 1024 * 1024


@contextmanager
def open_lmdb(fpath, write=False, is_subdir=False, map_size=MAP_SIZE):
    """
    Context manager for opening an LMDB database.

    Args:
        fpath: The path to the LMDB database file.
        write: Whether to open the database in read-write mode.
        is_subdir: Whether the LMDB database is a subdirectory.
        map_size: The size of the memory map in bytes.

    Yields:
        An LMDB environment object.

    Raises:
        Exception: If an error occurs while closing the environment.
    """
    fpath = str(fpath)
    env = lmdb.open(
        fpath,
        readonly=not write,
        subdir=is_subdir,
        map_size=map_size,
        readahead=False,
    )

    try:
        yield env
    finally:
        env.close()


@contextmanager
def operate_lmdb(fpath, write=False, is_subdir=False, map_size=MAP_SIZE):
    """
    Context manager for operating on an LMDB database.

    Args:
        fpath: The path to the LMDB database file.
        write: Whether to open the database in read-write mode.
        map_size: The size of the memory map in bytes.

    Yields:
        An LMDB transaction object.

    Raises:
        Exception: If an error occurs while committing or aborting the transaction.
    """
    fpath = str(fpath)
    env = lmdb.open(
        fpath,
        readonly=not write,
        map_size=map_size,
        subdir=is_subdir,
    )
    txn = env.begin(write=write)
    try:
        yield txn
        if write:
            txn.commit()
    except Exception as e:
        if write:
            txn.abort()
        raise e
    finally:
        env.close()


def count_lmdb(fpath, is_subdir=False) -> int:
    """
    Count the number of entries in an LMDB database.

    Args:
        fpath: The path to the LMDB database file.
        is_subdir: Whether the LMDB database is a subdirectory.

    Returns:
        The number of entries in the LMDB database.
    """
    fpath = str(fpath)
    with lmdb.open(fpath, readonly=True, subdir=is_subdir) as env:
        return env.stat()["entries"]


def iter_lmdb(fpath, is_subdir=False, map_size=MAP_SIZE, progress=False):
    """
    Iterate over the key-value pairs in an LMDB database.

    Args:
        fpath: The path to the LMDB database file.
        is_subdir: Whether the LMDB database is a subdirectory.

    Yields:
        A tuple containing the key and value of each entry in the LMDB database.
    """

    with operate_lmdb(
        fpath,
        write=False,
        is_subdir=is_subdir,
        map_size=map_size,
    ) as txn:
        cursor = txn.cursor()
        items_generator = cursor.iternext()
        if progress:
            progress_bar = tqdm(items_generator, unit="item", leave=True)
            for k, v in progress_bar:
                yield k, v
        else:
            for k, v in txn.cursor():
                yield k, v

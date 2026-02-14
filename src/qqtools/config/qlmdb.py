"""
usage demo:

# iterate all key-value pairs in the LMDB database
for k,v in iter_lmdb('data.lmdb'):
    k = k.decode('utf-8')
    v = pickle.loads(v)
    print(k, v)

# auto commit after context exit
with qt.operate_lmdb('data.lmdb', write=True) as env:
    # write
    txn.put(b"key1", b"value1")

    # get
    print(env.get(b"key1"))

    # delete
    env.delete(b"key2")


# read only
with qt.operate_lmdb('data.lmdb', write=False) as txn:
    v = txn.get(b"key1")
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING

from ..qimport import LazyImport

if TYPE_CHECKING:
    import lmdb
    from tqdm import tqdm
else:
    lmdb = LazyImport("lmdb")
    tqdm = LazyImport("tqdm", "tqdm")

MAP_SIZE = 100 * 1024 * 1024 * 1024


class ENVProxy:
    """
    handle batch commit for LMDB transactions. Usage:
    with qt.operate_lmdb('data.lmdb', write=True) as env:
        # write
        env.put(b"key1", b"value1")
        env.put(b"key2", b"value2")

        # get
        print(env.get(b"key1"))

        # delete
        env.delete(b"key2")

        # commit manually (optional)
        env.commit()
    """

    def __init__(self, fpath, write=False, subdir=False, *args, **kwargs):
        self.fpath = fpath
        self.write = write
        self.subdir = subdir

        self.args = args
        self.kwargs = kwargs
        self.env = None
        self._tnx = None
        self._cnt = 0
        self._batch_size = 500

    def __enter__(self):
        self.env = lmdb.open(
            self.fpath,
            readonly=not self.write,
            subdir=self.subdir,
            *self.args,
            **self.kwargs,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.write and exc_type is None:
            self.commit()
        self.env.close()

    @property
    def tnx(self):
        if self._tnx is None:
            self._tnx = self.env.begin(write=self.write)
        return self._tnx

    def commit(self):
        if self._tnx is not None:
            self._tnx.commit()
            self._tnx = None
            self._cnt = 0

    def put(self, k, v):
        assert self.write is True, "ENVProxy is not opened in write mode"
        self.tnx.put(k, v)
        self._cnt += 1

        if self._cnt >= self._batch_size:
            self.commit()

    def get(self, k):
        return self.tnx.get(k)

    def delete(self, k):
        assert self.write is True, "ENVProxy is not opened in write mode"
        self.tnx.delete(k)
        self._cnt += 1

        if self._cnt >= self._batch_size:
            self.commit()


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


def operate_lmdb(fpath, write=False, is_subdir=False, map_size=MAP_SIZE) -> ENVProxy:
    """
    Proxy Context manager for operating on an LMDB database.

    Args:
        fpath: The path to the LMDB database file.
        write: Whether to open the database in read-write mode.
        map_size: The size of the memory map in bytes.

    Yields:
        A proxy of LMDB transaction object.
    """
    fpath = str(fpath)

    env = ENVProxy(
        fpath,
        write=write,
        subdir=is_subdir,
        map_size=map_size,
    )
    return env


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

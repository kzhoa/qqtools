"""Publish a completed regular file on the destination filesystem."""

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def atomic_output_file(destination: Path, *, overwrite: bool = True) -> Iterator[Path]:
    # Follow existing symlinks just as opening the destination for writing would.
    destination = destination.resolve()
    try:
        mode = destination.stat().st_mode
    except FileNotFoundError:
        mode = None
    else:
        if not overwrite:
            raise FileExistsError(destination)
        if not stat.S_ISREG(mode):
            raise OSError(f"Expected a regular file at {destination}")
        if not os.access(destination, os.W_OK):
            raise PermissionError(f"Destination is not writable: {destination}")

    with TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / "payload"
        yield staged
        if mode is not None:
            staged.chmod(stat.S_IMODE(mode) & 0o777)
        if overwrite:
            staged.replace(destination)
        else:
            # If another writer published first, leave its completed file in place.
            os.link(staged, destination)

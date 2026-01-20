import sys
from pathlib import Path
from typing import Union


def find_root(start: Union[str, Path], return_str=False, marker="pyproject.toml") -> Union[str, Path]:
    current = Path(start).absolute()
    while current != current.parent:
        if (current / marker).exists():
            if return_str:
                return str(current)
            else:
                return current
        current = current.parent
    raise FileNotFoundError(f"cannot find: {marker}")


def update_sys(path: str):
    assert isinstance(path, (str, Path))
    if path not in set(sys.path):
        sys.path.insert(0, path)

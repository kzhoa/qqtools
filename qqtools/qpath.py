from pathlib import Path
from typing import Union


def find_root(start: Union[str, Path], marker="pyproject.toml") -> str:
    current = Path(start).absolute()
    while current != current.parent:
        if (current / marker).exists():
            return str(current)
        current = current.parent
    raise FileNotFoundError(f"cannot find: {marker}")

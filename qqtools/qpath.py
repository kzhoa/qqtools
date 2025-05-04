from pathlib import Path
from typing import Union


def find_root(start: Union[str, Path], return_str=True, marker="pyproject.toml") -> Union[str, Path]:
    current = Path(start).absolute()
    while current != current.parent:
        if (current / marker).exists():
            if return_str:
                return str(current)
            else:
                return current
        current = current.parent
    raise FileNotFoundError(f"cannot find: {marker}")

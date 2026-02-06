try:
    import ujson as json
except ImportError:
    import json

from pathlib import Path
from typing import Any, Callable, Optional, Union


def save_json(
    obj,
    fpath,
    indent: int = 2,
    sort_keys=False,
    default_serializer=None,
    overwrite=True,
    verbose=True,
) -> bool:
    fpath = Path(fpath) if not isinstance(fpath, Path) else fpath

    if Path(fpath).exists() and not overwrite:
        if verbose:
            print(f"File already exists: {fpath}")
        return False

    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(
                obj,
                f,
                indent=indent,
                sort_keys=sort_keys,
                default=default_serializer,
            )
        return True
    except Exception as e:
        if verbose:
            print(f"Error saving json to {fpath}: {e}")
        return False


def load_json(
    fpath: Union[str, Path],
    verbose: bool = True,
    encoding: str = "utf-8",
) -> Any:
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")

    if fpath.stat().st_size == 0:
        if verbose:
            print(f"File is empty: {fpath}")
        return {}

    try:
        with open(fpath, "r", encoding=encoding) as f:
            data = json.load(f)
        return data

    except Exception as e:
        if verbose:
            print(f"Error loading json from {fpath}: {e}")
        raise e

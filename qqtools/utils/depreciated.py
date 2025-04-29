import warnings
from functools import wraps
from typing import Optional, Callable


def deprecated(
    reason: Optional[str] = None, version: Optional[str] = None, replacement: Optional[str] = None
) -> Callable:

    def decorator(obj):
        is_class = isinstance(obj, type)
        message = f"{'Class' if is_class else 'Function'} `{obj.__name__}` is deprecated"
        if version:
            message += f" since version {version}"
        if reason:
            message += f".  {reason}"
        if replacement:
            message += f", use `{replacement}` instead."

        @wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapper if not is_class else type(obj.__name__, (obj,), {"__doc__": message})

    return decorator

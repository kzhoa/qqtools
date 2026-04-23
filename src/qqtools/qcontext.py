import contextvars
from contextlib import contextmanager
from typing import Any

from .qdict import qDict

_MISSING = object()
_ctx_storage = contextvars.ContextVar("qt_global_ctx")


class _ContextManager:
    """
    Unified Context API: Acts as both a provider (via __call__)
    and a consumer (via attribute access).
    """

    # ================= Consumer API =================
    def _get_current_ctx(self) -> qDict:
        ctx = _ctx_storage.get(_MISSING)
        if ctx is _MISSING:
            ctx = qDict(allow_notexist=False)
            _ctx_storage.set(ctx)
        return ctx

    @property
    def current(self) -> qDict:
        """
        Provides read-only access to the current context as a standard dictionary.
        Usage: current_ctx = qt.ctx.current
        """
        return self._get_current_ctx()

    def __getattr__(self, name: str) -> Any:
        """
        Retrieves a value from the current context.
        Usage: value = qt.ctx.param_name
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        return self.current.__getattr__(name)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            super().__setattr__(name, value)
            return

        raise AttributeError(
            f"🛑 Direct assignment to `qt.ctx.{name}` is prohibited to prevent state leakage.\n"
            f"  👉 Standard Scope:  `with qt.ctx({name}={repr(value)}):`\n"
            f"  👉 In-place Update: `qt.ctx.set('{name}', {repr(value)})`"
        )

    def get(self, name: str, default: Any = None) -> Any:
        """
        Safely retrieves a value with a fallback default.
        """
        return self.current.get(name, default)

    def set(self, name: str, value: Any):
        """
        Sets a value in the current context for the duration of the context.
        Usage: qt.ctx.set('param_name', value)
        """
        # Create a new context dict based on the current one
        self.current.__setitem__(name, value)

    def __getitem__(self, key: str):
        return self.current.__getitem__(key)

    def __setitem__(self, key: str, value: Any):
        raise TypeError(
            f"🛑 Direct assignment via `qt.ctx[{repr(key)}]` is prohibited to prevent state leakage.\n"
            f"  👉 Standard Scope:  `with qt.ctx({key}={repr(value)}):`\n"
            f"  👉 In-place Update: `qt.ctx.set({repr(key)}, {repr(value)})`"
        )

    def reset(self):
        """
        Resets the context to an empty state.
        Usage: qt.ctx.reset()
        """
        _ctx_storage.set(qDict(allow_notexist=False))

    # ================= Provider API =================

    @contextmanager
    def __call__(self, d=None, /, **kwargs):
        """
        Enables usage as a context manager.
        Usage:
            with qt.ctx(key=value): ...

            with qt.ctx({'key': value}): ...

        Supports nesting: inner scopes merge with and override outer scopes.
        """
        if d is not None and kwargs:
            raise ValueError(
                "Conflicting arguments: "
                "Either pass a dictionary `d` OR keyword arguments, "
                "but not both.\n"
                "Example:\n"
                "  qt.ctx({'key': value}) \n"
                "  qt.ctx(key=value)      "
            )
        new_args = d if d is not None else kwargs
        current_state = self.current.copy()  # Get current context
        current_state.update(new_args)  # Merge with new arguments, new args take precedence
        token = _ctx_storage.set(qDict(current_state))
        try:
            yield
        finally:
            # Revert to the previous state regardless of errors
            _ctx_storage.reset(token)

    def to_dict(self) -> qDict:
        """
        Expose the current context as a standard dictionary.
        Useful for debugging or when a dict interface is needed.
        """
        return self.current.to_dict()

    def __repr__(self):
        return f"ContextProxy({self.current})"


# Singleton instance exposed to the package
ctx = _ContextManager()

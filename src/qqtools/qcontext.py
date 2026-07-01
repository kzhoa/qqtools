import contextvars
import inspect
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional, Type, Union

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
        Exposes the live qDict for the current scope.
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
            f"🛑 Direct assignment to `qt.ctx.{name}` is prohibited on the proxy object.\n"
            f"  👉 Standard Scope:  `with qt.ctx({name}={repr(value)}):`\n"
            f"  👉 In-place Update: `qt.ctx.set('{name}', {repr(value)})`\n"
            f"  👉 Live Context:    `qt.ctx.current[{repr(name)}] = {repr(value)}`"
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
            f"🛑 Direct assignment via `qt.ctx[{repr(key)}]` is prohibited on the proxy object.\n"
            f"  👉 Standard Scope:  `with qt.ctx({key}={repr(value)}):`\n"
            f"  👉 In-place Update: `qt.ctx.set({repr(key)}, {repr(value)})`\n"
            f"  👉 Live Context:    `qt.ctx.current[{repr(key)}] = {repr(value)}`"
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

        Design note:
        We intentionally preserve the live-context mutation backdoor via
        `qt.ctx.set(...)` / `qt.ctx.current[...] = ...` for advanced users who
        want full manual control inside the active scope. Scope exit restores
        the previous binding via ContextVar token reset, but mutable shared
        objects are not deep-copied. If a caller mutates a shared object in
        place and that mutation leaks outward, that is considered expected user
        behavior rather than a bug in qt.ctx.
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
        current_state = self.current.copy()
        current_state.update(new_args)
        token = _ctx_storage.set(qDict(current_state))
        try:
            yield
        finally:
            _ctx_storage.reset(token)

    def to_dict(self) -> qDict:
        """
        Expose a plain-dict snapshot of the current context.
        Useful for debugging or when a dict interface is needed.
        """
        return self.current.to_dict()

    def __repr__(self):
        return f"ContextProxy({self.current})"


# Singleton instance exposed to the package
ctx = _ContextManager()


# ================= Dependency Injection Decorator =================


def use_ctx(cls_or_name: Union[Type, str, None] = None, *args: str, **aliases: str):
    """
    Dependency Injection Decorator.
    Automatically injects matching variables from `qt.ctx` into the class `__init__`.

    Priority: Explicit User Kwargs > Explicit User Positional Args > Context > Defaults.

    Features:
    - Zero runtime reflection overhead (Signature parsed once at definition).
    - Bulletproof positional argument tracking (Ignores *args and **kwargs).
    - MRO compatible (Preserves class hierarchy).

    Usage Modes:
        1. Fully Auto: `@qt.use_ctx`
        2. Whitelist:  `@qt.use_ctx("dim", "lr")`
        3. Aliasing:   `@qt.use_ctx(dim="global_dim")`
        4. Mixed:      `@qt.use_ctx("lr", dim="global_dim")`

    Example:
        Fully automatic injection:
            @qt.use_ctx
            class Attention:
                def __init__(self, dim=64, heads=8):
                    ...
            with qt.ctx(dim=512, heads=16):
                layer = Attention()  # dim=512, heads=16

        Whitelist mode only injects the named parameters:
            @qt.use_ctx("dim")
            class FeedForward:
                def __init__(self, dim=128, dropout=0.1):
                    ...
            with qt.ctx(dim=1024, dropout=0.5):
                ffn = FeedForward()  # dim=1024, dropout=0.1

        Alias mode maps an init parameter to a different ctx key:
            @qt.use_ctx(dim="global_hidden_size")
            class Projection:
                def __init__(self, dim=128):
                    ...
            with qt.ctx(global_hidden_size=2048):
                block = Projection()  # dim=2048

        Mixed mode supports both whitelist and alias mapping:
            @qt.use_ctx("layers", dropout="global_dropout")
            class Transformer:
                def __init__(self, layers=6, dropout=0.1, heads=8):
                    ...

        Manual arguments always win over injected ctx values:
            with qt.ctx(layers=12, global_dropout=0.3):
                model = Transformer(layers=24)  # layers=24, dropout=0.3, heads=8
    """

    # Pre-compute the injection mapping dictionary during the import phase
    inject_map = {}
    if isinstance(cls_or_name, str):
        inject_map[cls_or_name] = cls_or_name
    for name in args:
        inject_map[name] = name
    inject_map.update(aliases)

    def decorator(target_cls: Type) -> Type:
        if not inspect.isclass(target_cls):
            raise TypeError("@use_ctx can only be applied to classes.")

        original_init = target_cls.__init__
        sig = inspect.signature(original_init)

        # 1. Identify parameters that can actually be filled by positional arguments.
        # This prevents bugs when the __init__ signature contains *args or Keyword-Only params.
        positional_params = [
            name
            for name, param in sig.parameters.items()
            if name != "self" and param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
        ]

        # 2. Filter out valid injection candidates based on the whitelist/aliases
        target_params = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Context cannot inject into variadic arguments (*args, **kwargs)
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue

            # If no map is provided (Auto mode), all params are targets.
            # Otherwise, check if it's in the explicit map.
            if not inject_map or name in inject_map:
                target_params.append(name)

        @wraps(original_init)
        def new_init(self, *i_args, **i_kwargs):
            # Calculate which parameters the user has already provided via positional args
            filled_positions = positional_params[: len(i_args)]

            # Fetch the live pulse of the global context
            current_ctx = ctx.current

            # Inject missing values
            for name in target_params:
                # Do not overwrite if the user explicitly provided the parameter
                if name in i_kwargs or name in filled_positions:
                    continue

                # Determine which key to look up in the context (handling aliases)
                ctx_key = inject_map.get(name, name) if inject_map else name

                # Perform the injection
                if ctx_key in current_ctx:
                    i_kwargs[name] = current_ctx[ctx_key]

            return original_init(self, *i_args, **i_kwargs)

        target_cls.__init__ = new_init
        return target_cls

    # --- Dispatch Logic ---
    # If the first argument is a class, it was called without parentheses: `@qt.use_ctx`
    if inspect.isclass(cls_or_name):
        return decorator(cls_or_name)

    # Otherwise, it was called with parameters: `@qt.use_ctx(...)`
    return decorator

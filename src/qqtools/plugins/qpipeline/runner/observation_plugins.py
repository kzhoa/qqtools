"""Generic lifecycle and subscription contracts for optional fact observers."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Protocol

from .contracts import ObserverBindings


@dataclass(frozen=True, slots=True)
class ObservationContext:
    rank: int
    max_steps: int | None
    max_epochs: int | None


class ObservationPlugin(Protocol):
    identifier: str
    subscriptions: tuple[tuple[str, Callable[[object], None]], ...]

    def close(self) -> None: ...


class ObservationPluginFactory(Protocol):
    def __call__(self, context: ObservationContext) -> ObservationPlugin | None: ...


@dataclass(frozen=True, slots=True)
class ObservationDiagnostic:
    phase: str
    plugin_identifier: str | None = None
    event_name: str | None = None


_MAX_DIAGNOSTICS = 64
_MAX_DIAGNOSTIC_IDENTIFIER = 64


def validate_observation_plugin_factories(
    factories: Sequence[ObservationPluginFactory],
) -> tuple[ObservationPluginFactory, ...]:
    """Validate an explicit public provider sequence before runner setup begins."""
    if not isinstance(factories, Sequence) or isinstance(factories, (str, bytes, bytearray)):
        raise TypeError("observation_plugins must be a sequence of callable factories.")
    try:
        values = tuple(factories)
    except Exception as error:
        raise TypeError("observation_plugins must be a readable sequence of callable factories.") from error
    if any(not callable(factory) for factory in values):
        raise TypeError("Every observation plugin factory must be callable.")
    return values


def _diagnostic_identifier(identifier: str | None) -> str | None:
    return identifier[:_MAX_DIAGNOSTIC_IDENTIFIER] if identifier is not None else None


def _record_close_failure(
    diagnostics: deque[ObservationDiagnostic],
    identifier: str | None,
) -> None:
    diagnostics.append(ObservationDiagnostic("close_failure", _diagnostic_identifier(identifier)))


def _close_instance(
    instance: object,
    identifier: str | None,
    diagnostics: deque[ObservationDiagnostic],
    closed_object_ids: set[int],
) -> None:
    instance_id = id(instance)
    if instance_id in closed_object_ids:
        return
    closed_object_ids.add(instance_id)
    try:
        close = getattr(instance, "close", None)
        if not callable(close):
            diagnostics.append(ObservationDiagnostic("invalid_instance", _diagnostic_identifier(identifier)))
            return
        close()
    except Exception:
        _record_close_failure(diagnostics, identifier)


class ObservationPluginLifecycle:
    """Own installed plugin instances and close them once in reverse order."""

    def __init__(
        self,
        instances: Sequence[tuple[str, ObservationPlugin]],
        observers: ObserverBindings,
        diagnostics: deque[ObservationDiagnostic],
    ) -> None:
        self._instances = tuple(instances)
        self._observers = observers
        self._diagnostics = diagnostics
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for identifier, instance in reversed(self._instances):
            try:
                instance.close()
            except Exception:
                _record_close_failure(self._diagnostics, identifier)

    def drain_diagnostics(self, *, limit: int = _MAX_DIAGNOSTICS) -> tuple[ObservationDiagnostic, ...]:
        """Drain bounded primitive lifecycle and callback diagnostics without logging."""
        if type(limit) is not int or limit < 0:
            raise ValueError("Diagnostic limit must be a non-negative integer.")
        remaining = min(limit, _MAX_DIAGNOSTICS)
        if remaining == 0:
            return ()
        drained: list[ObservationDiagnostic] = []
        while self._diagnostics and remaining:
            drained.append(self._diagnostics.popleft())
            remaining -= 1
        if remaining:
            for event_name, _ in self._observers.drain_optional_diagnostics(limit=remaining):
                drained.append(ObservationDiagnostic("callback_failure", event_name=event_name))
        return tuple(drained)


def install_observation_plugins(
    factories: Sequence[ObservationPluginFactory],
    context: ObservationContext,
    observers: ObserverBindings,
) -> ObservationPluginLifecycle:
    """Create, validate, and atomically bind optional observation plugins."""
    providers = validate_observation_plugin_factories(factories)
    diagnostics: deque[ObservationDiagnostic] = deque(maxlen=_MAX_DIAGNOSTICS)
    created: list[object] = []
    closed_object_ids: set[int] = set()

    try:
        for factory in providers:
            try:
                instance = factory(context)
            except Exception:
                diagnostics.append(ObservationDiagnostic("factory_failure"))
                continue
            if instance is None:
                continue
            created.append(instance)

        validated: list[tuple[object, str, tuple[tuple[str, Callable[[object], None]], ...]]] = []
        for instance in created:
            try:
                identifier = getattr(instance, "identifier")
                subscriptions = getattr(instance, "subscriptions")
                if type(identifier) is not str or not identifier or len(identifier) > 128:
                    raise ValueError("invalid identifier")
                if type(subscriptions) is not tuple:
                    raise ValueError("subscriptions must be an immutable tuple")
                normalized: list[tuple[str, Callable[[object], None]]] = []
                for subscription in subscriptions:
                    if type(subscription) is not tuple or len(subscription) != 2:
                        raise ValueError("invalid subscription pair")
                    name, callback = subscription
                    if type(name) is not str or not callable(callback):
                        raise ValueError("invalid subscription declaration")
                    normalized.append((name, callback))
                if not callable(getattr(instance, "close", None)):
                    raise ValueError("plugin close must be callable")
            except Exception:
                diagnostics.append(ObservationDiagnostic("invalid_declaration"))
                _close_instance(instance, None, diagnostics, closed_object_ids)
                continue
            validated.append((instance, identifier, tuple(normalized)))

        identifier_counts = Counter(identifier for _, identifier, _ in validated)
        object_counts = Counter(id(instance) for instance, _, _ in validated)
        installable: list[tuple[object, str, tuple[tuple[str, Callable[[object], None]], ...]]] = []
        for instance, identifier, subscriptions in validated:
            if identifier_counts[identifier] > 1 or object_counts[id(instance)] > 1:
                diagnostics.append(ObservationDiagnostic("duplicate_identifier", _diagnostic_identifier(identifier)))
                _close_instance(instance, identifier, diagnostics, closed_object_ids)
                continue
            installable.append((instance, identifier, subscriptions))

        installed: list[tuple[str, ObservationPlugin]] = []
        for instance, identifier, subscriptions in installable:
            try:
                observers.bind_optional_bundle(subscriptions, policy="best_effort")
            except Exception:
                diagnostics.append(ObservationDiagnostic("subscription_failure", _diagnostic_identifier(identifier)))
                _close_instance(instance, identifier, diagnostics, closed_object_ids)
                continue
            installed.append((identifier, instance))  # type: ignore[arg-type]

        return ObservationPluginLifecycle(installed, observers, diagnostics)
    except BaseException:
        for instance in created:
            _close_instance(instance, None, diagnostics, closed_object_ids)
        raise

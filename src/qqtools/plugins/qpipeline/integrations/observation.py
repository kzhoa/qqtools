"""Explicit composition of optional qPipeline observation providers."""

from __future__ import annotations

from ..runner.observation_plugins import ObservationPluginFactory
from .qexp_progress import qexp_progress_factory

_DEFAULT_OBSERVATION_PLUGINS: tuple[ObservationPluginFactory, ...] = (qexp_progress_factory,)


def default_observation_plugins() -> tuple[ObservationPluginFactory, ...]:
    """Return the stable built-in observation provider sequence."""
    return _DEFAULT_OBSERVATION_PLUGINS

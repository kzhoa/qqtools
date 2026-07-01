from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from ..runner_utils.types import LoopSignal, RunningState
from .types import BaseEventContext, EVENT_SPECS, EventName, EventSpec, RunnerRuntimeView


class EventDispatcher:
    def __init__(self) -> None:
        self.listeners: Dict[str, List[Callable]] = {event.value: [] for event in EventName}
        self.specs: Dict[str, EventSpec] = dict(EVENT_SPECS)
        self.emitters: Dict[str, Callable[..., BaseEventContext]] = {
            event_name: self._compile_emitter(spec)
            for event_name, spec in self.specs.items()
        }

    def _compile_emitter(self, spec: EventSpec) -> Callable[..., BaseEventContext]:
        context_type = spec.context_type

        def _emit(
            *,
            state: RunningState,
            stage: Optional[str],
            max_epochs: Optional[int],
            max_steps: Optional[int],
            signal: Optional[LoopSignal] = None,
            **payload: Any,
        ) -> BaseEventContext:
            return context_type(
                runner=RunnerRuntimeView(
                    run_state=state,
                    stage=stage,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                ),
                signal=signal,
                **payload,
            )

        return _emit

    def _validate_event_name(self, event: Union[str, EventName]) -> str:
        event_name = event.value if isinstance(event, EventName) else str(event)
        if event_name not in self.listeners:
            allowed_events = ", ".join(sorted(self.listeners.keys()))
            raise ValueError(
                f"Unknown event: {event_name}. Register it in EventName first. Allowed events: {allowed_events}"
            )
        return event_name

    def add_listener(self, event: Union[str, EventName], listener: Callable) -> None:
        event_name = self._validate_event_name(event)
        self.listeners[event_name].append(listener)

    def remove_listener(self, event: Union[str, EventName], listener: Callable) -> None:
        event_name = self._validate_event_name(event)
        self.listeners[event_name].remove(listener)

    def has_listeners(self, event: Union[str, EventName]) -> bool:
        event_name = self._validate_event_name(event)
        return bool(self.listeners[event_name])

    def dispatch(
        self,
        event: Union[str, EventName],
        *,
        state: RunningState,
        stage: Optional[str],
        max_epochs: Optional[int],
        max_steps: Optional[int],
        signal: Optional[LoopSignal] = None,
        **payload: Any,
    ) -> None:
        event_name = self._validate_event_name(event)
        listeners = self.listeners[event_name]
        if not listeners:
            return

        context = self.emitters[event_name](
            state=state,
            stage=stage,
            max_epochs=max_epochs,
            max_steps=max_steps,
            signal=signal,
            **payload,
        )
        for listener in listeners:
            listener(context)

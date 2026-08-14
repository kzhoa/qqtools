from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

from ..runner_utils.types import LoopSignal, RunningState
from .types import BaseEventContext, CommandName, EVENT_SPECS, EventName, EventSpec, RunnerRuntimeView


class EventDispatcher:
    """Runner extension entry point for command invocation and event dispatch.

    Commands and events deliberately use separate identifiers, registries, and APIs:
    a command resolves one handler and returns its result, while an event notifies its
    registered listeners without aggregating a business result.
    """

    def __init__(self) -> None:
        self.listeners: Dict[str, List[Callable]] = {
            event.value: [] for event in EventName if event != EventName.ON_CHECKPOINT_SAVED
        }
        self._internal_listeners: Dict[str, List[Callable]] = {
            EventName.ON_CHECKPOINT_SAVED.value: []
        }
        self.handlers: Dict[CommandName, Callable] = {}
        self.specs: Dict[str, EventSpec] = dict(EVENT_SPECS)
        self.emitters: Dict[str, Callable[..., BaseEventContext]] = {
            event_name: self._compile_emitter(spec)
            for event_name, spec in self.specs.items()
        }

    def _compile_emitter(self, spec: EventSpec) -> Callable[..., Any]:
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
        if event_name not in self.specs:
            allowed_events = ", ".join(sorted(self.specs.keys()))
            raise ValueError(
                f"Unknown event: {event_name}. Register it in EventName first. Allowed events: {allowed_events}"
            )
        return event_name

    def _validate_public_event_name(self, event: Union[str, EventName]) -> str:
        event_name = self._validate_event_name(event)
        if event_name in self._internal_listeners:
            raise ValueError(f"Event {event_name} is internal and cannot be registered publicly.")
        return event_name

    @staticmethod
    def _validate_command_name(command: CommandName) -> CommandName:
        if not isinstance(command, CommandName):
            raise ValueError(f"Unknown command: {command}. Use a CommandName value.")
        return command

    def add_listener(self, event: Union[str, EventName], listener: Callable) -> None:
        event_name = self._validate_public_event_name(event)
        self.listeners[event_name].append(listener)

    def remove_listener(self, event: Union[str, EventName], listener: Callable) -> None:
        event_name = self._validate_public_event_name(event)
        self.listeners[event_name].remove(listener)

    def has_listeners(self, event: Union[str, EventName]) -> bool:
        event_name = self._validate_public_event_name(event)
        return bool(self.listeners[event_name])

    def set_handler(self, command: CommandName, handler: Callable) -> None:
        command_name = self._validate_command_name(command)
        if command_name in self.handlers:
            raise RuntimeError(f"Command handler is already registered: {command_name.value}")
        self.handlers[command_name] = handler

    def invoke(self, command: CommandName, context: object) -> Any:
        command_name = self._validate_command_name(command)
        handler = self.handlers.get(command_name)
        if handler is None:
            raise RuntimeError(f"No handler registered for command: {command_name.value}")
        return handler(context)

    def _add_internal_listener(self, event: EventName, listener: Callable) -> None:
        event_name = self._validate_event_name(event)
        if event_name not in self._internal_listeners:
            raise ValueError(f"Event {event_name} is not an internal notification.")
        self._internal_listeners[event_name].append(listener)

    def _dispatch_internal_context(self, event: EventName, context: object) -> None:
        event_name = self._validate_event_name(event)
        if event_name not in self._internal_listeners:
            raise ValueError(f"Event {event_name} is not an internal notification.")
        for listener in self._internal_listeners[event_name]:
            listener(context)

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
        event_name = self._validate_public_event_name(event)
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

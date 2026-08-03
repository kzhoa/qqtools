"""Initialization-time binding for evolvable qTask hooks."""

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, cast

from ...types import Stage


@dataclass(frozen=True)
class HookCallContract:
    hook_name: str
    core_parameters: tuple[str, ...]
    injectable_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        parameter_names = (*self.core_parameters, *self.injectable_parameters)
        if not self.hook_name:
            raise ValueError("HookCallContract hook_name must be non-empty.")
        if any(not parameter_name for parameter_name in parameter_names):
            raise ValueError("HookCallContract parameter names must be non-empty.")
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("HookCallContract parameter names must be unique.")

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset((*self.core_parameters, *self.injectable_parameters))


POST_METRICS_TO_VALUE_CONTRACT = HookCallContract(
    hook_name="post_metrics_to_value",
    core_parameters=("result",),
    injectable_parameters=("stage",),
)


class PostMetricsValueResolver(Protocol):
    def __call__(self, *, result: Mapping[str, Any], stage: Stage) -> float | None: ...


def _signature_error(contract: HookCallContract, signature: inspect.Signature, reason: str) -> ValueError:
    allowed_names = ", ".join((*contract.core_parameters, *contract.injectable_parameters))
    return ValueError(
        f"Unsupported qTask hook signature for '{contract.hook_name}{signature}': {reason}. "
        f"Allowed named parameters: {allowed_names}."
    )


def bind_task_hook(task: Any, contract: HookCallContract) -> Callable[..., Any]:
    """Validate one qTask hook once and return its named-argument resolver."""
    method = getattr(task, contract.hook_name, None)
    if not callable(method):
        raise ValueError(f"qTask hook '{contract.hook_name}' must be callable.")

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unable to inspect qTask hook '{contract.hook_name}': {error}") from error

    parameters = tuple(signature.parameters.values())
    declared_names = []
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise _signature_error(contract, signature, f"positional-only parameter '{parameter.name}' is unsupported")
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            raise _signature_error(contract, signature, "*args is unsupported")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            raise _signature_error(contract, signature, "**kwargs is unsupported")
        if parameter.name not in contract.parameter_names:
            raise _signature_error(contract, signature, f"unknown parameter '{parameter.name}'")
        declared_names.append(parameter.name)

    missing_core_parameters = set(contract.core_parameters).difference(declared_names)
    if missing_core_parameters:
        missing_names = ", ".join(sorted(missing_core_parameters))
        raise _signature_error(contract, signature, f"missing required parameter(s): {missing_names}")

    def resolver(**provided_values: Any) -> Any:
        return method(**{name: provided_values[name] for name in declared_names})

    return resolver


def bind_post_metrics_to_value(task: Any) -> PostMetricsValueResolver:
    """Bind qTask.post_metrics_to_value to qpipeline's metric contract."""
    return cast(PostMetricsValueResolver, bind_task_hook(task, POST_METRICS_TO_VALUE_CONTRACT))

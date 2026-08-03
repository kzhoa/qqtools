import pytest

from qqtools.plugins.qpipeline import Stage
from qqtools.plugins.qpipeline.runner.runner_utils.hook_binding import HookCallContract, bind_post_metrics_to_value


class _Task:
    pass


def test_hook_call_contract_accepts_unique_non_empty_names():
    contract = HookCallContract("hook", ("result",), ("stage",))
    assert contract.parameter_names == frozenset({"result", "stage"})


@pytest.mark.parametrize(
    ("hook_name", "core_parameters", "injectable_parameters"),
    [
        ("", ("result",), ("stage",)),
        ("hook", ("",), ("stage",)),
        ("hook", ("result",), ("result",)),
    ],
)
def test_hook_call_contract_rejects_invalid_registration(hook_name, core_parameters, injectable_parameters):
    with pytest.raises(ValueError):
        HookCallContract(hook_name, core_parameters, injectable_parameters)


@pytest.mark.parametrize(
    "method",
    [
        lambda result: result["metric"],
        lambda result, stage: (result["metric"], stage),
        lambda stage, result: (result["metric"], stage),
        lambda result, *, stage: (result["metric"], stage),
        lambda *, stage, result: (result["metric"], stage),
        lambda result, stage=None: (result["metric"], stage),
    ],
)
def test_bind_post_metrics_to_value_supports_declared_signatures(method):
    task = _Task()
    task.post_metrics_to_value = method

    resolver = bind_post_metrics_to_value(task)
    result = resolver(result={"metric": 1.5}, stage=Stage.VAL)

    assert result in (1.5, (1.5, Stage.VAL))


@pytest.mark.parametrize(
    ("method", "reason"),
    [
        (lambda: 0.0, "missing required parameter"),
        (lambda metrics: 0.0, "unknown parameter 'metrics'"),
        (lambda other=None: 0.0, "unknown parameter 'other'"),
        (lambda result, stgae=None: 0.0, "unknown parameter 'stgae'"),
        (lambda result, /: 0.0, "positional-only parameter 'result'"),
        (lambda result, *args: 0.0, r"\*args is unsupported"),
        (lambda result, **kwargs: 0.0, r"\*\*kwargs is unsupported"),
    ],
)
def test_bind_post_metrics_to_value_rejects_unsupported_signatures(method, reason):
    task = _Task()
    task.post_metrics_to_value = method

    with pytest.raises(ValueError, match=reason):
        bind_post_metrics_to_value(task)


def test_bound_resolver_does_not_reinspect_or_retry_user_type_error(monkeypatch):
    import qqtools.plugins.qpipeline.runner.runner_utils.hook_binding as hook_binding

    calls = 0
    original_signature = hook_binding.inspect.signature

    def counted_signature(method):
        nonlocal calls
        calls += 1
        return original_signature(method)

    class TypeErrorTask:
        def post_metrics_to_value(self, result, *, stage):
            raise TypeError("user hook failure")

    monkeypatch.setattr(hook_binding.inspect, "signature", counted_signature)
    resolver = bind_post_metrics_to_value(TypeErrorTask())

    with pytest.raises(TypeError, match="user hook failure"):
        resolver(result={}, stage=Stage.VAL)

    assert calls == 1


def test_stage_is_shared_by_task_runner_events_and_root_export():
    from qqtools.plugins.qpipeline import Stage as exported_stage
    from qqtools.plugins.qpipeline.runner.agent import Stage as agent_stage
    from qqtools.plugins.qpipeline.runner.events.types import Stage as event_stage
    from qqtools.plugins.qpipeline.task.qtask import Stage as task_stage

    assert exported_stage is Stage
    assert task_stage is Stage
    assert agent_stage is Stage
    assert event_stage is Stage

import contextvars

import pytest

import qqtools as qt
from qqtools.qdict import qDict


@pytest.fixture(autouse=True)
def reset_ctx():
    qt.ctx.reset()
    yield
    qt.ctx.reset()


def test_current_initializes_independent_qdict():
    current = qt.ctx.current

    assert isinstance(current, qDict)
    assert current.to_dict() == {}
    assert current.allow_notexist is False


def test_fresh_context_does_not_observe_default_context_mutation():
    qt.ctx.set("main_only", 1)

    fresh_context = contextvars.Context()
    fresh_before = fresh_context.run(qt.ctx.to_dict)
    fresh_context.run(lambda: qt.ctx.set("fresh_only", 2))
    fresh_after = fresh_context.run(qt.ctx.to_dict)

    assert qt.ctx.to_dict() == {"main_only": 1}
    assert fresh_before == {}
    assert fresh_after == {"fresh_only": 2}


def test_with_ctx_merges_nested_scopes_and_restores_previous_state():
    qt.ctx.set("base", "root")

    with qt.ctx(outer=1):
        assert qt.ctx.to_dict() == {"base": "root", "outer": 1}

        with qt.ctx({"outer": 2, "inner": 3}):
            assert qt.ctx.to_dict() == {"base": "root", "outer": 2, "inner": 3}

        assert qt.ctx.to_dict() == {"base": "root", "outer": 1}

    assert qt.ctx.to_dict() == {"base": "root"}


def test_reset_replaces_existing_context_with_empty_state():
    qt.ctx.set("value", 123)

    qt.ctx.reset()

    assert qt.ctx.to_dict() == {}

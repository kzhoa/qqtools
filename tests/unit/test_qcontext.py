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


#
# This test locks in the intended advanced-user contract:
# qt.ctx restores key bindings across scope exit, but it deliberately does not
# protect callers from in-place mutation of shared mutable objects reachable
# through the live context. That leakage is considered caller-managed behavior.
def test_with_ctx_allows_mutable_shared_object_leak_as_explicit_contract():
    shared_items = [1]
    qt.ctx.set("items", shared_items)

    with qt.ctx(flag=True):
        qt.ctx.items.append(2)

    assert shared_items == [1, 2]
    assert qt.ctx.to_dict() == {"items": [1, 2]}


def test_reset_replaces_existing_context_with_empty_state():
    qt.ctx.set("value", 123)

    qt.ctx.reset()

    assert qt.ctx.to_dict() == {}


def test_use_ctx_auto_injects_matching_init_params():
    @qt.use_ctx
    class Demo:
        def __init__(self, foo=None, bar=None):
            self.foo = foo
            self.bar = bar

    qt.ctx.set("foo", 1)
    qt.ctx.set("bar", 2)

    instance = Demo()

    assert instance.foo == 1
    assert instance.bar == 2


def test_use_ctx_whitelist_only_injects_selected_params():
    @qt.use_ctx("foo")
    class Demo:
        def __init__(self, foo=None, bar=None):
            self.foo = foo
            self.bar = bar

    qt.ctx.set("foo", 1)
    qt.ctx.set("bar", 2)

    instance = Demo()

    assert instance.foo == 1
    assert instance.bar is None


def test_use_ctx_alias_reads_from_mapped_context_key():
    @qt.use_ctx(foo="global_foo")
    class Demo:
        def __init__(self, foo=None, bar=None):
            self.foo = foo
            self.bar = bar

    qt.ctx.set("global_foo", 7)
    qt.ctx.set("foo", 99)

    instance = Demo()

    assert instance.foo == 7
    assert instance.bar is None


def test_use_ctx_mixed_mode_keeps_explicit_values_higher_priority():
    @qt.use_ctx("foo", bar="global_bar")
    class Demo:
        def __init__(self, foo=None, bar=None):
            self.foo = foo
            self.bar = bar

    qt.ctx.set("foo", 1)
    qt.ctx.set("global_bar", 2)

    instance = Demo(foo=10, bar=20)

    assert instance.foo == 10
    assert instance.bar == 20


def test_use_ctx_positional_args_override_context_injection():
    @qt.use_ctx("foo", "bar")
    class Demo:
        def __init__(self, foo=None, bar=None):
            self.foo = foo
            self.bar = bar

    qt.ctx.set("foo", 1)
    qt.ctx.set("bar", 2)

    instance = Demo(10)

    assert instance.foo == 10
    assert instance.bar == 2

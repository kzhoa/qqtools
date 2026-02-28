import argparse
import copy
import runpy
import sys

import pytest

from qqtools.qdict import qDict

# -------------------------------------------------------------------
# 1. Initialization & Basic Access
# -------------------------------------------------------------------


def test_init_empty():
    """Test empty initialization."""
    d = qDict()
    assert len(d) == 0
    assert isinstance(d, dict)


def test_init_from_dict():
    """Test initialization from a dictionary."""
    raw = {"name": "test", "val": 123}
    d = qDict(raw)
    assert d.name == "test"
    assert d["val"] == 123


def test_init_from_kwargs():
    """Test initialization from kwargs via from_args."""
    d = qDict.from_args(a=1, b=2)
    assert d.a == 1
    assert d.b == 2


def test_init_from_namespace():
    """Test initialization from argparse.Namespace."""
    ns = argparse.Namespace(x=10, y=20)
    d = qDict.from_namespace(ns)
    assert d.x == 10
    assert d.y == 20


def test_init_directly_from_namespace_in___init__branch():
    """Test the direct Namespace branch inside qDict.__init__."""
    ns = argparse.Namespace(alpha=1, beta=2)
    d = qDict(ns)
    assert d.alpha == 1
    assert d.beta == 2


def test_init_from_list():
    """Test initialization from key/value lists."""
    keys = ["a", "b"]
    vals = [1, 2]
    d = qDict.from_list(keys, vals)
    assert d.a == 1
    assert d.b == 2


def test_init_from_list_with_callable():
    """Test initialization from keys with callable-generated values."""
    keys = ["a", "b"]
    d = qDict.from_list(keys, lambda: 0)
    assert d.a == 0
    assert d.b == 0


def test_init_from_iterable_branch_with_recursive_dict_value():
    """Test iterable init branch with recursive nested-dict conversion."""
    d = qDict([("a", {"b": 1}), ("c", 2)], recursive=True)
    assert isinstance(d.a, qDict)
    assert d.a.b == 1
    assert d.c == 2


# -------------------------------------------------------------------
# 2. Recursive Behavior
# -------------------------------------------------------------------


def test_recursive_conversion_true():
    """Test recursive conversion of nested dictionaries to qDict."""
    data = {"a": {"b": {"c": 1}}}
    d = qDict(data, recursive=True)

    assert isinstance(d.a, qDict)
    assert isinstance(d.a.b, qDict)
    assert d.a.b.c == 1


def test_recursive_conversion_false():
    """Test behavior when recursive conversion is disabled."""
    data = {"a": {"b": 1}}
    d = qDict(data, recursive=False)

    assert isinstance(d.a, dict)
    assert not isinstance(d.a, qDict)
    # Without recursive conversion, nested dicts do not support dot access.
    with pytest.raises(AttributeError):
        _ = d.a.b


def test_build_recursive_value_returns_input_for_non_dict():
    """Test _build_recursive_value passthrough for non-dict values."""
    d = qDict()
    obj = 123
    assert d._build_recursive_value(obj, default_function=None, allow_notexist=True) == 123


# -------------------------------------------------------------------
# 3. Missing Key Handling
# -------------------------------------------------------------------


def test_allow_notexist_true():
    """Test allow_notexist=True (default behavior)."""
    d = qDict(allow_notexist=True)
    # Attribute access returns None.
    assert d.non_existent is None
    # Item access returns None.
    assert d["non_existent"] is None


def test_allow_notexist_false():
    """Test allow_notexist=False (raises errors for missing keys)."""
    d = qDict(allow_notexist=False)

    with pytest.raises(AttributeError):
        _ = d.non_existent_attr

    with pytest.raises(KeyError):
        _ = d["non_existent_key"]


def test_default_function():
    """Test default value generation via default_function."""
    d = qDict(default_function=lambda: "default_value")

    # Accessing a missing key triggers function generation and assignment.
    val = d.new_key
    assert val == "default_value"
    # Confirm the generated value is stored in the dictionary.
    assert "new_key" in d
    assert d["new_key"] == "default_value"


# -------------------------------------------------------------------
# 4. Setters & Updates
# -------------------------------------------------------------------


def test_setattr():
    """Test setting values via dot notation."""
    d = qDict()
    d.new_attr = 100
    assert d["new_attr"] == 100


def test_allow_notexist_setter_works_via_attr():
    """Test allow_notexist assignment via property setter."""
    d = qDict(allow_notexist=True)
    assert d.allow_notexist is True

    d.allow_notexist = False

    assert d.allow_notexist is False
    assert "allow_notexist" not in d


def test_default_function_setter_works_via_attr():
    """Test default_function assignment via property setter."""
    d = qDict(default_function=lambda: 1)

    new_default = lambda: 2
    d.default_function = new_default

    assert d.default_function is new_default
    assert "default_function" not in d

    d.default_function = None
    assert "_default_function" not in d.__dict__
    assert "default_function" not in d


def test_lazy_update():
    """Test lazy_update (only updates missing keys)."""
    d = qDict({"exists": 1})
    update_data = {"exists": 2, "new": 3}

    d.lazy_update(update_data)

    assert d.exists == 1  # Existing value should be preserved.
    assert d.new == 3  # New key should be added.


def test_recursive_update():
    """Test recursive_update behavior."""
    d = qDict({"info": {"name": "old", "age": 10}})
    new_data = {"info": {"name": "new", "city": "NY"}}

    d.recursive_update(new_data)

    assert d.info.name == "new"  # Updated.
    assert d.info.age == 10  # Preserved.
    assert d.info.city == "NY"  # Added.
    assert isinstance(d.info, qDict)  # Type preserved.


# -------------------------------------------------------------------
# 5. Utility Methods
# -------------------------------------------------------------------


def test_fetch():
    """Test fetching multiple values by keys."""
    d = qDict({"a": 1, "b": 2, "c": 3})
    res = d.fetch(["a", "c"])
    assert res == [1, 3]


def test_to_dict():
    """Test conversion back to a plain dict."""
    d = qDict({"a": {"b": 1}})
    plain = d.to_dict()

    assert type(plain) == dict
    assert type(plain["a"]) == dict
    assert plain["a"]["b"] == 1


def test_safe_pop():
    """Test safe_pop behavior."""
    d = qDict({"a": 1})
    val = d.safe_pop("a")
    assert val == 1
    assert "a" not in d

    val_none = d.safe_pop("z")
    assert val_none is None


def test_remove():
    """Test remove and chain-style return value."""
    d = qDict({"a": 1, "b": 2})
    res = d.remove("a")
    assert "a" not in d
    assert res is d  # Chain-call check.


def test_copy():
    """Test shallow copy behavior."""
    # Note: __copy__ implementation is slightly customized; validate core behavior.
    d = qDict({"a": 1}, allow_notexist=True)
    d_copy = d.copy()

    assert d_copy.a == 1
    assert d_copy is not d
    # Check that configuration flags are propagated.
    assert d_copy.allow_notexist is True


def test_recursive_conversion_uses_subclass():
    """Test recursive conversion preserves subclass type."""

    class ChildDict(qDict):
        pass

    data = {"a": {"b": 1}}
    d = ChildDict(data, recursive=True)

    assert isinstance(d, ChildDict)
    assert isinstance(d.a, ChildDict)
    assert d.a.b == 1


def test_copy_is_shallow_and_deepcopy_is_deep():
    """Test shallow copy aliasing versus deep copy isolation."""
    d = qDict({"a": {"nested": [1]}, "b": [2]}, recursive=True)

    shallow = copy.copy(d)
    deep = copy.deepcopy(d)

    assert shallow is not d
    assert deep is not d

    shallow.a.nested.append(3)
    assert d.a.nested == [1, 3]

    deep.a.nested.append(9)
    assert d.a.nested == [1, 3]
    assert deep.a.nested == [1, 9]


def test_deepcopy_preserves_config_flags():
    """Test deepcopy keeps allow_notexist and default_function settings."""
    default_factory = lambda: []
    d = qDict({"a": 1}, default_function=default_factory, allow_notexist=False)

    copied = copy.deepcopy(d)

    assert copied.allow_notexist is False
    assert copied.default_function is default_factory


def test_default_function_priority_over_allow_notexist():
    """Test default_function takes priority over strict missing-key behavior."""
    d = qDict(default_function=lambda: "x", allow_notexist=False)
    assert d.missing_key == "x"
    assert d["another_missing"] == "x"


def test_default_function_setter_none_then_respects_allow_notexist_false():
    """Test strict missing-key behavior after removing default_function."""
    d = qDict(default_function=lambda: 123, allow_notexist=False)
    d.default_function = None

    with pytest.raises(AttributeError):
        _ = d.not_exists

    with pytest.raises(KeyError):
        _ = d["not_exists"]


def test_getitem_special_key_default_function():
    """Test special-key access for internal _default_function."""
    fn = lambda: 1
    d = qDict(default_function=fn)
    assert d["_default_function"] is fn


def test_from_list_length_mismatch_raises_assertion():
    """Test from_list raises when key/value lengths differ."""
    with pytest.raises(AssertionError):
        qDict.from_list(["a", "b", "c"], [1, 2])


def test_recursive_update_exclude_keys_keeps_original_value():
    """Test recursive_update respects exclude_keys at top level."""
    d = qDict({"a": {"old": 1}, "b": {"keep": True}}, recursive=True)
    d.recursive_update({"a": {"new": 2}, "b": {"keep": False}}, exclude_keys=["b"])

    assert d.a.new == 2
    assert d.b.keep is True


def test_deepcopy_with_memo_returns_memoized_object():
    """Test __deepcopy__ returns memoized object when present."""
    d = qDict({"a": 1})
    marker = object()
    memo = {id(d): marker}
    assert d.__deepcopy__(memo) is marker


def test_default_function_exception_propagates_getattr_and_getitem():
    """Test exceptions from default_function propagate through accessors."""

    def boom():
        raise RuntimeError("default function failed")

    d = qDict(default_function=boom, allow_notexist=True)

    with pytest.raises(RuntimeError, match="default function failed"):
        _ = d.any_attr

    with pytest.raises(RuntimeError, match="default function failed"):
        _ = d["any_key"]


def test_toggle_default_function_none_to_callable_restores_generation():
    """Test toggling default_function restores generated missing values."""
    d = qDict(default_function=None, allow_notexist=False)

    with pytest.raises(AttributeError):
        _ = d.x

    d.default_function = lambda: "generated"
    assert d.x == "generated"
    assert d["y"] == "generated"


def test_fetch_with_missing_key_respects_default_function():
    """Test fetch returns generated defaults for missing keys when configured."""
    d = qDict({"a": 1}, default_function=lambda: 0, allow_notexist=False)
    assert d.fetch(["a", "b"]) == [1, 0]


def test_fetch_with_missing_key_raises_when_strict_and_no_default():
    """Test fetch raises KeyError for missing keys in strict mode."""
    d = qDict({"a": 1}, default_function=None, allow_notexist=False)
    with pytest.raises(KeyError):
        d.fetch(["a", "missing"])


def test_getattr_fallback_default_function_branch_when_getitem_raises():
    """Test __getattr__ fallback branch when __getitem__ is overridden to fail."""

    class BrokenGetItemQDict(qDict):
        def __getitem__(self, key):
            raise KeyError(key)

    d = BrokenGetItemQDict(default_function=lambda: "fallback", allow_notexist=False)
    with pytest.raises(KeyError):
        _ = d.anything
    assert d.get("anything") == "fallback"


def test_getattr_fallback_allow_notexist_branch_when_getitem_raises():
    """Test __getattr__ allow_notexist fallback when __getitem__ is broken."""

    class BrokenGetItemQDict(qDict):
        def __getitem__(self, key):
            raise KeyError(key)

    d = BrokenGetItemQDict(default_function=None, allow_notexist=True)
    assert d.anything is None


def test_recursive_update_deep_nested_structure():
    """Test recursive_update merges deep nested mappings correctly."""
    d = qDict(
        {
            "level1": {
                "level2": {
                    "keep": 1,
                    "nested": {"x": 1},
                }
            }
        },
        recursive=True,
    )

    d.recursive_update({"level1": {"level2": {"new_key": 2, "nested": {"y": 3}}}})

    assert d.level1.level2.keep == 1
    assert d.level1.level2.new_key == 2
    assert d.level1.level2.nested.x == 1
    assert d.level1.level2.nested.y == 3


def test_recursive_update_exclude_top_level_only():
    """Test exclude_keys applies at top-level keys in recursive_update."""
    d = qDict({"a": {"v": 1}, "b": {"v": 10}}, recursive=True)

    d.recursive_update({"a": {"v": 2}, "b": {"v": 20}}, exclude_keys=["a"])

    assert d.a.v == 1
    assert d.b.v == 20


def test_recursive_update_preserves_existing_subclass_for_nested_dict():
    """Test recursive_update preserves subclass type for existing nested mapping."""

    class ChildDict(qDict):
        pass

    d = ChildDict({"cfg": ChildDict({"x": 1})}, recursive=True)
    d.recursive_update({"cfg": {"y": 2}})

    assert isinstance(d.cfg, ChildDict)
    assert d.cfg.x == 1
    assert d.cfg.y == 2


def test_recursive_update_empty_input_is_noop():
    """Test recursive_update with empty dict is a no-op."""
    d = qDict({"a": 1}, recursive=True)
    out = d.recursive_update({})
    assert out is d
    assert d.a == 1


def test_lazy_update_non_dict_input_raises_assertion_error():
    """Test lazy_update rejects non-dict input."""
    d = qDict({"a": 1})
    with pytest.raises(AssertionError):
        d.lazy_update([("b", 2)])


def test_recursive_update_new_nested_key_uses_qdict_conversion_path():
    """Test new nested dict keys are converted to qDict during recursive_update."""
    d = qDict({"a": 1}, recursive=True)
    d.recursive_update({"new": {"x": 1}})
    assert isinstance(d.new, qDict)
    assert d.new.x == 1


def test_repr_long_branch_for_large_dict():
    """Test __repr__ long-format branch for larger dictionaries."""
    d = qDict({"k1": 1, "k2": 2, "k3": 3, "k4": 4, "k5": 5})
    text = repr(d)
    assert text.startswith("qDict{\n")
    assert "'k1':1" in text
    assert text.endswith("}")


def test_qdict_main_block_executes():
    """Test qdict module __main__ block executes without exceptions."""
    original_module = sys.modules.pop("qqtools.qdict", None)
    if original_module is not None:
        sys.modules["qqtools._qdict_test_backup"] = original_module

    try:
        runpy.run_module("qqtools.qdict", run_name="__main__")
    finally:
        backup_module = sys.modules.pop("qqtools._qdict_test_backup", None)
        if backup_module is not None:
            sys.modules["qqtools.qdict"] = backup_module

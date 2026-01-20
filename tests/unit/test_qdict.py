import argparse

import pytest

from qqtools.qdict import qDict

# -------------------------------------------------------------------
# 1. 初始化与基本访问测试 (Init & Basic Access)
# -------------------------------------------------------------------


def test_init_empty():
    """测试空初始化"""
    d = qDict()
    assert len(d) == 0
    assert isinstance(d, dict)


def test_init_from_dict():
    """测试从字典初始化"""
    raw = {"name": "test", "val": 123}
    d = qDict(raw)
    assert d.name == "test"
    assert d["val"] == 123


def test_init_from_kwargs():
    """测试从 kwargs 初始化 (使用类方法 from_args)"""
    d = qDict.from_args(a=1, b=2)
    assert d.a == 1
    assert d.b == 2


def test_init_from_namespace():
    """测试从 argparse.Namespace 初始化"""
    ns = argparse.Namespace(x=10, y=20)
    d = qDict.from_namespace(ns)
    assert d.x == 10
    assert d.y == 20


def test_init_from_list():
    """测试从列表初始化"""
    keys = ["a", "b"]
    vals = [1, 2]
    d = qDict.from_list(keys, vals)
    assert d.a == 1
    assert d.b == 2


def test_init_from_list_with_callable():
    """测试从列表初始化，值使用函数生成"""
    keys = ["a", "b"]
    d = qDict.from_list(keys, lambda: 0)
    assert d.a == 0
    assert d.b == 0


# -------------------------------------------------------------------
# 2. 递归特性测试 (Recursive Behavior)
# -------------------------------------------------------------------


def test_recursive_conversion_true():
    """测试默认递归将嵌套字典转换为 qDict"""
    data = {"a": {"b": {"c": 1}}}
    d = qDict(data, recursive=True)

    assert isinstance(d.a, qDict)
    assert isinstance(d.a.b, qDict)
    assert d.a.b.c == 1


def test_recursive_conversion_false():
    """测试关闭递归转换"""
    data = {"a": {"b": 1}}
    d = qDict(data, recursive=False)

    assert isinstance(d.a, dict)
    assert not isinstance(d.a, qDict)
    # 不开启递归时，子字典不能通过点号访问内部属性
    with pytest.raises(AttributeError):
        _ = d.a.b


# -------------------------------------------------------------------
# 3. 缺失键处理测试 (Missing Key Handling)
# -------------------------------------------------------------------


def test_allow_notexist_true():
    """测试允许键不存在 (默认行为)"""
    d = qDict(allow_notexist=True)
    # 属性访问返回 None
    assert d.non_existent is None
    # 字典访问返回 None
    assert d["non_existent"] is None


def test_allow_notexist_false():
    """测试不允许键不存在 (报错)"""
    d = qDict(allow_notexist=False)

    with pytest.raises(AttributeError):
        _ = d.non_existent_attr

    with pytest.raises(KeyError):
        _ = d["non_existent_key"]


def test_default_function():
    """测试使用 default_function 生成默认值"""
    d = qDict(default_function=lambda: "default_value")

    # 访问不存在的键应触发函数并设置值
    val = d.new_key
    assert val == "default_value"
    # 确认值已被实际存入字典
    assert "new_key" in d
    assert d["new_key"] == "default_value"


# -------------------------------------------------------------------
# 4. 属性设置与更新测试 (Setters & Updates)
# -------------------------------------------------------------------


def test_setattr():
    """测试通过点号设置属性"""
    d = qDict()
    d.new_attr = 100
    assert d["new_attr"] == 100


def test_lazy_update():
    """测试 lazy_update (只更新不存在的键)"""
    d = qDict({"exists": 1})
    update_data = {"exists": 2, "new": 3}

    d.lazy_update(update_data)

    assert d.exists == 1  # 应该保持原值
    assert d.new == 3  # 新值应该被添加


def test_recursive_update():
    """测试递归更新"""
    d = qDict({"info": {"name": "old", "age": 10}})
    new_data = {"info": {"name": "new", "city": "NY"}}

    d.recursive_update(new_data)

    assert d.info.name == "new"  # 更新
    assert d.info.age == 10  # 保持
    assert d.info.city == "NY"  # 新增
    assert isinstance(d.info, qDict)  # 保持类型


# -------------------------------------------------------------------
# 5. 辅助方法测试 (Utility Methods)
# -------------------------------------------------------------------


def test_fetch():
    """测试批量获取值"""
    d = qDict({"a": 1, "b": 2, "c": 3})
    res = d.fetch(["a", "c"])
    assert res == [1, 3]


def test_to_dict():
    """测试转换回普通字典"""
    d = qDict({"a": {"b": 1}})
    plain = d.to_dict()

    assert type(plain) == dict
    assert type(plain["a"]) == dict
    assert plain["a"]["b"] == 1


def test_safe_pop():
    """测试安全弹出"""
    d = qDict({"a": 1})
    val = d.safe_pop("a")
    assert val == 1
    assert "a" not in d

    val_none = d.safe_pop("z")
    assert val_none is None


def test_remove():
    """测试删除并返回自身"""
    d = qDict({"a": 1, "b": 2})
    res = d.remove("a")
    assert "a" not in d
    assert res is d  # 链式调用检查


def test_copy():
    """测试浅拷贝"""
    # 注意：源码中的 __copy__ 实现似乎有点特殊，这里测试其基本行为
    d = qDict({"a": 1}, allow_notexist=True)
    d_copy = d.copy()

    assert d_copy.a == 1
    assert d_copy is not d
    # 检查属性是否传递
    assert d_copy.allow_notexist is True

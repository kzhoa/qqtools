"""
qq:
dict["qtx"] cannot be another key
tested ~ torch.2.6.1
"""

import inspect
import types

import torch

import qqtools as qt

__all__ = ["qContextProvider", "get_default_context"]

HAS_GLOBALLY_REGISTRIED = False
_DEFAULT_CONTEXT = qt.qDict()


def is_instance(obj):
    return not inspect.isclass(obj) and not inspect.isfunction(obj) and not inspect.ismethod(obj)


def get_default_context():
    return _DEFAULT_CONTEXT


def qContextProvider(obj):
    if isinstance(obj, dict):

        def decorator(cls):
            return _qContextProvider(cls, context_dict=obj)

        return decorator
    elif inspect.isclass(obj):
        return _qContextProvider(cls=obj, context_dict=_DEFAULT_CONTEXT)
    else:
        raise TypeError(f"Unsupport Type: {type(obj)}")


def _qContextProvider(cls: torch.nn.Module, context_dict: qt.qDict):
    """"""

    def patch_cls(instance):
        original_init = instance.__init__
        orignal_getattr = instance.__getattr__
        orignal_setattr = instance.__setattr__

        def __appended__init__(self, *args, **kwargs):
            # set before init
            self.__dict__["qtx"] = context_dict
            self.__dict__["_qtx_patched"] = True
            original_init(self, *args, **kwargs)

        def __hook_getattr__(self, name: str):
            if name == "qtx":
                return self.__dict__["qtx"]
            else:
                return orignal_getattr(self, name)

        def __hook_setattr__(self, name, value):
            if name == "qtx":
                self.__dict__["qtx"] = value
                return
            else:
                orignal_setattr(self, name, value)

        assert inspect.isclass(instance)
        instance.__init__ = __appended__init__
        instance.__getattr__ = __hook_getattr__
        instance.__setattr__ = __hook_setattr__
        return instance

    def patch_instance(instance):
        assert is_instance(instance)

        orignal_getattr = instance.__getattr__
        orignal_setattr = instance.__setattr__

        def __hook_getattr__(self, name: str):
            if name == "qtx":
                return self.__dict__["qtx"]
            elif name in self.__dict__:
                return self.__dict__[name]
            else:
                return orignal_getattr(name)

        def __hook_setattr__(self, name, value):
            if name == "qtx":
                self.__dict__["qtx"] = value
                return
            elif isinstance(value, torch.nn.Module):
                _value = patch_instance(value)
                orignal_setattr(name, _value)
            else:
                orignal_setattr(name, value)

        # avoid double patch
        if "_qtx_patched" in instance.__dict__:
            return instance

        instance.__dict__["qtx"] = context_dict
        instance.__dict__["_qtx_patched"] = True
        instance.__getattr__ = types.MethodType(__hook_getattr__, instance)
        instance.__setattr__ = types.MethodType(__hook_setattr__, instance)

        # tail recursive
        submodules = instance.__dict__.get("_modules")
        for name, submodule in submodules.items():
            submodules[name] = patch_instance(submodule)
        return instance

    # global hook
    def _global_regist_module_hook(module, name, submodule):
        if "_qtx_patched" in module.__dict__:
            return patch_instance(submodule)

    hooks_dict = torch.nn.modules.module._global_module_registration_hooks
    if _global_regist_module_hook not in hooks_dict.values():
        torch.nn.modules.module.register_module_module_registration_hook(_global_regist_module_hook)

    return patch_cls(cls)

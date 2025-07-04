"""
qq:
dict["qtx"] cannot be another key
tested ~ torch.2.6.1
"""

import inspect
import types

import qqtools as qt
import torch

__all__ = ["qContextProvider"]

HAS_GLOBALLY_REGISTRIED = False


def is_instance(obj):
    return not inspect.isclass(obj) and not inspect.isfunction(obj) and not inspect.ismethod(obj)


def qContextProvider(cls: torch.nn.Module):
    """
    provide `self.qtx` to every sub-`nn.Module` of a root `nn.Module`.
    """
    context_dict = qt.qDict()

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
            elif isinstance(value, torch.nn.Module):
                _value = patch_instance(value)
                orignal_setattr(self, name, _value)
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
        orignal_getattribute = instance.__getattribute__

        def __hook_getattribute__(self, name: str):
            if name == "qtx":
                return self.__dict__["qtx"]
            elif name in self.__dict__:
                return self.__dict__[name]
            else:
                return __hook_getattr__(self, name)

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
        instance.__getattribute__ = types.MethodType(__hook_getattribute__, instance)
        instance.__getattr__ = types.MethodType(__hook_getattr__, instance)
        instance.__setattr__ = types.MethodType(__hook_setattr__, instance)

        return instance

    # global hook (optional) can be removed
    def _global_regist_module_hook(module, name, submodule):
        if "_qtx_patched" in module.__dict__:
            return patch_instance(submodule)

    hooks_dict = torch.nn.modules.module._global_module_registration_hooks
    global HAS_GLOBALLY_REGISTRIED
    if not HAS_GLOBALLY_REGISTRIED:
        torch.nn.modules.module.register_module_module_registration_hook(_global_regist_module_hook)
        HAS_GLOBALLY_REGISTRIED = True
    return patch_cls(cls)

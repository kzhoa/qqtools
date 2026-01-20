"""
qq:
dict["qtx"] cannot be another key
tested ~ torch.2.6.1
"""

import inspect
import types

import torch

import qqtools as qt

__all__ = ["qContextProvider", "patch_instance", "patch_cls", "get_default_context"]

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


def patch_instance(instance, context_dict):
    assert is_instance(instance)

    orignal_getattr = instance.__getattr__ if hasattr(instance, "__getattr__") else None
    orignal_setattr = instance.__setattr__ if hasattr(instance, "__setattr__") else None

    def __hook_getattr__(self, name: str):
        if name == "qtx":
            return self.__dict__["qtx"]
        elif orignal_getattr is not None:
            return orignal_getattr(name)
        else:
            # fallback
            return self.__dict__[name]

    def __hook_setattr__(self, name, value):
        if name == "qtx":
            self.__dict__["qtx"] = value
            return
        elif isinstance(value, torch.nn.Module):
            _value = patch_instance(value, context_dict)
            # in this case original_setattr exists
            orignal_setattr(name, _value)
        elif orignal_setattr is not None:
            orignal_setattr(name, value)
        else:
            # fallback
            self.__dict__[name] = value

    # avoid double patch
    if "_qtx_patched" in instance.__dict__:
        return instance

    instance.__dict__["qtx"] = context_dict
    instance.__dict__["_qtx_patched"] = True
    instance.__getattr__ = types.MethodType(__hook_getattr__, instance)
    instance.__setattr__ = types.MethodType(__hook_setattr__, instance)
    instance._orignal_getattr = orignal_getattr
    instance._orignal_setattr = orignal_setattr
    if hasattr(instance, "post_init") and callable(getattr(instance, "post_init")):
        instance.post_init()

    # tail recursive
    submodules = instance.__dict__.get("_modules", {})
    for name, submodule in submodules.items():
        submodules[name] = patch_instance(submodule, context_dict)
    return instance


def patch_cls(cls, context_dict):
    original_init = cls.__init__ if hasattr(cls, "__init__") else None
    orignal_getattr = cls.__getattr__ if hasattr(cls, "__getattr__") else None
    orignal_setattr = cls.__setattr__ if hasattr(cls, "__setattr__") else None

    def __appended__init__(self, *args, **kwargs):
        # set before init
        self.__dict__["qtx"] = context_dict
        self.__dict__["_qtx_patched"] = True

        if original_init is not None:
            original_init(self, *args, **kwargs)

    def __hook_getattr__(self, name: str):
        if name == "qtx":
            return self.__dict__["qtx"]
        elif orignal_getattr is not None:
            return orignal_getattr(self, name)
        else:
            # Fallback Strategy
            return self.__dict__[name]

    def __hook_setattr__(self, name, value):
        if name == "qtx":
            self.__dict__["qtx"] = value
            return
        if isinstance(value, torch.nn.Module):
            _value = patch_instance(value, context_dict)
            # in this case original_setattr exists
            orignal_setattr(self, name, _value)
        elif orignal_setattr is not None:
            orignal_setattr(self, name, value)
        else:
            # Fallback Strategy
            self.__dict__[name] = value

    assert inspect.isclass(cls)

    cls.__init__ = __appended__init__
    cls.__getattr__ = __hook_getattr__
    cls.__setattr__ = __hook_setattr__
    cls._original_init = original_init
    cls._orignal_getattr = orignal_getattr
    cls._orignal_setattr = orignal_setattr

    return cls


def _qContextProvider(cls: torch.nn.Module, context_dict: qt.qDict):
    """"""

    # # global hook
    # def _global_regist_module_hook(module, name, submodule):
    #     if "_qtx_patched" in module.__dict__:
    #         return patch_instance(submodule, context_dict)

    # hooks_dict = torch.nn.modules.module._global_module_registration_hooks
    # if _global_regist_module_hook not in hooks_dict.values():
    #     torch.nn.modules.module.register_module_module_registration_hook(_global_regist_module_hook)

    return patch_cls(cls, context_dict)

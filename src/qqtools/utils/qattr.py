import inspect
from typing import Optional, Type, Union


def hasattr_safe(obj, attr) -> bool:
    try:
        getattr(obj, attr)
        return True
    except AttributeError:
        return False


def getmultiattr(o: object, name: str, default=None) -> Optional[object]:
    if not name:
        return default
    for n in name.split("."):
        if hasattr(o, n):
            o = getattr(o, n)
        else:
            return default
    return o


def is_override(
    sub_class_or_instance: Union[Type, object],
    method_name: str,
    base_class: Type,
) -> bool:
    """
    Check if a method is overridden in a subclass or its inheritance chain.

    Args:
        sub_class_or_instance: Either a class (type) or an instance of a class.
                              If an instance is provided, its class will be used.
        method_name: Name of the method to check for overriding.
        base_class: The base class to compare against. The function checks if
                   the method implementation differs from this class's version.

    Returns:
        True if the method is overridden anywhere in the inheritance chain
        between the subclass and the base_class (inclusive of intermediate
        classes that inherit from base_class). False otherwise.

    Important Notes on Override Semantics:
    1. Direct Override: Returns True if the subclass directly defines the method
                       and that definition is different from base_class's version.

    2. Indirect/Inherited Override: Returns True if any class in the MRO between
                                   the subclass and base_class (excluding base_class
                                   itself) defines the method. This includes cases
                                   where an intermediate class overrides the method
                                   and the subclass inherits that override.

    3. Method from Unrelated Branch: Returns False if the method is defined in
                                    a class that doesn't inherit from base_class
                                    (e.g., in multiple inheritance scenarios where
                                    the method comes from a different parent branch).

    4. Base Class Implementation: Returns False if the first definition found in MRO
                                 is exactly in base_class, meaning no override occurs
                                 in any subclass.

    Edge Cases:
        - Returns False if base_class doesn't have the method at all.
        - Returns False for methods defined in object or other built-in classes
          unless explicitly overridden in the inheritance chain.
        - Handles properties, classmethods, and staticmethods the same as regular methods.

    """
    if isinstance(sub_class_or_instance, type):
        sub_class = sub_class_or_instance
    else:
        sub_class = sub_class_or_instance.__class__

    if not hasattr(base_class, method_name):
        return False

    mro = inspect.getmro(sub_class)
    for cls in mro:
        if method_name in cls.__dict__:
            if cls is base_class:
                return False
            if base_class in inspect.getmro(cls):
                return True
            return False

    return False

import sys
from importlib import import_module

__all__ = ["LazyImport", "is_imported", "import_common"]


def _try_import(pkg):
    try:
        _module = __import__(str(pkg))
        return _module
    except ImportError:
        return None


def is_imported(module_name: str):
    ans = module_name in sys.modules
    print(f"{module_name}{''if ans else ' not'} in sys.modules")
    return ans


class LazyImport:
    """LazyImport

    Examples:
        # Case 1: Standard module import
        np = LazyImport("numpy")
        x = np.array([1, 2, 3])

        # Case 2: Direct class import
        Path = LazyImport("pathlib", object_name="Path")
        p = Path("file.txt")
    """

    def __init__(self, module_name, object_name=None):
        self.module_name = module_name
        self.object_name = object_name
        self._target = None

    def _load_target(self):
        module = import_module(self.module_name)
        if not self.object_name:
            self._target = module
        else:
            self._target = getattr(module, self.object_name)

    def __getattr__(self, name):
        if self._target is None:
            self._load_target()
        return getattr(self._target, name)

    def __call__(self, *args, **kwargs):
        if self._target is None:
            self._load_target()
        return self._target(*args, **kwargs)


def import_common(g=None):
    if g is None:
        g = globals()
    g["torch"] = LazyImport("torch")
    g["F"] = LazyImport("torch.nn.functional")
    g["np"] = LazyImport("numpy")
    g["pd"] = LazyImport("pandas")
    g["plt"] = LazyImport("matplotlib", "plot")

    g["Path"] = LazyImport("pathlib", "Path")
    g["os"] = LazyImport("os")
    g["sys"] = LazyImport("sys")
    g["time"] = LazyImport("time")
    g["json"] = LazyImport("json")
    g["pickle"] = LazyImport("pickle")
    g["tqdm"] = LazyImport("tqdm", "tqdm")
    g["trange"] = LazyImport("tqdm", "trange")

    # typing
    g["Any"] = LazyImport("typing", "Any")
    g["Dict"] = LazyImport("typing", "Dict")
    g["Iterable"] = LazyImport("typing", "Iterable")
    g["List"] = LazyImport("typing", "List")
    g["Optional"] = LazyImport("typing", "Optional")
    g["Sequence"] = LazyImport("typing", "Sequence")
    g["Tuple"] = LazyImport("typing", "Tuple")
    g["Union"] = LazyImport("typing", "Union")

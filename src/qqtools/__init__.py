# isort:skip_file
import importlib
import sys

_lazy_exports: dict[str, tuple[str, str]] = {}


def lazy_export(module_name: str, *object_names: str) -> None:
    if not object_names:
        raise ValueError("lazy_export requires at least one object name")

    for object_name in object_names:
        if object_name in _lazy_exports:
            raise ValueError(f"lazy export {object_name!r} is already registered")
        _lazy_exports[object_name] = (module_name, object_name)


# first-class instance
from .version import __version__
from .qcontext import ctx, use_ctx

# first-class class
from .qimport import LazyImport
from .qdict import qDict

lazy_export(".data.qdatalist", "qDataList", "qList")

# -- torch related class--
lazy_export(".qtimer", "Timer")
lazy_export(".torch.qdataset", "qData", "qDictDataloader", "qDictDataset")
lazy_export(".torch.qlmdbdataset", "qLmdbDataset")
lazy_export(".torch.nn.donothing", "Donothing")

# first-class module
nn = LazyImport("qqtools.torch.nn")
qdist = LazyImport("qqtools.torch.qdist")
data = LazyImport("qqtools.data")

# first-class function
lazy_export(".qimport", "import_common")
lazy_export(".config.qssert", "batch_assert_type")
lazy_export(".config.yaml", "dump_yaml", "load_yaml")
lazy_export(".config.qpickle", "load_pickle", "save_pickle")
lazy_export(".config.qjson", "load_json", "save_json")
lazy_export(".config.qsyspath", "find_root", "update_sys")
lazy_export(".config.qlmdb", "operate_lmdb", "open_lmdb", "count_lmdb", "iter_lmdb")
lazy_export(".torch.qcontextprovider", "qContextProvider")
lazy_export(".torch.qmgraph", "qtriplets")
lazy_export(".qm.refe", "calc_refe")

# training
lazy_export(".torch.qcheckpoint", "recover", "save_ckp")
lazy_export(".torch.qgpu", "parse_device")
lazy_export(".torch.qfreeze", "freeze_rand", "freeze_module", "unfreeze_module")
lazy_export(
    ".torch.qsplit",
    "random_split_train_valid",
    "random_split_train_valid_test",
    "get_data_splits",
)
lazy_export(".torch.nn.donothing", "donothing")
lazy_export(".torch.qscatter", "scatter", "softmax")

# type & check
lazy_export(
    ".utils.qtyping",
    "Bool",
    "Float",
    "Long",
    "Float16",
    "Float32",
    "Float64",
    "Int32",
    "Int64",
    "Float32Array",
    "Float64Array",
    "BoolArray",
    "Int32Array",
    "Int64Array",
)
lazy_export(".utils.qtypecheck", "ensure_scala", "ensure_numpy", "str2number", "is_number", "is_inf")
lazy_export(".utils.check", "check_values_allowed", "is_alias_exists")

# attr
from .utils.qattr import hasattr_safe, getmultiattr, is_override

# --- optional dependencies ---
# net IO rely on `requests`
from .config.fetch.gdown import download_from_gdrive_sharelink


# plugins
def __getattr__(name):
    if name == "plugins":

        return importlib.import_module(".plugins", __name__)

    try:
        module_name, object_name = _lazy_exports[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name, __name__), object_name)
    setattr(sys.modules[__name__], name, value)
    return value

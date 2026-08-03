import importlib
import sys

from qqtools.qimport import LazyImport

_lazy_exports: dict[str, tuple[str, str]] = {}


def _lazy_export(module_name: str, *object_names: str) -> None:
    for object_name in object_names:
        _lazy_exports[object_name] = (module_name, object_name)


prepare_cmd_args = LazyImport("qqtools.plugins.qpipeline.cmd_args", "prepare_cmd_args")
create_pipeline = LazyImport("qqtools.plugins.qpipeline.entry", "create_pipeline")
get_param_stats = LazyImport("qqtools.plugins.qpipeline.entry_utils.info", "get_param_stats")
prepare_dataloder = LazyImport("qqtools.plugins.qpipeline.entry_utils.loader", "prepare_dataloder")
prepare_optimizer = LazyImport("qqtools.plugins.qpipeline.entry_utils.optimizer", "prepare_optimizer")
build_loader = LazyImport("qqtools.plugins.qpipeline.entry_utils.loader", "build_loader")
build_loader_ddp = LazyImport("qqtools.plugins.qpipeline.entry_utils.loader", "build_loader_ddp")
build_loader_trival = LazyImport("qqtools.plugins.qpipeline.entry_utils.loader", "build_loader_trival")
parse_comboloss_params = LazyImport("qqtools.plugins.qpipeline.entry_utils.loss", "parse_comboloss_params")
parse_loss_name = LazyImport("qqtools.plugins.qpipeline.entry_utils.loss", "parse_loss_name")
prepare_loss = LazyImport("qqtools.plugins.qpipeline.entry_utils.loss", "prepare_loss")
prepare_scheduler = LazyImport("qqtools.plugins.qpipeline.entry_utils.scheduler", "prepare_scheduler")
middleware_extra_ckp_caches = LazyImport("qqtools.plugins.qpipeline.middleware.ef", "middleware_extra_ckp_caches")

_lazy_export(".qpipeline", "qPipeline")
_lazy_export(".task.qtask", "PotentialTaskBase", "qTaskBase")
_lazy_export(".types", "Stage")


def __getattr__(name: str):
    try:
        module_name, object_name = _lazy_exports[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(importlib.import_module(module_name, __name__), object_name)
    setattr(sys.modules[__name__], name, value)
    return value

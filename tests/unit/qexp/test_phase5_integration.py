import importlib
import sys
import tomllib
from pathlib import Path


def _load_pyproject() -> dict:
    pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))


def test_pyproject_exposes_qexp_console_script():
    project = _load_pyproject()["project"]

    assert project["scripts"]["qexp"] == "qqtools.plugins.qexp.cli:main"


def test_pyproject_exp_extra_covers_qexp_runtime_dependencies():
    project = _load_pyproject()["project"]
    exp_dependencies = project["optional-dependencies"]["exp"]

    assert "libtmux>=0.55,<0.56" in exp_dependencies
    assert "psutil" in exp_dependencies
    assert "nvidia-ml-py" in exp_dependencies


def test_qexp_plugin_surface_is_available_via_plugins_import():
    from qqtools.plugins import qexp

    assert callable(qexp.submit)
    assert callable(qexp.cancel)
    assert callable(qexp.clean)
    assert callable(qexp.get_status_snapshot)


def test_import_qqtools_does_not_expose_root_qexp():
    import qqtools

    assert not hasattr(qqtools, "qexp")


def test_import_qqtools_plugins_qexp_remains_lazy_for_optional_runtime_dependencies():
    for module_name in ("qqtools.plugins", "qqtools.plugins.qexp"):
        sys.modules.pop(module_name, None)
    for module_name in ("libtmux", "psutil", "pynvml"):
        sys.modules.pop(module_name, None)

    qexp = importlib.import_module("qqtools.plugins.qexp")

    assert callable(qexp.submit)
    assert "libtmux" not in sys.modules
    assert "psutil" not in sys.modules
    assert "pynvml" not in sys.modules


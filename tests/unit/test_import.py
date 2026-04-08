import sys


def test_import_mypackage():
    import qqtools

    assert qqtools is not None
    print(qqtools.__file__)
    print("✅ qqtools import successfully.")
    print("✅ qqtools version:", qqtools.__version__)
    assert "libtmux" not in sys.modules
    assert "psutil" not in sys.modules
    assert "pynvml" not in sys.modules

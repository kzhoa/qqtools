import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.joinpath("src")))


def test_import_mypackage():
    import qqtools

    assert qqtools is not None
    print(qqtools.__file__)
    print("✅ qqtools import successfully.")
    print("✅ qqtools version:", qqtools.__version__)

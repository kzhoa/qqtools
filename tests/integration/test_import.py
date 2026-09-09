import os
import subprocess
import sys
from pathlib import Path


def test_import_mypackage():
    source_root = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import qqtools; "
            "assert qqtools.__version__; "
            "assert not {'libtmux', 'psutil', 'pynvml'}.intersection(sys.modules)",
        ],
        env={**os.environ, "PYTHONPATH": str(source_root)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

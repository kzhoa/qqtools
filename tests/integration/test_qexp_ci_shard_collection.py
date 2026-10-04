"""Exercise serial and xdist partition collection in real pytest processes."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("workers", [0, 2])
def test_real_pytest_shards_capture_full_collection_and_execute_once(tmp_path, workers):
    (tmp_path / "conftest.py").write_text("from tests.conftest import pytest_addoption, pytest_configure\n")
    (tmp_path / "test_cases.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value', range(8))\ndef test_value(value):\n    assert value >= 0\n"
    )
    collections = []
    for index in range(2):
        directory = tmp_path / f"reports-{index}"
        env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + str(ROOT / "src")}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(tmp_path / "test_cases.py"),
                "--rootdir",
                str(tmp_path),
                "-c",
                "/dev/null",
                "-n",
                str(workers),
                "-q",
                f"--qexp-shard={index}/2",
                f"--qexp-shard-directory={directory}",
                f"--qexp-collection-manifest={directory / 'selected.txt'}",
                f"--qexp-timing-json={directory / 'timing.json'}",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        full_manifests = list(directory.glob("full-collection-*.txt"))
        assert len(full_manifests) == (workers or 1)
        assert all(len(path.read_text().splitlines()) == 8 for path in full_manifests)
        selected = (directory / "selected.txt").read_text().splitlines()
        assert len(selected) == 4
        timing = json.loads((directory / "timing.json").read_text())
        assert len(timing["reports"]) == 12
        collections.append(set(selected))
    assert not collections[0] & collections[1]
    assert len(collections[0] | collections[1]) == 8

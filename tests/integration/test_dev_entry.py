from __future__ import annotations

import configparser
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], "2 passed"),
        (["-q"], "2 passed"),
        (["-k", "retry"], "1 passed"),
        (["tests/integration/test_other.py::test_other", "-q"], "1 passed"),
    ],
)
def test_unit_lane_uses_pytest_parser_for_default_scope(tmp_path, args, expected):
    for directory, content in {
        "unit": "def test_retry(): pass\ndef test_normal(): pass\n",
        "integration": "def test_other(): pass\n",
    }.items():
        target = tmp_path / "tests" / directory / "test_other.py"
        target.parent.mkdir(parents=True)
        target.write_text(content)
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    command = config["testenv:unit"]["commands"].strip().replace("{posargs:-q}", shlex.join(args or ["-q"]))
    result = subprocess.run([sys.executable, *shlex.split(command)[1:]], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
for module_name in list(sys.modules.keys()):
    if module_name == "qqtools" or module_name.startswith("qqtools."):
        sys.modules.pop(module_name)

from qqtools.plugins.qpipeline.runner.progress import resolve_render_mode


@pytest.mark.parametrize(
    "requested_mode,has_rich,has_tqdm,expected",
    [
        (None, True, True, "rich"),
        ("auto", True, True, "rich"),
        ("auto", False, True, "tqdm"),
        ("auto", False, False, "plain"),
        ("rich", False, True, "tqdm"),
        ("tqdm", False, False, "plain"),
    ],
)
def test_resolve_render_mode(requested_mode, has_rich, has_tqdm, expected):
    assert resolve_render_mode(requested_mode, has_rich, has_tqdm) == expected

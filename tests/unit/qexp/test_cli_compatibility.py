from __future__ import annotations

import pytest

from qqtools.plugins.qexp.cli import main


@pytest.mark.parametrize(
    "argv",
    [
        ["--shared-root", "/tmp/project/.qexp", "--machine", "gpu-1", "agent", "add-project", "--adopt-existing"],
        ["--shared-root", "/tmp/project/.qexp", "--machine", "gpu-1", "agent", "remove-project", "project-id"],
        ["--shared-root", "/tmp/project/.qexp", "--machine", "gpu-1", "agent", "migrate-project"],
    ],
)
def test_exact_retired_agent_argv_get_bounded_compatibility_diagnostic(argv: list[str], capsys) -> None:
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert "QQTOOLS-COMPAT-0014" in captured.err
    assert "--shared-root" in captured.err
    assert "--project" in captured.err
    assert captured.out == ""

from pathlib import Path

import pytest

from qqtools.plugins.qexp.cli.project_presentation import abbreviate_home, encode_display_text, format_project_line


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain α", "plain α"),
        ("back\\slash", r"back\\slash"),
        ("line\ncarriage\rtab\t", r"line\ncarriage\rtab\t"),
        ("\x00\x01\x1b\x7f\x80\x9f", r"\x00\x01\x1b\x7f\x80\x9f"),
        ("\u2028\u2029\u200b\U000e0001", r"\u2028\u2029\u200b\U000e0001"),
    ],
)
def test_encode_display_text_keeps_every_path_on_one_line(value: str, expected: str) -> None:
    encoded = encode_display_text(value)
    assert encoded == expected
    assert "\n" not in encoded
    assert "\r" not in encoded
    assert "\t" not in encoded


def test_generated_escape_backslashes_are_not_reescaped() -> None:
    assert encode_display_text("\\\n") == r"\\\n"


def test_abbreviate_home_uses_only_the_selected_home_tree(tmp_path: Path) -> None:
    home = tmp_path / "home"
    assert abbreviate_home(home, home=home) == "~"
    assert abbreviate_home(home / ".qqtools" / "qexp-context.json", home=home) == ("~/.qqtools/qexp-context.json")
    assert abbreviate_home(tmp_path / "other", home=home) == str((tmp_path / "other").resolve())


@pytest.mark.parametrize(
    ("source", "suffix"),
    [
        (None, ""),
        ("explicit", ""),
        ("cli", ""),
        ("environment", " (from $QEXP_SHARED_ROOT)"),
        ("manifest_ancestor", " (from manifest directory)"),
    ],
)
def test_format_project_line_maps_non_context_sources(tmp_path: Path, source: str | None, suffix: str) -> None:
    project = tmp_path / "project"
    assert format_project_line(project, source=source) == f"Project: {project.resolve()}{suffix}"


def test_format_project_line_distinguishes_cwd_from_parent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    child = project / "runs" / "one"
    assert format_project_line(project, source="cwd", invocation_cwd=project) == f"Project: {project.resolve()}"
    assert format_project_line(project, source="cwd_ancestor", invocation_cwd=child) == (
        f"Project: {project.resolve()} (from parent directory)"
    )


def test_format_project_line_uses_actual_abbreviated_context_file(tmp_path: Path) -> None:
    home = Path.home().resolve()
    context_file = home / ".qqtools" / "qexp-context.json"
    assert format_project_line(tmp_path / "project", source="saved", context_file=context_file).endswith(
        " (from ~/.qqtools/qexp-context.json)"
    )


@pytest.mark.parametrize(
    ("source", "kwargs"),
    [
        ("cwd", {}),
        ("cwd_ancestor", {}),
        ("saved", {}),
        ("unknown", {}),
    ],
)
def test_format_project_line_rejects_missing_context_or_unknown_source(source: str, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        format_project_line(Path("project"), source=source, **kwargs)

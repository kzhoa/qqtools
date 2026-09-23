import pytest

from qqtools.plugins.qexp.manifest import normalize_command_submission, parse_submission_manifest


def test_manifest_live_progress_precedence_and_inheritance(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        """
defaults:
  live_progress: true
tasks:
  - command: [python, one.py]
  - command: [python, two.py]
    live_progress: false
  - command: [python, three.py]
    live_progress: null
""",
        encoding="utf-8",
    )
    inherited = parse_submission_manifest(path)
    assert [item["live_progress"] for item in inherited.specs] == [True, False, True]

    overridden = parse_submission_manifest(path, live_progress_override=False)
    assert [item["live_progress"] for item in overridden.specs] == [False, False, False]
    assert [item["live_progress"] for item in overridden.field_sources] == ["cli", "cli", "cli"]


def test_omitted_live_progress_keeps_legacy_raw_shape(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text("tasks:\n  - command: [python, train.py]\n", encoding="utf-8")
    manifest = parse_submission_manifest(path)
    assert "live_progress" not in manifest.specs[0]
    command, _ = normalize_command_submission(["python", "train.py"])
    assert "live_progress" not in command


@pytest.mark.parametrize("value", ["true", 1, 0, [], {}])
def test_manifest_rejects_non_boolean_live_progress(tmp_path, value):
    path = tmp_path / "tasks.yaml"
    path.write_text(f"tasks:\n  - command: [python, train.py]\n    live_progress: {value!r}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="live_progress"):
        parse_submission_manifest(path)


def test_manifest_rejects_duplicate_live_progress(tmp_path):
    path = tmp_path / "tasks.yaml"
    path.write_text(
        "tasks:\n  - command: [python, train.py]\n    live_progress: true\n    live_progress: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_submission_manifest(path)

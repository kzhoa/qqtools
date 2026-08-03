import csv
from threading import Event

import pytest

from qqtools.plugins.qpipeline.runner.runner_utils.sheet_logger import (
    SheetLogger,
    adapt_qpipeline_metric_row,
)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


def test_csv_schema_expands_in_observed_order(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = SheetLogger(path, columns=["epoch", "global_step"])

    logger.write({"epoch": 1, "global_step": 10, "loss": 0.4, "f1-score": 0.8})
    logger.write({"epoch": 2, "global_step": 20, "accuracy": 0.9})
    logger.close()

    assert _read_csv(path) == [
        ["epoch", "global_step", "loss", "f1-score", "accuracy"],
        ["1", "10", "0.4", "0.8", ""],
        ["2", "20", "", "", "0.9"],
    ]


def test_recover_preserves_existing_schema_and_adds_initial_columns(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("old,dynamic\n1,2\n", encoding="utf-8")

    logger = SheetLogger(path, columns=["epoch", "old", "global_step"], recover=True)
    logger.write({"epoch": 3, "old": 4, "global_step": 30})
    logger.close()

    assert _read_csv(path) == [
        ["old", "dynamic", "epoch", "global_step"],
        ["1", "2", "", ""],
        ["4", "", "3", "30"],
    ]


def test_recover_rejects_ragged_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("epoch,loss\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row width"):
        SheetLogger(path, columns=["epoch"], recover=True)


def test_recover_false_preserves_old_file_until_first_commit_or_close(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text("old\n1\n", encoding="utf-8")

    logger = SheetLogger(path, columns=["epoch"], recover=False)
    logger.abort()
    assert path.read_text(encoding="utf-8") == "old\n1\n"

    logger = SheetLogger(path, columns=["epoch"], recover=False)
    logger.close()
    assert _read_csv(path) == [["epoch"]]


def test_async_writer_serializes_schema_updates(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = SheetLogger(path, columns=["epoch"], async_write=True, buffer_size=2)
    logger.write({"epoch": 1, "a": 1})
    logger.write({"epoch": 2, "b": 2})
    logger.close()

    assert _read_csv(path) == [["epoch", "a", "b"], ["1", "1", ""], ["2", "", "2"]]


def test_async_abort_commits_rows_accepted_before_abort(tmp_path):
    path = tmp_path / "metrics.csv"
    first_row_started = Event()
    release_first_row = Event()

    def adapt_row(row):
        if row["epoch"] == 1:
            first_row_started.set()
            assert release_first_row.wait(timeout=5)
        return dict(row)

    logger = SheetLogger(path, columns=["epoch"], async_write=True, row_adapter=adapt_row)
    logger.write({"epoch": 1})
    assert first_row_started.wait(timeout=5)
    logger.write({"epoch": 2})
    release_first_row.set()
    logger.abort()

    assert _read_csv(path) == [["epoch"], ["1"], ["2"]]


def test_async_abort_does_not_materialize_an_empty_run(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = SheetLogger(path, columns=["epoch"], async_write=True, recover=False)

    logger.abort()

    assert not path.exists()


def test_metric_adapter_validates_keys_and_scalar_values():
    assert adapt_qpipeline_metric_row({"val/accuracy": 0.5}) == {"val/accuracy": 0.5}
    with pytest.raises(ValueError, match="Invalid metric key"):
        adapt_qpipeline_metric_row({" bad": 1})
    with pytest.raises(TypeError, match="scalar"):
        adapt_qpipeline_metric_row({"loss": [1, 2]})

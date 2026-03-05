from types import SimpleNamespace

from qqtools.plugins.qpipeline.runner.eval_summary_formatter import EvalSummaryFormatter


def test_table_marks_time_target_in_main_cell():
    eval_results = {
        "train_batch_time": 0.3,
        "train_loss": 0.2,
        "train_metric": 0.1,
        "val_metric": 0.5,
    }

    _, _, table_lines, _ = EvalSummaryFormatter.format_all(
        eval_results=eval_results,
        epoch=1,
        step=10,
        target_key="train_batch_time",
        target_mode="min",
        is_best=False,
        previous_best={"metric": 0.29, "epoch": 0, "step": 0},
        best_model_tracker=SimpleNamespace(mode="min", best_metric=0.29, best_epoch=0, best_step=0),
        color_new_best=False,
    )

    batch_time_row = next(line for line in table_lines if line.strip().startswith("batch_time"))
    assert "(*)" in batch_time_row
    assert "(*) marks the primary target cell in table." in table_lines

    others_line = next((line for line in table_lines if line.startswith("Others:")), "")
    assert "train_batch_time" not in others_line


def test_table_marks_non_stage_target_in_others():
    eval_results = {
        "custom_score": 0.9,
        "train_loss": 0.2,
        "train_metric": 0.1,
        "val_metric": 0.5,
    }

    _, _, table_lines, _ = EvalSummaryFormatter.format_all(
        eval_results=eval_results,
        epoch=2,
        step=20,
        target_key="custom_score",
        target_mode="max",
        is_best=False,
        previous_best={"metric": 0.88, "epoch": 1, "step": 10},
        best_model_tracker=SimpleNamespace(mode="max", best_metric=0.95, best_epoch=1, best_step=10),
        color_new_best=False,
    )

    others_line = next(line for line in table_lines if line.startswith("Others:"))
    assert "custom_score: 0.9000 (*)" in others_line
    assert "(*) marks the primary target metric in Others." in table_lines

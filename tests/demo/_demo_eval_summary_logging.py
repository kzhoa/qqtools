import logging
from types import SimpleNamespace

from qqtools.plugins.qpipeline.qlogger import ConsoleLogger
from qqtools.plugins.qpipeline.runner.runner_utils.eval_formatter import EvalSummaryListener
from qqtools.plugins.qpipeline.runner.runner_utils.types import EventContext, RunningState


def _make_eval_results(val_metric: float, ema_val_metric: float):
    return {
        "ema_test_mae": 0.0234,
        "ema_test_metric": 0.0234,
        "ema_test_mse": 0.0011,
        "ema_val_mae": ema_val_metric,
        "ema_val_metric": ema_val_metric,
        "ema_val_mse": 0.0012,
        "test_mae": 0.0642,
        "test_metric": 0.0642,
        "test_mse": 0.0058,
        "train_batch_time": 0.2815,
        "train_loss": 0.0086,
        "train_metric": 0.0739,
        "val_mae": val_metric,
        "val_metric": val_metric,
        "val_mse": 0.0059,
    }


def _build_context(
    state: RunningState,
    eval_results,
    is_best: bool,
    previous_best,
    best_model_tracker,
) -> EventContext:
    return EventContext(
        state=state,
        stage="eval",
        eval_results=eval_results,
        is_best=is_best,
        previous_best=previous_best,
        best_model_tracker=best_model_tracker,
    )


def main():
    logger = ConsoleLogger(level=logging.INFO)
    listener = EvalSummaryListener(logger=logger, target_key="val_metric", target_mode="min", save_dir="./logs")

    print("\n========== Scenario 1: NOT_BEST ==========")
    state = RunningState(epoch=23, global_step=82487)
    best_model_tracker = SimpleNamespace(mode="min", best_metric=0.0612, best_epoch=20, best_step=71111)
    previous_best = {"metric": 0.0612, "epoch": 20, "step": 71111}
    eval_results = _make_eval_results(val_metric=0.0645, ema_val_metric=0.0237)
    listener.on_validation_end(
        _build_context(
            state=state,
            eval_results=eval_results,
            is_best=False,
            previous_best=previous_best,
            best_model_tracker=best_model_tracker,
        )
    )

    print("\n========== Scenario 2: NEW_BEST ==========")
    state = RunningState(epoch=24, global_step=86000)
    best_model_tracker = SimpleNamespace(mode="min", best_metric=0.0581, best_epoch=24, best_step=86000)
    previous_best = {"metric": 0.0612, "epoch": 20, "step": 71111}
    eval_results = _make_eval_results(val_metric=0.0581, ema_val_metric=0.0219)
    listener.on_validation_end(
        _build_context(
            state=state,
            eval_results=eval_results,
            is_best=True,
            previous_best=previous_best,
            best_model_tracker=best_model_tracker,
        )
    )


if __name__ == "__main__":
    main()

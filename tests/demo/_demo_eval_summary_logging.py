import logging
from types import SimpleNamespace

from qqtools.plugins.qpipeline.qlogger import ConsoleLogger
from qqtools.plugins.qpipeline.runner.runner import RunningAgent


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


def _build_agent() -> RunningAgent:
    agent = RunningAgent.__new__(RunningAgent)
    agent.logger = ConsoleLogger(level=logging.INFO)
    agent.config = SimpleNamespace(checkpoint={"target": "val_metric", "mode": "min"})
    agent.state = SimpleNamespace(epoch=0, global_step=0)
    return agent


def main():
    agent = _build_agent()

    print("\n========== Scenario 1: NOT_BEST ==========")
    agent.state.epoch = 23
    agent.state.global_step = 82487
    agent.best_model_tracker = SimpleNamespace(mode="min", best_metric=0.0612, best_epoch=20, best_step=71111)
    previous_best = {"metric": 0.0612, "epoch": 20, "step": 71111}
    eval_results = _make_eval_results(val_metric=0.0645, ema_val_metric=0.0237)
    agent._log_eval_summary(eval_results, is_best=False, previous_best=previous_best)

    print("\n========== Scenario 2: NEW_BEST ==========")
    agent.state.epoch = 24
    agent.state.global_step = 86000
    agent.best_model_tracker = SimpleNamespace(mode="min", best_metric=0.0581, best_epoch=24, best_step=86000)
    previous_best = {"metric": 0.0612, "epoch": 20, "step": 71111}
    eval_results = _make_eval_results(val_metric=0.0581, ema_val_metric=0.0219)
    agent._log_eval_summary(eval_results, is_best=True, previous_best=previous_best)


if __name__ == "__main__":
    main()

import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Ensure repository root is on sys.path so `src` package can be imported
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from qqtools.plugins.qpipeline.runner import progress as progress_mod


class DummyState:
    def __init__(self):
        self.epoch = 0
        self.max_epochs = 1


class DummyContext(SimpleNamespace):
    pass


def simulate(strategy):
    state = DummyState()
    total_batches = 300
    ctx = DummyContext(
        state=state,
        total_batches=total_batches,
        batch_idx=0,
        batch_metrics={},
        avg_bank={},
        lr=None,
        stage="train",
        eval_results=None,
    )
    strategy.on_epoch_start(ctx)
    for i in range(total_batches):
        ctx.batch_idx = i
        ctx.batch_metrics = {"loss": 0.1 * (i + 1)}
        ctx.avg_bank = {"loss": 0.1 * (i + 1)}
        strategy.on_batch_end(ctx)
        time.sleep(0.03)
        if i == 100:
            print("\n--- START EVAL ---\n")
            # Show pre-eval completed count if available
            try:
                if hasattr(strategy, "pbar") and getattr(strategy, "pbar") is not None:
                    print("Pre-eval pbar.n:", getattr(strategy.pbar, "n", None))
                if hasattr(strategy, "displayer") and getattr(strategy.displayer, "progress", None) is not None:
                    pid = getattr(strategy.displayer, "progress_task_id", None)
                    if pid is not None:
                        try:
                            tasks = list(strategy.displayer.progress.tasks)
                            print("Pre-eval rich completed:", tasks[0].completed if tasks else None)
                        except Exception:
                            pass
            except Exception:
                pass

            strategy.on_eval_start(ctx)
            # Simulate evaluation logging: prefer Rich Console.print when available
            try:
                if hasattr(strategy, "displayer") and getattr(strategy.displayer, "console", None) is not None:
                    strategy.displayer.console.print("Eval logs: metric=0.123")
                else:
                    print("Eval logs: metric=0.123")
            except Exception:
                print("Eval logs: metric=0.123")
            time.sleep(0.6)
            strategy.on_eval_end(ctx)

            # Show post-eval completed count if available
            try:
                if hasattr(strategy, "pbar") and getattr(strategy, "pbar") is not None:
                    print("Post-eval pbar.n:", getattr(strategy.pbar, "n", None))
                if hasattr(strategy, "displayer") and getattr(strategy.displayer, "progress", None) is not None:
                    pid = getattr(strategy.displayer, "progress_task_id", None)
                    if pid is not None:
                        try:
                            tasks = list(strategy.displayer.progress.tasks)
                            print("Post-eval rich completed:", tasks[0].completed if tasks else None)
                        except Exception:
                            pass
            except Exception:
                pass

            print("\n--- END EVAL ---\n")
            time.sleep(0.1)
    strategy.on_epoch_end(ctx)
    strategy.on_run_end()


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("progress_demo")

    # Choose available strategy
    if getattr(progress_mod, "HAS_RICH", False):
        print("Using RichProgress (HAS_RICH=True)")
        strat = progress_mod.RichProgress()
    elif getattr(progress_mod, "HAS_TQDM", False):
        print("Using TqdmProgress (HAS_TQDM=True)")
        strat = progress_mod.TqdmProgress()
    else:
        print("Using PlainProgress")
        strat = progress_mod.PlainProgress(logger)

    simulate(strat)


if __name__ == "__main__":
    main()

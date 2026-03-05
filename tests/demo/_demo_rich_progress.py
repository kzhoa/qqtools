import time
import logging
from qqtools.plugins.qpipeline.runner.runner_utils.progress import ProgressTracker
from qqtools.plugins.qpipeline.runner.runner_utils.types import EventContext, RunningState
from qqtools.plugins.qpipeline.qlogger import ConsoleLogger


def main():
    # 1. Setup Logger with RichHandler
    logger = ConsoleLogger(level=logging.INFO)
    logger.info("Demo starting...")

    # 2. Setup ProgressTracker
    tracker = ProgressTracker(logger=logger, print_freq=10, render_type="rich")
    state = RunningState()

    # 3. Simulate Training Loop
    total_epochs = 10
    batches_per_epoch = 50

    for epoch in range(total_epochs):
        # Fire Epoch Start
        state.epoch = epoch
        ctx = EventContext(state=state, total_batches=batches_per_epoch, stage="train")
        tracker.on_epoch_start(ctx)

        for batch_idx in range(batches_per_epoch):
            time.sleep(0.05)  # Simulate work

            # Fire Progress Tick
            ctx.batch_idx = batch_idx
            ctx.batch_metrics = {"loss": 0.5 - batch_idx * 0.001, "acc": 0.1 + batch_idx * 0.01}
            ctx.avg_bank = {"loss": 0.4}
            ctx.lr = 0.001
            tracker.on_progress_tick(ctx)

            # Simulate log output pushing live down
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}: Random log message to push scroll")

            # Fire Table Update
            if batch_idx % 10 == 0:
                tracker.on_table_update(ctx)

        # Fire Epoch End
        tracker.on_epoch_end(ctx)
        logger.info(f"Epoch {epoch} finished!")
        time.sleep(1)

    tracker.on_run_end()


if __name__ == "__main__":
    main()

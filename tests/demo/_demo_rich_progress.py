import time
import logging
from qqtools.plugins.qpipeline.runner.contracts import EpochCommittedFact, EpochStartedFact, ProgressTickFact
from qqtools.plugins.qpipeline.runner.runner_utils.progress import ProgressTracker
from qqtools.plugins.qpipeline.types import Stage
from qqtools.plugins.qpipeline.qlogger import ConsoleLogger


def main():
    # 1. Setup Logger with RichHandler
    logger = ConsoleLogger(level=logging.INFO)
    logger.info("Demo starting...")

    # 2. Setup ProgressTracker
    tracker = ProgressTracker(logger=logger, print_freq=10, render_type="rich")

    # 3. Simulate Training Loop
    total_epochs = 10
    batches_per_epoch = 50

    for epoch in range(total_epochs):
        # Fire Epoch Start
        tracker.on_epoch_start(
            EpochStartedFact(
                epoch=epoch,
                global_step=epoch * batches_per_epoch,
                total_batches=batches_per_epoch,
            )
        )

        for batch_idx in range(batches_per_epoch):
            time.sleep(0.05)  # Simulate work

            # Fire Progress Tick
            ctx = ProgressTickFact(
                stage=Stage.TRAIN,
                epoch=epoch,
                global_step=epoch * batches_per_epoch + batch_idx,
                batch_index=batch_idx,
                total_batches=batches_per_epoch,
                batch_metrics={"loss": 0.5 - batch_idx * 0.001, "acc": 0.1 + batch_idx * 0.01},
                average_metrics={"loss": 0.4},
                lr=0.001,
            )
            tracker.on_progress_tick(ctx)

            # Simulate log output pushing live down
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}: Random log message to push scroll")

            # Fire Table Update
            if batch_idx % 10 == 0:
                tracker.on_table_update(ctx)

        # Fire Epoch End
        tracker.on_epoch_end(
            EpochCommittedFact(
                completed_epoch=epoch,
                next_epoch=epoch + 1,
                global_step=(epoch + 1) * batches_per_epoch,
                epoch_metrics={},
            )
        )
        logger.info(f"Epoch {epoch} finished!")
        time.sleep(1)

    tracker.on_run_end()


if __name__ == "__main__":
    main()

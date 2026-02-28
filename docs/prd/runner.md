
# Runner




# Functional Specifications

## 1 Unified Training Loop

- Description: A single training loop that progresses by step. It automatically handles epoch boundaries and data iteration.
- Key Control Parameters:
  - run_mode('epoch'| 'step'): Determines how eval_intervalis interpreted.
  - eval_interval(int): How frequently to run evaluation. Interpreted as epochsif run_mode='epoch', or as stepsif run_mode='step'.
  - max_epochs/ max_steps(Optional[int]): Stopping conditions.
- Internal Logic:
    - Fetch next batch (handles epoch rollover automatically).
    - Forward/backward pass, optimizer step.
    - Evaluation Trigger: Checks if epoch % eval_interval == 0(epoch mode) or global_step % eval_interval == 0(step mode).
    - Regular Save Trigger: Checks if save_intervalis set and global_step % save_interval == 0.
    - Repeat until max_epochs/ max_stepsreached or early stopping triggers.

## 2 Evaluation & Checkpointing

- Evaluation: Runs on the validation (and optionally test) dataset when triggered. Can evaluate both the base model and an optional EMA (Exponential Moving Average) model.
- Checkpoint Types:
    - Best Model: Saved automatically after evaluation if metrics improve. Only the latest best model is kept.
    - Regular Checkpoint: Saved at intervals defined by save_interval(in steps). All are kept.
- Checkpoint Contents: Model state (raw + EMA), optimizer state, scheduler state, full training state (epoch, step, metrics), and task-specific state.


## 3 Early Stopping

- Configuration: Defined via `early_stop` dict (target, patience, mode, min_delta).
- Logic: Monitors the target metric (e.g., `val_metric`). Counter increments if no improvement (exceeding min_delta) is seen. Training stops when counter >= patience.


## 4 Advanced Features (Integrated)

- Gradient Clipping: Via clip_gradparameter.
- EMA Model: Via use_emaand ema_decayparameters.
- Learning Rate Scheduling: Supports standard torch.optim.lr_schedulerand custom qWarmupScheduler.
- Distributed Training: Basic support via distributedand rankparameters.
- Profiling: Integrated PyTorch profiler via use_profiler.
- Event System: Listener hooks (on_epoch_end, on_batch_end, etc.) for non-invasive extensions.





# tests

- Test basic training loop
- Test epoch mode
- Test step mode
- Test early stopping functionality
- Test checkpoint save/load
- Test EMA (Exponential Moving Average)
- Test the distributedparameter
- Test profiler

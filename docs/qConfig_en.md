# `qConfig` Configuration File Specification

This document provides detailed explanations of all fields, types, default values, and usage specifications for `qConfig` training configuration files (train.yml).

## Table of Contents

1. [Quick Example](#quick-example)
2. [Top-Level Fields Overview](#top-level-fields-overview)
3. [optim - Optimizer Configuration](#optim---optimizer-configuration)
4. [runner - Training Runner Configuration](#runner---training-runner-configuration)
5. [task - Task Configuration](#task---task-configuration)
6. [Complete Examples](#complete-examples)
7. [Configuration Field Interactions](#configuration-field-interactions)

---

## Quick Example

Here is a complete minimal configuration file example:

```yaml
seed: 42
log_dir: ./logs/experiment_001
print_freq: 10

optim:
  loss: mse
  optimizer: adamw
  optimizer_params:
    lr: 1.0e-3
    weight_decay: 1.0e-4
  scheduler: cosine
  scheduler_params:
    T_max: 100
    eta_min: 1.0e-6
  warmup_params:
    warmup_steps: 1000
    warmup_factor: 0.01

runner:
  run_mode: epoch
  max_epochs: 100
  eval_interval: 1
  clip_grad: 1.0
  early_stop:
    target: val_loss
    patience: 20
    mode: min

task:
  dataset: qm9
  dataloader:
    batch_size: 32
    num_workers: 4
    pin_memory: true

model:
  # model field is completely free to use
  # Users should fill in all configurations needed to initialize the model here
  # The content of this field is completely determined by the specific model implementation
  num_layers: 4
  hidden_dim: 128
  dropout: 0.1
```

### Explanation

- **seed**: Random seed for reproducibility
- **log_dir**: Directory for saving logs and checkpoints
- **print_freq**: Batch interval for printing logs
- **optim**: Optimizer, loss function, and learning rate schedule configuration (see below)
- **runner**: Training run parameters and early stopping configuration (see below)
- **task**: Task configuration including dataset and data loading (see below)
- **model**: **Completely free field**. Users should fill this with all configurations needed to initialize the model. The content is determined by the specific model implementation; the framework does not impose restrictions.

---

## Top-Level Fields Overview

Configuration files contain both user-specified fields and framework-reserved fields that are automatically set.

### User-Specified Fields

| Field         | Required | Type    | Description                                                                         |
| ------------- | -------- | ------- | ----------------------------------------------------------------------------------- |
| `seed`        | ✅        | Integer | Random seed for reproducibility                                                     |
| `log_dir`     | ✅        | String  | Directory for saving logs and checkpoints                                           |
| `optim`       | ✅        | Object  | Optimizer configuration (loss, optimizer, scheduler)                                |
| `runner`      | ✅        | Object  | Training runner configuration (mode, epochs, eval)                                  |
| `task`        | ✅        | Object  | Task configuration (dataset, data loading)                                          |
| `print_freq`  | ❌        | Integer | Batch interval for printing logs (default: 10)                                      |
| `ckp_file`    | ❌        | String  | Checkpoint file path for resuming training                                          |
| `init_file`   | ❌        | String  | Initialization file path for pretrained weights                                     |
| `render_type` | ❌        | String  | Log and progress bar rendering style (default: auto)                                |
| `model`       | ❌        | Object  | Entirely user-defined; the framework does not read or process this field in any way |

### Framework-Reserved Fields (Automatically Set)

The following fields are automatically set by the framework based on the runtime environment and `args` parameter. Users do not need to configure them:

| Field         | Type    | Description                                                 |
| ------------- | ------- | ----------------------------------------------------------- |
| `device`      | String  | Computing device (cuda/cpu), automatically selected         |
| `rank`        | Integer | Process rank in distributed training (0 for single machine) |
| `distributed` | Boolean | Whether distributed training is used, auto-determined       |

---

- **Relation**: Mutually exclusive with `init_file`; Priority: `ckp_file` > `init_file`
- **Note**: File must exist, otherwise error will occur

### init_file (Model Initialization)

```yaml
init_file: ./pretrained_weights.pt
```

- **Type**: String or null
- **Default**: null
- **Description**: Initialize model with pretrained weights without restoring optimizer state
- **Effect**: Only loads model weights; optimizer and scheduler start fresh
- **Relation**: Mutually exclusive with `ckp_file`; if both are specified, `ckp_file` takes priority
- **Use Case**: Transfer learning, fine-tuning pretrained models

### render_type (Log and Progress Bar Rendering)

```yaml
render_type: auto # Default value
```

- **Type**: String
- **Default**: `auto`
- **Supported Values**:
  - `auto`: Automatically choose based on environment (tqdm if available, else plain text)
  - `plain`: Plain text output only
  - `tqdm`: Progress bar using tqdm library
  - `rich`: Rich formatted output with colors and progress bars (requires `rich` library)

- **Description**: Controls how logs and progress bars are displayed during training
- **Recommendation**:
  - `auto`: Suitable for most cases
  - `rich`: For better visual experience in interactive environments
  - `tqdm`: For standard progress bar display
  - `plain`: For batch jobs or when tqdm/rich are not available

---

## optim - Optimizer Configuration

The optimizer configuration includes loss function, optimizer, learning rate scheduler, and warmup strategy.

### Fields Summary

| Field              | Required | Type             | Description                           |
| ------------------ | -------- | ---------------- | ------------------------------------- |
| `loss`             | ✅        | String or Object | Loss function                         |
| `optimizer`        | ✅        | String           | Optimizer type                        |
| `optimizer_params` | ✅        | Object           | Optimizer parameters                  |
| `scheduler`        | ❌        | String           | Learning rate scheduler               |
| `scheduler_params` | ❌        | Object           | Scheduler parameters                  |
| `warmup_params`    | ❌        | Object           | Warmup parameters                     |
| `ema_params`       | ❌        | Object           | Exponential Moving Average parameters |

### loss (Loss Function)

```yaml
# Single loss function
optim:
  loss: mse
```

```yaml
# Composite loss function
optim:
  loss: comboloss
  loss_params:
    energy: ["mse", 1.0]
    force: ["l1", 0.5]
```

- **Type**: String or Dictionary
- **Supported Loss Functions**:
  - `mse`: Mean Squared Error
  - `mae` / `l1`: Mean Absolute Error
  - `l2mae`: L2 Norm Mean Absolute Error
  - `bce`: Binary Cross Entropy
  - `ce` / `cross_entropy`: Cross Entropy
  - `focal` / `focal_loss`: Focal Loss
  - `comboloss`: Weighted composite loss

#### comboloss - Weighted Composite Loss

`comboloss` is used for multi-target or multi-task learning, allowing separate loss function and weight configuration for different targets/tasks.

**Usage Example**:

```yaml
optim:
  loss: comboloss
  loss_params:
    energy: ["mse", 1.0] # Energy target, uses MSE, weight 1.0
    force: ["l1", 0.5] # Force field target, uses L1, weight 0.5
    dipole: ["mae", 0.2] # Dipole moment target, uses MAE, weight 0.2
```

**Format Specification**:

- Format: `{target_name: [loss_name, weight]}`
- `target_name`: Target/task name (user-defined)
- `loss_name`: Loss function name for this target
- `weight`: Weight of this target loss in the final loss

**Calculation**:
$$\text{Total Loss} = w_1 \times L_1 + w_2 \times L_2 + w_3 \times L_3$$

where $w_i$ is the weight and $L_i$ is the loss for each target.

### optimizer (Optimizer Type)

```yaml
optim:
  optimizer: adamw
  optimizer_params:
    lr: 1.0e-3
```

- **Type**: String
- **Supported Optimizers**:
  - `adamw`: AdamW (Recommended)
  - `adam`: Adam
  - `sgd`: Stochastic Gradient Descent
  - `rmsprop`: RMSprop

- **Recommendation**:
  - Most scenarios: `adamw`
  - Computation constrained: `sgd`
  - Sparse gradients: `adam`

### optimizer_params (Optimizer Parameters)

#### Common Parameters

```yaml
optim:
  optimizer_params:
    lr: 1.0e-3 # Learning rate (Required)
    weight_decay: 1.0e-4 # L2 regularization
```

| Parameter      | Type  | Range   | Required | Description                                  |
| -------------- | ----- | ------- | -------- | -------------------------------------------- |
| `lr`           | Float | ≥ 1e-8  | ✅        | Base learning rate                           |
| `weight_decay` | Float | ≥ 0     | ❌        | L2 regularization coefficient (default 0)    |
| `eps`          | Float | ≥ 1e-10 | ❌        | Numerical stability term (Adam default 1e-8) |

#### Adam-Specific Parameters

```yaml
optim:
  optimizer_params:
    lr: 1.0e-3
    betas: [0.9, 0.999] # (beta1, beta2)
    eps: 1.0e-9
    amsgrad: false # Use AMSGrad variant
```

| Parameter | Type    | Range      | Default      | Description                                              |
| --------- | ------- | ---------- | ------------ | -------------------------------------------------------- |
| `betas`   | Array   | [0-1, 0-1] | [0.9, 0.999] | Exponential decay rates for 1st and 2nd moment estimates |
| `amsgrad` | Boolean | -          | false        | Use AMSGrad variant (more stable)                        |

#### SGD-Specific Parameters

```yaml
optim:
  optimizer_params:
    lr: 1.0e-2
    momentum: 0.9
```

| Parameter  | Type  | Range | Description                      |
| ---------- | ----- | ----- | -------------------------------- |
| `momentum` | Float | ≥ 0   | Momentum coefficient (default 0) |

### scheduler (Learning Rate Scheduler)

Learning rate schedulers automatically adjust the learning rate during training.

```yaml
optim:
  scheduler: cosine
  scheduler_params:
    T_max: 100
    eta_min: 1.0e-6
```

- **Type**: String
- **Supported Schedulers**:
  - `cosine`: Cosine Annealing
  - `step`: Step Decay
  - `plateau`: Reduce on Plateau (Recommended for val_metric)
  - `multi_step`: Multi-step Decay
  - `lambda`: Custom Lambda Function Decay

#### scheduler_params (Scheduler Parameters)

##### Cosine Annealing

```yaml
optim:
  scheduler: cosine
  scheduler_params:
    T_max: 100 # Cosine period (usually equals max_epochs)
    eta_min: 1.0e-6 # Minimum learning rate
```

| Parameter | Type    | Range | Default | Description                                |
| --------- | ------- | ----- | ------- | ------------------------------------------ |
| `T_max`   | Integer | ≥ 1   | None    | Period length, usually set to `max_epochs` |
| `eta_min` | Float   | ≥ 0   | 0       | Minimum learning rate                      |

- **Effect**: Learning rate decays from initial value to `eta_min` following cosine curve over `T_max` epochs
- **Applicable For**: Tasks requiring gradual learning rate reduction

##### Step LR

```yaml
optim:
  scheduler: step
  scheduler_params:
    step_size: 30 # Decay every 30 epochs
    gamma: 0.1 # Decay factor
```

| Parameter   | Type    | Range | Default | Description                             |
| ----------- | ------- | ----- | ------- | --------------------------------------- |
| `step_size` | Integer | ≥ 1   | 30      | Decay period (epochs)                   |
| `gamma`     | Float   | 0-1   | 0.1     | Learning rate multiplied by this factor |

- **Effect**: Every `step_size` epochs, multiply learning rate by `gamma`
- **Example**: If lr=0.1, step_size=30, gamma=0.1, then at epoch 30, lr becomes 0.01

##### MultiStep LR

```yaml
optim:
  scheduler: multi_step
  scheduler_params:
    milestones: [30, 60, 90] # Decay epochs
    gamma: 0.1 # Decay factor
```

| Parameter    | Type          | Description                               |
| ------------ | ------------- | ----------------------------------------- |
| `milestones` | Integer Array | List of epochs where learning rate decays |
| `gamma`      | Float         | Decay factor                              |

- **Effect**: Adjust learning rate at specified epochs
- **Applicable For**: When optimal decay epochs are known

##### ReduceLROnPlateau

```yaml
optim:
  scheduler: plateau
  scheduler_params:
    mode: min
    factor: 0.8
    patience: 10
    min_lr: 1.0e-7
```

| Parameter   | Type    | Range   | Default | Description                                  |
| ----------- | ------- | ------- | ------- | -------------------------------------------- |
| `mode`      | String  | min/max | min     | Optimization direction (min=lower is better) |
| `factor`    | Float   | 0-1     | 0.1     | Decay factor                                 |
| `patience`  | Integer | ≥ 1     | 10      | Epochs to wait before decaying               |
| `threshold` | Float   | ≥ 0     | 1e-4    | Minimum threshold for improvement            |
| `min_lr`    | Float   | ≥ 0     | 1e-6    | Minimum learning rate lower bound            |

- **Effect**: When monitored metric shows no improvement for `patience` epochs, multiply learning rate by `factor`
- **Applicable For**: Adaptive learning rate adjustment based on validation metrics (Recommended)
- **Relation**: Works best with `runner.early_stop.target`

##### Lambda LR

```yaml
optim:
  scheduler: lambda
  scheduler_params:
    lr_lambda: "lambda epoch: 0.95 ** epoch"
```

| Parameter   | Type   | Description                                                              |
| ----------- | ------ | ------------------------------------------------------------------------ |
| `lr_lambda` | String | Python lambda function; parameter is epoch or step; returns decay factor |

- **Effect**: Learning rate multiplied by function return value
- **Example**: `lambda epoch: 0.95 ** epoch` means multiply by 0.95 each epoch

### warmup_params (Warmup Parameters)

Warmup gradually increases learning rate at the beginning of training to stabilize training.

```yaml
optim:
  warmup_params:
    warmup_steps: 1000 # Absolute steps
    warmup_factor: 0.01 # Initial LR factor
```

or

```yaml
optim:
  warmup_params:
    warmup_epochs: 5 # Converted to steps
    warmup_factor: 0.1
```

| Parameter       | Type    | Range | Default | Description                                         |
| --------------- | ------- | ----- | ------- | --------------------------------------------------- |
| `warmup_steps`  | Integer | ≥ 0   | 0       | Warmup steps (higher priority than `warmup_epochs`) |
| `warmup_epochs` | Integer | ≥ 0   | 0       | Warmup epochs (only used if `warmup_steps` ≤ 0)     |
| `warmup_factor` | Float   | 0-1   | 0.1     | Initial learning rate factor (relative to base lr)  |

- **Effect**:
  - Step 0: lr = base_lr × warmup_factor
  - Step warmup_steps: lr = base_lr
  - Linear interpolation between
- **Relation**:
  - `warmup_steps > 0`: Use directly, ignore `warmup_epochs`
  - `warmup_steps ≤ 0 && warmup_epochs > 0`: Convert to steps (requires batches_per_epoch)
  - Both ≤ 0: Warmup disabled

- **Recommendation**:
  - Small models: warmup_steps = 1000
  - Large models/large datasets: warmup_steps = 5000-10000

### ema_params (Exponential Moving Average Parameters)

EMA maintains a moving average of model parameters, usually with better generalization.

```yaml
optim:
  ema_params:
    ema: true
    ema_decay: 0.99
```

| Parameter   | Type    | Range | Default | Description                                         |
| ----------- | ------- | ----- | ------- | --------------------------------------------------- |
| `ema`       | Boolean | -     | false   | Enable EMA                                          |
| `ema_decay` | Float   | 0-1   | 0.99    | EMA decay coefficient (closer to 1 = slower change) |

- **Effect**:
  - During training: Update both main and EMA model
  - During evaluation: Use EMA model for evaluation (optional)
  - Final saving: Save EMA model as best model

- **Recommended Values**:
  - 0.999: More stable EMA
  - 0.99: Balanced EMA
  - 0.9: Quick response

- **When to Use**:
  - Large datasets: Recommended
  - Dynamic learning rate scheduler: Recommended
  - Small datasets: Optional

---

## runner - Training Runner Configuration

Training runner configuration controls the main training loop, evaluation frequency, checkpoint saving, and early stopping.

### Run Mode: epoch vs step

`qConfig` supports two training modes:

#### Mode 1: Epoch Mode (Recommended)

```yaml
runner:
  run_mode: epoch
  max_epochs: 1000
  eval_interval: 1 # Evaluate every 1 epoch
  early_stop:
    target: val_metric
    patience: 50
    mode: min
```

- **Characteristics**: Process complete dataset once per epoch
- **When to Use**: Most scenarios, especially for moderately-sized datasets
- **Key Parameters**: `max_epochs`, `eval_interval` (counted in epochs)

#### Mode 2: Step Mode

```yaml
runner:
  run_mode: step
  max_steps: 100000 # Maximum 100k steps
  eval_interval: 5000 # Evaluate every 5000 steps
  early_stop:
    target: val_metric
    patience: 5
    mode: min
```

- **Characteristics**: Update every N samples (not limited to complete epochs)
- **When to Use**: Very large datasets, online learning, precise step control
- **Key Parameters**: `max_steps`, `eval_interval` (counted in steps)
- **Note**: In this mode, if `max_epochs` is also specified, it will also become a training stopping condition (training stops when either condition is met)

### Fields Summary

| Field           | Required | Type    | Range      | Default | Description                   |
| --------------- | -------- | ------- | ---------- | ------- | ----------------------------- |
| `run_mode`      | ✅        | String  | epoch/step | epoch   | Training mode                 |
| `max_epochs`    | ⚠️        | Integer | ≥ 1        | None    | Maximum epochs                |
| `max_steps`     | ⚠️        | Integer | ≥ 1        | null    | Maximum steps                 |
| `eval_interval` | ❌        | Integer | ≥ 1        | 1       | Evaluation interval           |
| `save_interval` | ❌        | Integer | ≥ 1        | null    | Regular save interval (steps) |
| `clip_grad`     | ❌        | Float   | ≥ 0.1      | null    | Gradient clipping threshold   |
| `early_stop`    | ✅        | Object  | -          | -       | Early stopping configuration  |

### max_epochs / max_steps (Training Boundaries)

```yaml
# Epoch mode:
runner:
  run_mode: epoch
  max_epochs: 100
```

```yaml
# Step mode:
runner:
  run_mode: step
  max_steps: 100000
```

- **max_epochs**:
  - Type: Integer, ≥ 1
  - Description: Maximum number of training epochs
  - Required for epoch mode; optional for step mode

- **max_steps**:
  - Type: Integer or null
  - Description: Maximum number of training steps
  - Recommended for step mode; optional for epoch mode (dual limit)

- **Logic**:
  - `run_mode='epoch'`: Controlled primarily by `max_epochs`
  - `run_mode='step'`: Controlled primarily by `max_steps` (if specified)
  - Both specified: Stop when either condition is reached

### eval_interval (Evaluation Interval)

```yaml
runner:
  eval_interval: 5
```

- **Type**: Integer, ≥ 1
- **Default**: 1
- **Description**: Evaluation frequency. The unit of this interval is automatically determined by `run_mode`:
  - If `run_mode='epoch'`: Specifies the number of **epochs** between evaluations.
  - If `run_mode='step'`: Specifies the number of **steps** between evaluations.
- **Effect**: Triggers validation evaluation, best model checking, early stopping check
- **Recommendation**:
  - Small datasets: 1-5
  - Large datasets: 10-50

### save_interval (Save Interval)

```yaml
runner:
  save_interval: 1000
```

- **Type**: Integer or null
- **Default**: null (Follows evaluation frequency)
- **Description**: Regular checkpoint saving interval (always in steps, unaffected by `run_mode`)
- **Effect**:
  - null: Automatically saves a regular checkpoint every time an evaluation is triggered based on `eval_interval` (whether by epoch or step).
  - Positive integer: Save checkpoint every N steps

- **Note**:
  - Separate from best model checkpoint
  - Protects against interruption during long training

### clip_grad (Gradient Clipping)

```yaml
runner:
  clip_grad: 1.0
```

- **Type**: Float or null
- **Default**: null (Disabled)
- **Range**: ≥ 0.1
- **Description**: Maximum allowed gradient norm
- **Effect**:
  - null: No gradient clipping
  - Positive number: Limit gradient norm using torch.nn.utils.clip*grad_norm*

- **When to Use**:
  - RNN/LSTM: Strongly recommended (prevent gradient explosion)
  - Transformer: Recommended (clip_grad=1.0 usually works)
  - CNN: Optional

### early_stop (Early Stopping Configuration)

```yaml
runner:
  early_stop:
    target: val_metric # Monitored metric
    patience: 50 # Wait epochs
    mode: min # Optimization direction
    min_delta: 1.0e-6 # Improvement threshold
```

| Parameter     | Type    | Range   | Default    | Description                                         |
| ------------- | ------- | ------- | ---------- | --------------------------------------------------- |
| `target`      | String  | -       | val_metric | Monitored metric name (e.g., `val_loss`, `val_mae`) |
| `patience`    | Integer | ≥ 1     | -          | Epochs to wait before stopping                      |
| `mode`        | String  | min/max | min        | min=lower is better, max=higher is better           |
| `min_delta`   | Float   | ≥ 0     | 0.0        | Minimum change to qualify as improvement            |
| `lower_bound` | Float   | -       | null       | Stop if metric falls below this value               |
| `upper_bound` | Float   | -       | null       | Stop if metric rises above this value               |

- **Effect**:
  - Stop training when `target` metric shows no improvement for `patience` epochs
  - Improvement defined as: `old_value - new_value >= min_delta` (mode='min') or `new_value - old_value >= min_delta` (mode='max')

- **Common Configurations**:
  - Regression: `mode: min, target: val_loss` or `target: val_mae`
  - Classification: `mode: max, target: val_accuracy`
  - Ranking: `mode: min, target: val_ndcg`

- **Relation**: Works best with `ReduceLROnPlateau` scheduler

---

## task - Task Configuration

Task configuration defines the dataset, target variable, and data loading parameters.

### Basic Structure

```yaml
task:
  dataset: maceoff # Dataset identifier
  target: energy # Target variable (optional)
  dataloader:
    batch_size: 32
    eval_batch_size: 64
    num_workers: 4
    pin_memory: true
```

### Field Descriptions

| Field        | Type   | Description                                         |
| ------------ | ------ | --------------------------------------------------- |
| `dataset`    | String | Dataset identifier (e.g., `maceoff`, `qm9`, `md17`) |
| `target`     | String | Target variable label (dataset-dependent)           |
| `dataloader` | Object | Data loading configuration                          |

**Note**: The `task` field can theoretically be customized and extended, except for `dataset` and `dataloader` which are reserved fields (required by the framework). Other fields are determined by the specific implementation.

### dataloader Sub-fields

| Field             | Type    | Range | Default | Description                        |
| ----------------- | ------- | ----- | ------- | ---------------------------------- |
| `batch_size`      | Integer | ≥ 1   | -       | Training batch size (Required)     |
| `eval_batch_size` | Integer | ≥ 1   | -       | Evaluation batch size (Optional)   |
| `num_workers`     | Integer | ≥ 0   | 0       | Number of data loading workers     |
| `pin_memory`      | Boolean | -     | true    | Pin memory for faster GPU transfer |

- **batch_size**:
  - Required
  - Larger batch size: More memory but smaller gradient noise
  - Smaller batch size: Less memory but larger gradient noise

- **eval_batch_size**:
  - Usually larger than `batch_size` (no gradient updates needed)
  - Defaults to `batch_size` if not specified

- **num_workers**:
  - 0: Main process loads data (simple but potentially slow)
  - > 0: Multiprocess data loading (recommended, requires compatible data loading)
  - Recommendation: 4-8 (depending on CPU cores)

- **pin_memory**:
  - GPU training: true (faster CPU-GPU transfer)
  - CPU training: false (save memory)

---

## Complete Examples

### Example 1: Basic Epoch Mode Training (Recommended for Beginners)

```yaml
# configs/basic_training.yaml

seed: 42
log_dir: ./logs/basic_experiment
print_freq: 10

task:
  dataset: qm9
  target: homo
  dataloader:
    batch_size: 32
    eval_batch_size: 64
    num_workers: 4
    pin_memory: true

optim:
  loss: mse

  optimizer: adamw
  optimizer_params:
    lr: 1.0e-3
    weight_decay: 1.0e-4

  scheduler: cosine
  scheduler_params:
    T_max: 100
    eta_min: 1.0e-6

  warmup_params:
    warmup_steps: 1000
    warmup_factor: 0.01

runner:
  run_mode: epoch
  max_epochs: 100
  eval_interval: 1
  clip_grad: 1.0
  early_stop:
    target: val_loss
    patience: 20
    mode: min
    min_delta: 1.0e-6
```

### Example 2: Step Mode + EMA

```yaml
# configs/step_mode_with_ema.yaml

seed: 42
log_dir: ./logs/ema_experiment
print_freq: 50

task:
  dataset: maceoff
  target: energy
  dataloader:
    batch_size: 64
    eval_batch_size: 128
    num_workers: 8
    pin_memory: true

optim:
  loss: mse

  optimizer: adamw
  optimizer_params:
    lr: 2.0e-3
    weight_decay: 1.0e-4
    betas: [0.9, 0.999]

  scheduler: plateau
  scheduler_params:
    mode: min
    factor: 0.8
    patience: 5
    min_lr: 1.0e-7

  warmup_params:
    warmup_steps: 5000
    warmup_factor: 0.01

  ema_params:
    ema: true
    ema_decay: 0.999

runner:
  run_mode: step
  max_steps: 100000
  eval_interval: 5000
  save_interval: 10000
  clip_grad: 10.0
  early_stop:
    target: val_loss
    patience: 3
    mode: min
```

### Example 3: MultiStep Scheduler + Gradient Clipping

```yaml
# configs/multistep_scheduler.yaml

seed: 42
log_dir: ./logs/multistep_experiment
print_freq: 20

task:
  dataset: md17
  dataloader:
    batch_size: 32
    eval_batch_size: 128
    num_workers: 4
    pin_memory: true

optim:
  loss: mae

  optimizer: sgd
  optimizer_params:
    lr: 1.0e-2
    momentum: 0.9
    weight_decay: 1.0e-4

  scheduler: multi_step
  scheduler_params:
    milestones: [30, 60, 90]
    gamma: 0.1

  warmup_params:
    warmup_epochs: 3
    warmup_factor: 0.1

runner:
  run_mode: epoch
  max_epochs: 100
  eval_interval: 2
  clip_grad: 5.0
  early_stop:
    target: val_loss
    patience: 15
    mode: min
```

### Example 4: Resume Training from Checkpoint

```yaml
# configs/resume_training.yaml

seed: 42
log_dir: ./logs/resumed_experiment
print_freq: 10

# Resume from checkpoint (restores all states)
ckp_file: ./logs/previous_experiment/checkpoint_epoch_50.pt

task:
  dataset: qm9
  target: homo
  dataloader:
    batch_size: 32
    eval_batch_size: 64
    num_workers: 4
    pin_memory: true

optim:
  loss: mse
  optimizer: adamw
  optimizer_params:
    lr: 1.0e-3
    weight_decay: 1.0e-4

  scheduler: cosine
  scheduler_params:
    T_max: 100
    eta_min: 1.0e-6

  warmup_params:
    warmup_steps: 1000
    warmup_factor: 0.01

runner:
  run_mode: epoch
  max_epochs: 150
  eval_interval: 1
  clip_grad: 1.0
  early_stop:
    target: val_loss
    patience: 20
    mode: min
```

### Example 5: Transfer Learning (Fine-tuning Pretrained Model)

```yaml
# configs/fine_tuning.yaml

seed: 42
log_dir: ./logs/finetuning_experiment
print_freq: 10

# Load model weights only, reinitialize optimizer and scheduler
init_file: ./pretrained_models/qm9_pretrained.pt

task:
  dataset: md17
  dataloader:
    batch_size: 16
    eval_batch_size: 32
    num_workers: 4
    pin_memory: true

optim:
  loss: mse
  optimizer: adamw
  optimizer_params:
    lr: 1.0e-4 # Smaller learning rate for fine-tuning
    weight_decay: 1.0e-4

  scheduler: cosine
  scheduler_params:
    T_max: 50
    eta_min: 1.0e-7

  warmup_params:
    warmup_steps: 500
    warmup_factor: 0.1

runner:
  run_mode: epoch
  max_epochs: 50
  eval_interval: 1
  clip_grad: 1.0
  early_stop:
    target: val_loss
    patience: 10
    mode: min
```

---

## Configuration Field Interactions

Certain fields in the configuration file have logical relationships with each other. Understanding these interactions is crucial for proper configuration.

### 1. Relationship Between run_mode and Training Control Parameters

#### Impact of run_mode on max_epochs / max_steps

| Scenario                    | run_mode | max_epochs    | max_steps     | Behavior                    |
| --------------------------- | -------- | ------------- | ------------- | --------------------------- |
| Standard epoch mode         | epoch    | Specified     | Not specified | Stop at max_epochs          |
| Standard step mode          | step     | Not specified | Specified     | Stop at max_steps           |
| Dual condition in step mode | step     | Specified     | Specified     | Stop when either is reached |
| Misconfiguration            | epoch    | Not specified | Specified     | Warning: max_steps ignored  |

**Special Behaviors**:

- **max_epochs in step mode**: When `max_epochs` is also specified in step mode, it becomes an **additional stopping condition**. Training stops when either `max_steps` **or** `max_epochs` is reached
- **max_steps in epoch mode**: Specifying `max_steps` in epoch mode is **ignored** by the framework; only `max_epochs` is effective

#### Impact of run_mode on eval_interval Meaning

```yaml
# Epoch mode: Evaluate every 5 epochs
runner:
  run_mode: epoch
  eval_interval: 5

# Step mode: Evaluate every 5000 steps
runner:
  run_mode: step
  eval_interval: 5000
```

**Important**: The unit of `eval_interval` automatically switches based on `run_mode`:

- `run_mode=epoch`: Counted in epochs
- `run_mode=step`: Counted in steps

#### Impact of run_mode on save_interval Meaning

```yaml
# Epoch mode: Follows eval trigger
runner:
  run_mode: epoch
  eval_interval: 5 
  save_interval: null # Will save every 5 epochs

# Step mode: Specific interval overrides eval follow behavior
runner:
  run_mode: step
  eval_interval: 2000
  save_interval: 10000 # Will strictly save every 10000 steps
```

**Key Interactions**:

- If `save_interval` is `null` (default), it automatically matches the evaluation frequency. The actual intervals will adapt to the `run_mode`:
  - `run_mode=epoch`: Saves a regular checkpoint along with validation every `eval_interval` epochs.
  - `run_mode=step`: Saves a regular checkpoint along with validation every `eval_interval` steps.
- If `save_interval` is set to a specific positive Integer, it is explicitly executed in **steps**, entirely independent of the `run_mode` and the evaluation trigger.

### 2. Priority Between warmup_steps and warmup_epochs in warmup_params

```yaml
# Priority: warmup_steps > warmup_epochs
optim:
  warmup_params:
    warmup_steps: 1000 # This takes precedence
    warmup_epochs: 5 # Ignored
```

**Rules**:

- If `warmup_steps > 0`: Use `warmup_steps` directly, ignore `warmup_epochs`
- If `warmup_steps ≤ 0` and `warmup_epochs > 0`: Automatically convert `warmup_epochs` to steps
  - Conversion formula: `warmup_steps = warmup_epochs × batches_per_epoch`
  - Requires framework to obtain `batches_per_epoch` from the data loader
- If both ≤ 0: Warmup is disabled

**Recommendation**: Always prefer `warmup_steps` to avoid depending on `batches_per_epoch` conversion

### 3. Relationship Between early_stop and checkpoint

```yaml
runner:
  early_stop:
    target: val_loss # Monitored metric
    mode: min # Optimization direction
    patience: 20

# Corresponding checkpoint configuration (automatically derived by framework)
# checkpoint:
#   target: val_loss
#   mode: min
```

**Key Points**:

- `early_stop.target` and `early_stop.mode` also control the criterion for best model evaluation
- Framework automatically transfers `early_stop` configuration to `checkpoint` configuration
- Best model is saved based on the same metric and optimization direction
- If `early_stop.target` shows no improvement, early stopping is triggered and new best models are no longer saved

### 4. Adaptation Between scheduler and run_mode

#### Cosine Annealing (Recommended)

```yaml
# Epoch mode
optim:
  scheduler: cosine
  scheduler_params:
    T_max: 100              # Set to max_epochs
    eta_min: 1.0e-6

# Step mode
optim:
  scheduler: cosine
  scheduler_params:
    T_max: 100000           # Set to max_steps
    eta_min: 1.0e-6
```

**Correlation**: T_max should be adjusted based on `run_mode`:

- Epoch mode: T_max ≈ max_epochs
- Step mode: T_max ≈ max_steps

#### Plateau Scheduler (Adaptive)

```yaml
optim:
  scheduler: plateau
  scheduler_params:
    target: val_loss # Monitored metric (should match early_stop.target)
    patience: 5
    factor: 0.8
```

**Key Relation**:

- `scheduler.target` and `early_stop.target` should **remain consistent**
- Plateau scheduler automatically adjusts learning rate based on metric
- If monitored metric shows no long-term improvement, learning rate gradually decreases

### 5. Mutual Exclusivity Between ckp_file and init_file

```yaml
# Cannot specify both; behavior when both are specified:
ckp_file: ./checkpoint_epoch_50.pt # Restores complete training state
init_file: ./pretrained_weights.pt # Loads only model weights
```

**Exclusivity Rules**:

- If both are specified, framework **prioritizes ckp_file**, ignoring init_file
- ckp_file: Restores model + optimizer + scheduler + training state (epoch/step/best_metric)
- init_file: Loads only model weights; optimizer and scheduler are reinitialized

**Use Cases**:

- Resuming interrupted training: Use `ckp_file`
- Transfer learning/fine-tuning: Use `init_file`

### 6. Relationship Between clip_grad and Model Type

| Model Type  | Recommended Value | Description                            |
| ----------- | ----------------- | -------------------------------------- |
| RNN/LSTM    | 1.0-5.0           | Prevents gradient explosion (required) |
| Transformer | 1.0               | Balances stability and performance     |
| GNN         | 0.5-1.0           | Adjust based on graph structure        |
| CNN         | null              | Usually not needed                     |

**Boundary Behaviors**:

- `clip_grad: null` or unspecified: Disable gradient clipping
- `clip_grad: 0`: Ineffective (use null to disable)
- `clip_grad < 0.1`: Over-clipping may cause training failure

### 7. Relationship Between batch_size and Learning Rate

**Rule of Thumb**:

- Increase batch_size → Learning rate should increase proportionally
- Decrease batch_size → Learning rate should decrease proportionally
- Typical adjustment: $\text{new\_lr} = \text{old\_lr} \times \frac{\text{new\_batch\_size}}{\text{old\_batch\_size}}$

**Example**:

```yaml
# Scenario 1: Small batch size
dataloader:
  batch_size: 16
optim:
  optimizer_params:
    lr: 1.0e-4              # Relatively small learning rate

# Scenario 2: Large batch size
dataloader:
  batch_size: 256
optim:
  optimizer_params:
    lr: 1.0e-3              # Proportionally higher learning rate
```

### 8. Relationship Between distributed and pin_memory

```yaml
# GPU distributed training
pin_memory: true            # Enable memory pinning for faster data transfer

# CPU training or memory-constrained
pin_memory: false           # Disable to save memory
```

**Key Points**:

- Framework automatically adapts behavior based on `device` and `distributed`
- Users do not need to manually specify `distributed` (framework auto-detects)
- `pin_memory` should be manually adjusted based on hardware configuration

### 9. Boundary Behaviors in Special Scenarios

#### Scenario A: step mode + Both max_epochs and max_steps Specified

```yaml
runner:
  run_mode: step
  max_epochs: 100 # Upper limit
  max_steps: 50000 # Upper limit
  eval_interval: 5000 # Evaluate every 5000 steps
```

**Behavior**:

- Training stops when `global_step >= max_steps` **or** `epoch >= max_epochs`
- If data is limited, entire epochs may complete in step mode before max_steps, triggering epoch increment
- Framework logs the actual stopping reason (max_steps vs max_epochs)

#### Scenario B: warmup_steps > max_steps

```yaml
runner:
  run_mode: step
  max_steps: 100
optim:
  warmup_params:
    warmup_steps: 5000 # Warmup period longer than total training steps
```

**Warning**:

- Framework outputs warning but does not auto-correct
- Warmup may persist throughout training, learning rate never reaches target
- Users should manually adjust `warmup_steps < max_steps`

#### Scenario C: eval_interval > max_steps

```yaml
runner:
  run_mode: step
  max_steps: 1000
  eval_interval: 5000 # Evaluation interval exceeds total steps
```

**Behavior**:

- May not trigger any evaluation (except possibly at training end)
- Framework logs warning
- Recommend setting `eval_interval < max_steps`

---

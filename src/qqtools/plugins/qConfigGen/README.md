

# Format Requirement

```yaml
BASE: # reserved KEYWORD. Indicate yaml files to be overrided. can be either str, list, dict.

log_dir: # reserved KEYWORD. all of config files, ckp files, log files will be saved here.
config_file: # reserved KEYWORD.
ckp_file: # reserved KEYWORD. Control the recovery flow with checkpoint file.
init_file: # reserved KEYWORD

optim: # required
    optimizer: # required
    optimizer_params": # required
    scheduler":  # optional
    scheduler_params": # optional
    warmup":   # optional
    
runner: # required
    epochs:  # required
```


# optimizer

| **Name**  | **supported params**                         |
| :-------- | :------------------------------------------- |
| `adamw`   | `lr`, `betas`, `eps`, `weight_decay`         |
| `sgd`     | `lr`, `momentum`, `weight_decay`, `nesterov` |
| `rmsprop` | `lr`, `alpha`, `weight_decay`, `momentum`    |
| `adagrad` | `lr`, `lr_decay`, `weight_decay`             |
| `adam`    | `lr`, `betas`, `eps`, `weight_decay`         |



# scheduler

| **Name**         | **Supported Parameters**                                                  |
| :--------------- | :------------------------------------------------------------------------ |
| `step`           | `step_size`, `gamma`                                                      |
| `multistep`      | `milestones`, `gamma`                                                     |
| `exponential`    | `gamma`                                                                   |
| `cosine`         | `T_max`, `eta_min`                                                        |
| `cosine_restart` | `T_0`, `T_mult`, `eta_min`                                                |
| `lambda`         | `lr_lambda` (need be supplied as a function)                              |
| `plateau`        | `mode`, `factor`, `patience`, `threshold`, `min_lr`                       |
| `cyclic`         | `base_lr`, `max_lr`, `step_size_up`, `step_size_down`, `mode`             |
| `onecycle`       | `max_lr`, `total_steps`, `pct_start`, `anneal_strategy`, `cycle_momentum` |
| `linear`         | `start_factor`, `end_factor`, `total_iters`                               |

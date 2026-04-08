# qpipeline Log Format

本文档说明 `src/qqtools/plugins/qpipeline` 当前训练日志中三类核心输出的含义、触发时机和时序差异。

目标范围：

- terminal event 单行终态日志
- `[Eval Summary]` 多行块
- `--- Epoch N Results ---` 及其 `[train]` / `[val]` / `[test]` 行


## 日志类别总览

训练过程中常见的结构化日志可分为三类：

1. Terminal event

```text
Training finished: reason=max_steps
Training finished: reason=max_epochs
Training finished: reason=early_stop
Training stopped: reason=user_interrupt
Training failed: reason=oom
Training failed: reason=exception
```

2. Evaluation summary

```text
[Eval Summary] Epoch: E, Step: S
  - Primary Target: ...
  - Best Tracker: ...
  - Validation:
    - [Main] ...
  - Testing:
    - [Main] ...

[Eval Summary Table] Epoch: E | Step: S | Target: ...
...
```

3. Epoch result summary

```text
--- Epoch N Results ---
[train] loss: ... metric: ...
[val] metric: ... source=current_eval|latest_eval_reuse|missing
[test] metric: ... source=current_eval|latest_eval_reuse|missing
```

这三类日志的触发机制不同：

- terminal event 跟整个 run 的最终退出边界绑定
- `[Eval Summary]` 跟 evaluation 触发点绑定
- `--- Epoch N Results ---` 跟 epoch 完成事件绑定


## Terminal Event

terminal event 表示一次训练 run 的最终结束状态。

当前实现保证：

- 每个 run 恰好输出一条 terminal event
- terminal event 由 `train_runner` 外层边界统一发出
- 外部系统可以只看这条日志判断本次 run 的最终状态

允许的文本形式如下：

```text
Training finished: reason=max_steps
Training finished: reason=max_epochs
Training finished: reason=early_stop
Training stopped: reason=user_interrupt
Training failed: reason=oom
Training failed: reason=exception
```

其中：

- `finished` 表示训练以完成语义退出
- `stopped` 表示用户中断
- `failed` 表示异常失败

`reason` 当前使用以下规范值：

- `max_steps`
- `max_epochs`
- `early_stop`
- `user_interrupt`
- `oom`
- `exception`

如果调用方直接使用 `train_runner` 的返回值，`terminal_event` 负载当前包含：

- `status`
- `reason`
- `text`
- `epoch`
- `step`
- 失败时附带 `exception_type`

返回值中的 `early_stopped` 采用更严格的语义：

- `early_stopped == (terminal_event.reason == "early_stop")`
- `user_interrupt` 不再计入 `early_stopped=True`
- `max_steps` / `max_epochs` / `oom` / `exception` 都对应 `early_stopped=False`

默认行为中：

- terminal event 不持久化原始异常 message
- terminal event 不在该日志路径中持久化 traceback
- 若需要排查失败类型，应优先使用 `reason` 和 `exception_type`


## 触发规则

### `[Eval Summary]`

`[Eval Summary]` 在一次 evaluation 完成后打印。

evaluation 是否触发由以下条件决定：

- `run_mode=epoch` 时，按 epoch 检查 `eval_interval`
- `run_mode=step` 时，按真实 `optimizer.step()` 完成次数检查 `eval_interval`

换句话说：

- `epoch` 模式中，evaluation 只会发生在 epoch 末尾
- `step` 模式中，evaluation 可以发生在 epoch 中间


### `--- Epoch N Results ---`

`--- Epoch N Results ---` 只在一个 epoch 真正结束后打印一次。

它不跟 `eval_interval` 直接绑定，也不会因为 step 模式下命中了 evaluation 周期而额外打印。

换句话说：

- 不管 `run_mode=epoch` 还是 `run_mode=step`
- 只要 epoch 没结束，就不会打印 `--- Epoch N Results ---`

当前 phase2 语义下，epoch-result 中的 `[val]` / `[test]` 行会显式标注指标来源：

- `source=current_eval`
  表示本次 epoch-end 边界刚好触发了 evaluation，当前指标是这一边界新鲜计算得到的
- `source=latest_eval_reuse`
  表示当前 epoch-end 没有触发新的 evaluation，日志复用了最近一次已存在的 evaluation 指标
- `source=missing`
  表示到这个 epoch-end 为止仍没有可用的对应指标值，此时日志会打印 `metric: n/a`

这条来源语义是按“epoch-end 边界是否新鲜求值”来定义的，而不是按“本 epoch 内是否曾经做过 evaluation”来定义。
因此在 `run_mode=step` 下，即使同一个 epoch 中间已经出现过 `[Eval Summary]`，只要 epoch 结束时没有再次 evaluation，epoch-result 仍会标记为 `latest_eval_reuse`。


## Run Mode 语义

### `run_mode=epoch`

`eval_interval` 的单位是 epoch。

示例：

- `eval_interval=1` 表示每个 epoch 结束后都做一次 evaluation
- `eval_interval=2` 表示每 2 个 epoch 结束后做一次 evaluation

同一个 epoch 末尾的常见顺序为：

1. 最后一个 batch 跑完
2. 若命中 `eval_interval`，执行 evaluation
3. 打印 `[Eval Summary]`
4. 触发 epoch end
5. 打印 `--- Epoch N Results ---`

因此在 `epoch` 模式下，如果某个 epoch 命中了 evaluation，通常会先看到 `[Eval Summary]`，再看到 `--- Epoch N Results ---`。


### `run_mode=step`

`eval_interval` 的单位是全局 step，其中 step 指真实完成的 `optimizer.step()` 次数。

示例：

- `eval_interval=100` 表示每完成 100 次真实优化步就做一次 evaluation

这意味着：

- 如果一个 epoch 很长，可能在同一个 epoch 内打印多次 `[Eval Summary]`
- `--- Epoch N Results ---` 仍然只会在该 epoch 全部 batch 跑完后打印一次

因此在 `step` 模式下，日志常见形态是：

1. epoch 内部多次出现 `[Eval Summary]`
2. epoch 最后才出现一次 `--- Epoch N Results ---`


## 典型时序示例

### 示例 1: `run_mode=epoch, eval_interval=2`

第 0 个 epoch：

```text
Starting training (mode=epoch, eval_interval=2, ...)
... train batch logs ...
--- Epoch 0 Results ---
[train] loss: ...
```

第 1 个 epoch：

```text
... train batch logs ...
[Eval Summary] Epoch: 1, Step: S
  - Primary Target: val_metric: ...
  - Best Tracker: ...
  - Validation:
    - [Main] ...
  - Testing:
    - [Main] ...

[Eval Summary Table] Epoch: 1 | Step: S | Target: val_metric(...)
...

--- Epoch 1 Results ---
[train] loss: ...
[val] metric: ... source=current_eval
[test] metric: ... source=current_eval
```


### 示例 2: `run_mode=step, eval_interval=100`

如果某个 epoch 内累计完成了 300 个真实优化步，则可能看到：

```text
Starting training (mode=step, eval_interval=100, ...)
... train batch logs ...

[Eval Summary] Epoch: 0, Step: 100
...

... train batch logs ...

[Eval Summary] Epoch: 0, Step: 200
...

... train batch logs ...

[Eval Summary] Epoch: 0, Step: 300
...

--- Epoch 0 Results ---
[train] loss: ...
[val] metric: ... source=latest_eval_reuse
[test] metric: ... source=latest_eval_reuse
```

这个例子说明：

- `[Eval Summary]` 的打印频率可能高于 epoch 结果
- `--- Epoch N Results ---` 本身不是 step 周期日志


## `[val]` / `[test]` 行的含义

`--- Epoch N Results ---` 中的：

```text
[val] metric: ... source=current_eval|latest_eval_reuse|missing
[test] metric: ... source=current_eval|latest_eval_reuse|missing
```

来自当前运行状态中的最近一次 evaluation 结果。

这意味着：

- 它们表示当前 state 中缓存的最新 `val_metric` / `test_metric`
- 这两行在 epoch 结束时会固定打印；如果当前没有可用指标值，则会输出 `metric: n/a source=missing`
- `source=current_eval` 表示 epoch-end 边界刚好触发了新的 evaluation
- `source=latest_eval_reuse` 表示 epoch-end 没有新 eval，而是复用了最近一次已有指标
- 不一定意味着 epoch 结束时刚刚又做了一次 evaluation

特别是在 `run_mode=step` 下，如果本 epoch 的最后一次 evaluation 发生在中段，那么 epoch 结束时打印的 `[val]` / `[test]` 很可能只是复用了那次中段 evaluation 的结果。


## Truth Source

对于“是否发生了一次新的 evaluation”这个问题，本文档采用以下真相来源规则。

1. `Eval Summary` 是 evaluation event 的主数据源。
2. `--- Epoch N Results ---` 只是 epoch 收尾摘要，不是 evaluation event 记录。
3. `Epoch Results` 中出现 `[val]` / `[test]`，并不构成“刚刚发生了一次新 eval”的证据。
4. 是否发生了新的 eval，应以是否出现了新的 `Eval Summary` 为准。

换句话说：

- 如果您想知道“这次训练在什么时刻真正执行了一次 evaluation”，请看 `Eval Summary`
- 如果您想知道“这个 epoch 收尾时有哪些已知指标”，可以看 `Epoch Results`
- 但不要根据 `Epoch Results` 中的 `[val]` / `[test]` 反推“一定发生了新的 eval”

这条规则在 `run_mode=step` 下尤其重要，因为同一个 epoch 中可能早已发生过一次或多次 evaluation，而 epoch end 只是把最近一次已知的 eval 指标带到收尾摘要里。


## 机器解析建议

如果外部系统需要从日志中还原训练过程，建议按事件类型分别解析，而不要混用不同日志块的语义。

推荐规则：

- 识别 run terminal state：解析 terminal event
- 识别 evaluation event：解析 `Eval Summary`
- 识别 epoch completion：解析 `--- Epoch N Results ---`
- 判断是否发生了新的 eval：以是否出现新的 `Eval Summary` 为准
- 不要根据 `Epoch Results` 中的 `[val]` / `[test]` 判断“epoch end 又做了一次 eval”

对于接入方来说，可以把这三类日志理解为：

- terminal event 负责记录 run 最终退出语义
- `Eval Summary` 负责记录 evaluation 事实和 evaluation 结果
- `Epoch Results` 负责记录 epoch 收尾时的训练摘要

它们可以引用同一批指标值，但不代表同一种事件。


## 边界行为

### `step` 模式下中途命中 `max_steps`

如果 `run_mode=step` 且训练在 epoch 中途达到 `max_steps`，训练循环可能直接停止，此时：

- 若该 step 命中了 `eval_interval`，可能已经打印了 `[Eval Summary]`
- 但因为当前 epoch 并未自然结束，可能不会打印 `--- Epoch N Results ---`

示例：

```text
[Eval Summary] Epoch: 0, Step: 1000
Training loop stopping at epoch 0, step 1000.
Reached max_steps=1000
Training finished: reason=max_steps
```

此时没有 epoch result summary 是符合当前实现的。

其中：

- `Training loop stopping...` 和 `Reached max_steps=...` 属于调试级别的边界邻接日志
- run 的最终终态仍应以 `Training finished: reason=max_steps` 为准


## 编号说明

当前实现中的 `epoch` 字段采用内部状态编号。

日志中出现的：

```text
Epoch 0
Epoch 1
```

表示 runner 当前状态中的 epoch 计数值，属于实现语义。该编号是在 `on_epoch_end` 打印后再递增，因此日志显示的是当前收尾时刻的内部 epoch 编号。


## 总结

- `[Eval Summary]` 跟 evaluation 触发点走
- terminal event 是 run 级别唯一终态事件
- `epoch` 模式下 evaluation 按 epoch 触发
- `step` 模式下 evaluation 按真实 optimizer step 触发
- `Eval Summary` 才是 evaluation event 的主数据源
- `--- Epoch N Results ---` 永远只在 epoch 真正结束时打印
- `Epoch Results` 只是 epoch 收尾摘要，不是 eval 事件记录
- `step` 模式下可以出现多个 `[Eval Summary]` 对应一个 `--- Epoch N Results ---`
- epoch result 中的 `[val]` / `[test]` 会固定打印，缺值时使用 `metric: n/a source=missing`
- epoch result 中的 `[val]` / `[test]` 是最近一次 evaluation 的缓存值或缺值占位，不必然表示 epoch 末尾重新 evaluation
- 是否发生了新的 eval，以有没有新的 `Eval Summary` 为准

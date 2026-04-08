# qexp v2 Product Spec

状态：草稿

更新时间：2026-04-08

## 目标

本文档只回答四个问题：

1. `qexp` 到底是什么
2. 用户平时怎么用
3. 哪些能力属于 `qexp`
4. 哪些能力不属于 `qexp`

实现细节、共享目录布局、锁、并发一致性不在本文档讨论，见配套实现文档：

- [qexp_v2_runtime_spec_20260408.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_v2_runtime_spec_20260408.md)

## 产品定位

`qexp` 是一个轻量实验提交队列。

它负责：

- 提交 task
- 批量提交 task
- 记录调度事件
- 管理轻量本机 agent
- 展示队列与机器状态
- 取消与重试

它不负责：

- 深度学习训练日志管理
- 训练指标日志格式
- artifact 管理
- 实验产物归档
- 远程跨机器投递
- 长期常驻后台服务

## 核心边界

### 1. 默认只服务当前机器

`qexp submit` 默认只把 task 提交到当前机器。

不承诺：

- 从机器 A 执行命令，把 task 远程投给机器 B
- 从机器 A 远程唤起机器 B 的 agent

多台机器共享的是：

- 元数据视图
- 队列状态
- 批次信息

不是远程控制权。

### 2. agent 默认按需运行

默认模式：

- `agent_mode = on_demand`
- 有任务时可自动拉起
- 空闲 10 分钟自动退出

只有用户显式启用时，agent 才允许常驻：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a --agent-mode persistent
```

或：

```bash
qexp agent start --persistent
```

### 3. 共享模式下 machine 名必须显式指定

由于同一 Docker 镜像可启动多个容器，`hostname` 不可靠。

因此共享模式下：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a
```

中的 `--machine` 必填。

不以 hostname 作为 machine 身份主键。

### 4. 日志边界只到调度事件

`qexp` 只记录：

- task 是否提交成功
- task 何时开始
- task 何时结束
- task 是否发生异常
- 异常类别是什么

训练任务具体输出什么日志、怎么记录、怎么存，完全由训练框架自身负责。

## 术语

为避免混用，v2 只使用三层术语：

- `machine`：一台执行 task 的机器或容器实例
- `task`：一次具体提交与执行对象
- `batch`：一组一起提交、一起观察、一起重试的 task 集合

不再把 `job` 设为一等对象。

当前代码与新术语的映射关系：

- 当前 `qExpTask.task_id` = 新术语中的 `task_id`
- 当前 `qExpTask.name` = 新术语中的 `task name`
- 当前 CLI `--job-id` 实际只是“显式 task_id”

## 用户工作流

### 1. 首次接入

共享模式下，每台机器第一次都要执行：

```bash
qexp init --shared-root /path/to/shared/qexp --machine gpu2a
```

如果希望该机器 agent 常驻：

```bash
qexp init --shared-root /path/to/shared/qexp --machine gpu2a --agent-mode persistent
```

这里的 `/path/to/shared/qexp` 只是示例，真实目录可以是别的样子。

### 2. 单任务提交

这是最高频入口，必须足够轻：

```bash
qexp submit -- python train.py --config configs/a.yaml
```

或者：

```bash
qexp submit --task-id qm9_seed_1 --name "qm9 seed 1" -- python train.py --config configs/a.yaml
```

红线：

- 交一个 task 不允许强制写 yaml
- 不允许强制先构造 batch

### 3. 批量提交

只有提交一组 task 时，才使用 manifest：

```bash
qexp batch-submit --file runs.yaml
```

manifest 只服务于 batch，不反向污染单任务入口。

默认约束：

- manifest 中的 task 必须属于当前机器
- 若声明其他 machine，默认报错

### 4. 观察

日常观察入口应保持扁平：

```bash
qexp list
qexp inspect task_xxx
qexp top
qexp top --all
qexp batches
qexp batch batch_xxx
qexp machines
```

### 5. 取消与重试

```bash
qexp cancel task_xxx
qexp retry task_xxx
qexp batch-retry-failed batch_xxx
qexp batch-retry-cancelled batch_xxx
```

要求：

- 不要求用户重写原 submit 命令
- 不要求用户回头找原 yaml
- 不要求用户先 clean 再 submit
- `retry` 必须生成新的 task，并保留 lineage

## 最终 CLI

### 高频命令

- `qexp init`
- `qexp submit`
- `qexp batch-submit`
- `qexp list`
- `qexp inspect`
- `qexp top`
- `qexp cancel`
- `qexp retry`
- `qexp batches`
- `qexp batch`
- `qexp machines`

### 低频命令

- `qexp agent start`
- `qexp agent stop`
- `qexp agent status`
- `qexp doctor`

## `qexp --help` 草案

```text
usage: qexp <command> [options]

Lightweight experiment submission queue for the current machine.

By default, qexp submits tasks to the current machine only.
It uses a shared control root for metadata and an on-demand local agent.
The local agent exits automatically after 10 minutes of idleness unless
persistent mode was explicitly enabled.

Common commands:
  init            Initialize qexp on the current machine
  submit          Submit one task on the current machine
  batch-submit    Submit a batch manifest on the current machine
  list            List tasks
  inspect         Show one task
  top             Show live queue / machine overview
  cancel          Cancel one task
  retry           Retry one task
  batches         List batches
  batch           Show one batch
  machines        List visible machines

Low-frequency commands:
  agent           Manage the local agent
  doctor          Diagnose or repair qexp metadata

Examples:
  qexp init --shared-root /path/to/shared/qexp --machine gpu2a
  qexp submit -- python train.py --config configs/a.yaml
  qexp batch-submit --file runs.yaml
  qexp list
  qexp inspect task_xxx
  qexp top
  qexp cancel task_xxx
  qexp retry task_xxx
```

## 应砍掉的命令

以下命令不应出现在主帮助第一页：

- `qexp task ...`
- `qexp clone`
- `qexp batch-export`
- `qexp machine inspect`
- `qexp agent wake`

如果未来确实保留，也只能是低频扩展。

## 验收标准

- 用户在共享模式下必须通过 `--machine` 显式声明当前机器身份。
- 单任务提交只需要 `qexp submit -- ...`。
- 批量提交才需要 `qexp batch-submit --file ...`。
- 默认不支持跨机器远程投递。
- 默认不支持长期常驻后台进程。
- `qexp` 不管理训练日志体系，只记录调度事件。

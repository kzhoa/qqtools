# qexp Manual

状态：草稿

更新时间：2026-04-08

## 目标

本文档面向直接使用 `qexp` 的用户，只回答这些问题：

1. `qexp` 现在默认怎么工作
2. 第一次接入时要做什么
3. 单任务、批量任务怎么提交
4. 平时怎么观察、取消、重试
5. 遇到常见问题时先检查什么

本文档只保留用户真正需要理解的行为、命令和排障方式，不要求读者再跳转到开发规格文档。

## 当前版本行为

从 `v1.2.7` 开始，`qexp` 默认使用 v2 shared-root 引擎。

这意味着：

- 默认入口是多机器共享视图的 v2
- 需要显式提供 machine 身份
- 任务元数据保存在 shared root
- agent 默认按需启动，不要求长期常驻

如果您要临时回退到旧版单机引擎，可以显式使用：

```bash
qexp --v1 ...
```

或者：

```bash
QEXP_VERSION=1 qexp ...
```

## 基本概念

- `shared root`：多机器共享的 `qexp` 控制目录，保存任务、批次、索引和事件
- `machine`：当前执行任务的一台机器或一个容器实例
- `runtime root`：当前机器本地运行目录，保存 agent pid、日志等本地状态
- `task`：一次实际提交与执行对象
- `batch`：一组一起提交、一起观察、一起重试的 task

task 的常见状态流转是：

`queued -> dispatching -> starting -> running -> succeeded / failed / cancelled`

其中：

- `queued`：已入队，等待某台机器的 agent 拉取
- `dispatching`：agent 已接手，正在分配执行资源
- `starting`：执行包装器已准备启动用户命令
- `running`：用户命令已开始运行
- `succeeded` / `failed` / `cancelled`：终态

## 环境要求

### v2 基本要求

- Python 环境中已安装 `qqtools`
- 有一个所有相关机器都可访问的 shared root
- 当前机器有一个可写的本地 runtime 目录

### agent / 调度要求

如果您希望任务提交后自动被调度执行，还需要：

- Linux
- `tmux`
- `libtmux`
- 可用 GPU 检测后端，例如 `pynvml` 或 `nvidia-smi`

如果这些依赖不齐，任务仍可能被成功写入队列，但不会自动开始执行。

## 快速开始

### 第一步：初始化一台机器

每台机器第一次接入 shared queue 时都需要初始化一次：

```bash
qexp init --shared-root /mnt/share/my_qexp --machine gpu-a
```

如果您希望 agent 在该机器上常驻：

```bash
qexp init --shared-root /mnt/share/my_qexp --machine gpu-a --agent-mode persistent
```

`--agent-mode on_demand` 是默认模式，agent 会在需要调度时被拉起，并在空闲一段时间后自动退出；`persistent` 适合长期跑任务的固定机器，agent 会持续常驻，减少反复拉起的等待。

初始化后，后续命令可以继续显式传参：

```bash
qexp --shared-root /mnt/share/my_qexp --machine gpu-a list
```

也可以使用环境变量：

```bash
export QEXP_SHARED_ROOT=/mnt/share/my_qexp
export QEXP_MACHINE=gpu-a
qexp list
```

如果本机 runtime 目录需要自定义，也可以显式指定：

```bash
qexp init \
  --shared-root /mnt/share/my_qexp \
  --machine gpu-a \
  --runtime-root /data/local/qexp-runtime
```

### 第二步：提交第一个任务

```bash
qexp --shared-root /mnt/share/my_qexp --machine gpu-a \
  submit -- python train.py --config configs/a.yaml
```

### 第三步：观察执行情况

```bash
qexp --shared-root /mnt/share/my_qexp --machine gpu-a list
qexp --shared-root /mnt/share/my_qexp --machine gpu-a top
qexp --shared-root /mnt/share/my_qexp --machine gpu-a logs <task_id>
```

## 常用命令

### 单任务提交

最常见的提交方式：

```bash
qexp submit -- python train.py --config configs/a.yaml
```

显式指定 task id 和展示名：

```bash
qexp submit \
  --task-id qm9_seed_1 \
  --name "qm9 seed 1" \
  --gpus 1 \
  -- python train.py --config configs/a.yaml
```

说明：

- `--` 后面的内容会原样作为用户命令
- `--gpus` 表示请求的 GPU 数量
- `--task-id` 不传时会自动生成

### 批量提交

使用 manifest 批量提交：

```bash
qexp batch-submit --file runs.yaml
```

一个最小 manifest 示例：

```yaml
batch:
  name: sweep-a

tasks:
  - command: ["python", "train.py", "--config", "configs/a.yaml"]
  - command: ["python", "train.py", "--config", "configs/b.yaml"]
```

一个更完整的 manifest 示例：

```yaml
batch:
  name: sweep-a
  policy:
    allow_retry_failed: true
    allow_retry_cancelled: false

defaults:
  requested_gpus: 1

tasks:
  - task_id: qm9_seed_1
    name: qm9 seed 1
    command: ["python", "train.py", "--config", "configs/a.yaml", "--seed", "1"]

  - task_id: qm9_seed_2
    name: qm9 seed 2
    requested_gpus: 2
    command: ["python", "train.py", "--config", "configs/a.yaml", "--seed", "2"]
```

字段约定：

- `batch.name`：批次展示名
- `batch.policy.allow_retry_failed`：是否允许执行 `qexp batch-retry-failed`
- `batch.policy.allow_retry_cancelled`：是否允许执行 `qexp batch-retry-cancelled`
- `defaults.requested_gpus`：本批次任务默认 GPU 数量
- `tasks[].task_id`：可选；不写时自动生成
- `tasks[].name`：可选；用于展示
- `tasks[].requested_gpus`：单任务覆盖默认 GPU 数量
- `tasks[].command`：必填；实际执行命令

### 查看任务列表

```bash
qexp list
qexp list --phase queued
qexp list --batch <batch_id>
```

### 查看单个任务详情

```bash
qexp inspect <task_id>
```

### 查看总览

仅看当前 machine：

```bash
qexp top
```

查看所有 machine：

```bash
qexp top --all
```

### 查看机器列表

```bash
qexp machines
```

### 查看批次

```bash
qexp batches
qexp batch <batch_id>
```

### 查看日志

查看某个 task 的已落盘日志：

```bash
qexp logs <task_id>
```

持续跟随日志：

```bash
qexp logs -f <task_id>
```

### 取消与重试

取消任务：

```bash
qexp cancel <task_id>
```

重试一个已结束任务：

```bash
qexp retry <task_id>
```

批量重试失败任务：

```bash
qexp batch-retry-failed <batch_id>
```

批量重试已取消任务：

```bash
qexp batch-retry-cancelled <batch_id>
```

### 清理旧记录

先 dry run：

```bash
qexp clean --dry-run
```

默认只清理 7 天前的成功任务：

```bash
qexp clean
```

显式指定时间阈值：

```bash
qexp clean --older-than-seconds 259200
```

连失败和取消一起清理：

```bash
qexp clean --include-failed
```

## Agent 管理

### 手动启动 agent

前台运行：

```bash
qexp agent start
```

常驻模式：

```bash
qexp agent start --persistent
```

后台启动：

```bash
qexp agent start --background
```

### 停止 agent

```bash
qexp agent stop
```

### 查看 agent 状态

```bash
qexp agent status
```

## Doctor 命令

用于排查和修复共享状态问题：

```bash
qexp doctor verify
qexp doctor rebuild-index
qexp doctor repair-orphans
qexp doctor cleanup-locks
```

各子命令含义：

- `qexp doctor verify`：只读完整性检查，不会修改文件；主要检查任务文件是否可读、文件名和 task_id 是否一致、revision 是否有效
- `qexp doctor rebuild-index`：重建索引，适合索引与任务实际状态不一致时使用
- `qexp doctor repair-orphans`：把长时间失去机器心跳、但仍停留在活动态的任务修复为 `orphaned`
- `qexp doctor cleanup-locks`：清理残留过久的锁文件，适合异常退出后锁未释放的场景

推荐顺序：

1. 先跑 `qexp doctor verify`
2. 如果怀疑索引不一致，再跑 `qexp doctor rebuild-index`
3. 如果任务疑似卡死或丢失归属，再考虑 `repair-orphans` 和 `cleanup-locks`

## 典型工作流

### 单机日常使用

```bash
export QEXP_SHARED_ROOT=/mnt/share/my_qexp
export QEXP_MACHINE=gpu-a

qexp submit -- python train.py --config configs/a.yaml
qexp list
qexp top
qexp logs <task_id>
```

### 多机共享同一队列

机器 A：

```bash
qexp init --shared-root /mnt/share/my_qexp --machine gpu-a
```

机器 B：

```bash
qexp init --shared-root /mnt/share/my_qexp --machine gpu-b
```

之后两台机器都连接到同一个 `shared root`，但各自使用自己的 `machine` 身份和本地 runtime 目录。

任务最终在哪台机器执行，取决于哪台机器上的 agent 实际把该任务从队列中拉起并开始调度。

## 常见问题

### 1. `submit` 成功了，但任务一直停在 `queued`

优先检查：

- 当前机器的 agent 是否真的在运行：`qexp agent status`
- 是否安装了 `tmux`
- 是否安装了 `libtmux`
- 当前机器是否能检测到 GPU

如果自动拉起失败，可以先手动执行：

```bash
qexp agent start
```

### 2. `qexp` 提示缺少 `--shared-root` 或 `--machine`

说明当前命令处于 v2 模式，但没有拿到必要的 machine 上下文。

解决方式：

- 显式传 `--shared-root` 和 `--machine`
- 或设置 `QEXP_SHARED_ROOT` / `QEXP_MACHINE`

### 3. 我只想继续使用旧版单机 qexp

可以显式切回 v1：

```bash
qexp --v1 status
```

或：

```bash
QEXP_VERSION=1 qexp status
```

### 4. `logs` 或任务执行时报本地路径不可写

说明当前 machine 的 runtime root 不可写。

优先处理方式：

- 给当前用户提供一个可写目录
- 或显式指定 `--runtime-root`
- 或设置 `QEXP_RUNTIME_ROOT`

例如：

```bash
export QEXP_RUNTIME_ROOT=/data/local/qexp-runtime
```

## 参数约定

### 全局参数

v2 命令支持这些全局参数：

- `--shared-root <path>`：共享控制目录；多台机器看到的是同一份任务与索引
- `--machine <name>`：当前机器身份；必须在 shared root 内唯一
- `--runtime-root <path>`：当前机器的本地运行目录；保存 agent pid、日志等本地状态；不传时使用默认本地目录

这些参数一般写在子命令前：

```bash
qexp --shared-root /mnt/share/my_qexp --machine gpu-a list
```

### `submit` 参数

- `--task-id <id>`：可选；自定义任务 ID；不传则自动生成
- `--name <text>`：可选；任务展示名
- `--gpus <int>`：请求的 GPU 数量；默认是 `1`
- `-- <your command...>`：分隔符之后的内容会原样作为用户命令执行

### `top` 参数

- `--all`：显示所有机器的概览；不传时只看当前 machine

### `logs` 参数

- `-f` / `--follow`：持续跟随日志输出；不传时只打印当前已落盘内容

### `clean` 参数

- `--dry-run`：只展示将被清理的记录，不实际删除
- `--include-failed`：把失败和取消的终态任务也纳入清理范围
- `--older-than-seconds <int>`：只清理早于该秒数阈值的任务；默认 `604800`，即 7 天

## 边界说明

`qexp` 负责轻量实验排队与调度，不负责：

- 训练指标体系设计
- artifact 托管
- 模型版本管理
- 数据版本管理
- 远程跨机器命令代理

如果您需要的是完整实验平台，`qexp` 不是那类系统；它更接近一个轻量共享队列和调度外壳。

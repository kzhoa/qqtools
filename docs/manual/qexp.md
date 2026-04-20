# qexp Manual

状态：草稿

更新时间：2026-04-17

## 目标

本文档面向直接使用 `qexp` 的用户，只回答这些问题：

1. `qexp` 现在默认怎么工作
2. 第一次接入时要做什么
3. 单任务、批量任务怎么提交
4. 平时怎么观察、取消、重试
5. 遇到常见问题时先检查什么

本文档只保留用户真正需要理解的行为、命令和排障方式，不要求读者再跳转到开发规格文档。

## 当前版本行为

当前 `qexp` 使用 shared-root 引擎。

这意味着：

- 默认入口是多机器共享视图
- 需要显式提供 machine 身份
- 任务元数据保存在 shared root
- agent 默认按需启动，不要求长期常驻

## 基本概念

- `shared root`：多机器共享的 `qexp` 控制目录，保存任务、批次、索引和事件
- `machine`：当前执行任务的一台机器或一个容器实例
- `runtime root`：当前机器本地运行目录，保存 agent pid、日志等本地状态
- `task`：一次实际提交与执行对象
- `group`：项目内长期归组键，用来把一组相关 task 归在同一个工作上下文里
- `batch`：一组一起提交、一起观察、一起重试的 task
- `name`：单个 task 的展示名
- `task_id`：单个 task 的唯一标识

这些概念的关系要分清：

- `group` 解决“这些 task 长期属于哪一组工作上下文”
- `batch` 解决“这些 task 是否是这一次一起提交的一批”
- `name` 解决“这个 task 给人看的名字是什么”
- `task_id` 解决“系统内部唯一标识是什么”
- 一个 `group` 内可以有多个 `batch`
- `group` 不等于 `batch`
- `name` 不等于 `group`
- `task_id` 不等于 `group`

这几个字段的职责必须固定：

- `group` 只承担归组职责，并直接映射到 tmux session
- `name` 只承担展示职责，不作为归组主键
- `task_id` 只承担唯一标识职责，用于 `inspect`、`logs`、`cancel`、`retry`

task 的常见状态流转是：

`queued -> dispatching -> starting -> running -> succeeded / failed / cancelled`

其中：

- `queued`：已入队，等待某台机器的 agent 拉取
- `dispatching`：agent 已接手，正在分配执行资源
- `starting`：执行包装器已准备启动用户命令
- `running`：用户命令已开始运行
- `succeeded` / `failed` / `cancelled`：终态

## 环境要求

### 基本要求

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
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

`shared root` 不是任意目录，而是项目控制目录；正式形态固定为 `project_root/.qexp`。

正确形态：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

不推荐：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp/exp1/shared --machine gpu-a
qexp init --shared-root /mnt/share/myproject/.qexp/exp2/shared --machine gpu-a
```

原因：

- `shared root` 代表一整套项目级控制平面
- 同一项目内的 task、batch、machine、索引和事件应共享同一套真相
- 如果按实验计划拆多个 `shared root`，同项目内的资源视图、观察面和队列状态会被打碎
- 当前运行时会拒绝不符合 `project_root/.qexp` 约束的根路径

如果您希望 agent 在该机器上常驻：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a --agent-mode persistent
```

`--agent-mode on_demand` 是默认模式，agent 会在需要调度时被拉起，并在空闲一段时间后自动退出；`persistent` 适合长期跑任务的固定机器，agent 会持续常驻，减少反复拉起的等待。

`qexp init` 成功后会自动保存当前 CLI context（`shared_root`、`machine`，以及显式传入的 `runtime_root`）。

因此初始化后，后续命令通常不需要再重复填写这些参数：

```bash
qexp list
qexp top
qexp logs <task_id>
```

如果您想显式切换或覆盖默认 context，可以使用：

```bash
qexp use --shared-root /mnt/share/myproject/.qexp --machine gpu-a
qexp use --show
```

当然，后续命令也仍然可以继续显式传参：

```bash
qexp --shared-root /mnt/share/myproject/.qexp --machine gpu-a list
```

也可以使用环境变量；优先级是：

1. 命令行参数
2. 环境变量
3. 已保存的 context

例如：

```bash
export QEXP_SHARED_ROOT=/mnt/share/myproject/.qexp
export QEXP_MACHINE=gpu-a
qexp list
```

如果本机 runtime 目录需要自定义，也可以显式指定：

```bash
qexp init \
  --shared-root /mnt/share/myproject/.qexp \
  --machine gpu-a \
  --runtime-root /data/local/qexp-runtime
```

`init` 也可以安全重复执行。重复执行时，它会确保目录结构存在、刷新当前 machine 注册信息，并更新本地保存的 context。

### 第二步：提交第一个任务

```bash
qexp submit -- python train.py --config configs/a.yaml
```

### 第三步：观察执行情况

```bash
qexp list
qexp top
qexp logs <task_id>
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

如果您想把多个相关 task 放进同一个工作上下文里，推荐显式声明 `group`：

```bash
qexp submit \
  --group contract_n_4and6 \
  --name n4 \
  --gpus 1 \
  -- python train.py --n 4
```

推荐理解方式：

- `group` 不是业务里的“实验计划”正式术语
- 但您可以把脑中的一个实验计划映射成一个稳定的 `group`
- 同一 `group` 下的 task 会被视为同一组长期相关任务

`group` 的约束如下：

- 允许字符：字母、数字、`.`、`_`、`-`
- 区分大小写
- 禁止保留名：`experiments`
- 禁止保留名：`qqtools_internal`
- 禁止以 `.` 或 `-` 开头
- 长度上限为 `64`
- 校验通过后直接映射为 tmux session 名，不做二次 sanitize

推荐命名习惯：

- 一个实验计划对应一个稳定 `group`
- 一次具体 run 用 `name`
- 若需要一次性批量提交，再用 `batch`

例如：

- `group=contract_n_4and6`
- `name=n4`
- `name=n6`

### 批量提交

使用 manifest 批量提交：

```bash
qexp batch-submit --file runs.yaml
```

一个最小 manifest 示例：

```yaml
batch:
  name: sweep-a
  group: contract_n_4and6

tasks:
  - command: ["python", "train.py", "--config", "configs/a.yaml"]
  - command: ["python", "train.py", "--config", "configs/b.yaml"]
```

一个更完整的 manifest 示例：

```yaml
batch:
  name: sweep-a
  group: contract_n_4and6
  policy:
    allow_retry_failed: true
    allow_retry_cancelled: false

defaults:
  requested_gpus: 1

tasks:
  - task_id: qm9_seed_1
    name: qm9 seed 1
    group: contract_n_4and6
    command: ["python", "train.py", "--config", "configs/a.yaml", "--seed", "1"]

  - task_id: qm9_seed_2
    name: qm9 seed 2
    group: regrouped_debug
    requested_gpus: 2
    command: ["python", "train.py", "--config", "configs/a.yaml", "--seed", "2"]
```

字段约定：

- `batch.name`：批次展示名
- `batch.group`：该批任务默认归属的长期 `group`
- `batch.policy.allow_retry_failed`：是否允许执行 `qexp batch-retry-failed`
- `batch.policy.allow_retry_cancelled`：是否允许执行 `qexp batch-retry-cancelled`
- `defaults.requested_gpus`：本批次任务默认 GPU 数量
- `tasks[].task_id`：可选；不写时自动生成
- `tasks[].name`：可选；用于展示
- `tasks[].group`：可选；单任务归组；优先级高于 `batch.group`
- `tasks[].requested_gpus`：单任务覆盖默认 GPU 数量
- `tasks[].command`：必填；实际执行命令

归组优先级：

1. `tasks[].group` 优先级最高
2. 若 `tasks[].group` 缺失，则继承 `batch.group`
3. 若二者都缺失，则该 task 的 `group = null`

建议这样理解：

- `batch` 表示“这次一起交的一批”
- `group` 表示“长期属于哪组工作上下文”

如果您今天交一批、明天再补交一批，但它们都属于同一个实验计划，推荐：

- 使用不同 `batch`
- 继续复用同一个 `group`

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

显式把重试后的新 task 放进另一个 group：

```bash
qexp retry <task_id> --group regrouped_debug
```

规则：

- `qexp retry <task_id>` 默认继承原 task 的 `group`
- `qexp retry <task_id> --group <group>` 使用显式覆盖值，不继承原 group
- retry 只创建新的 task，不改写原 task 的 `group`

原位替换一个已结束且允许 `resubmit` 的 task：

```bash
qexp resubmit <task_id> -- python train.py --config configs/a.yaml
```

显式覆盖新的展示名和 group：

```bash
qexp resubmit <task_id> --name rerun_a --group regrouped_debug -- python train.py --config configs/a.yaml
```

规则：

- `resubmit` 只允许 `failed` / `cancelled`
- `resubmit` 不允许 batch 成员 task
- `resubmit` 会删除旧 task 正式记录，再用同一个 `task_id` 创建新的首次提交真相
- 若命令中途失败并留下未完成替换事务，执行 `qexp doctor repair` 继续收敛
- 当正式 task 已被删、替换仍未完成时，`qexp inspect <task_id>` 会显示未完成 `resubmit` 的操作态提示

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

精确清理单个终态 task：

```bash
qexp clean --task-id <task_id>
```

先预览单 task clean 的结果：

```bash
qexp clean --task-id <task_id> --dry-run
```

说明：

- `qexp clean` 默认是批量清理模式
- `qexp clean --task-id <task_id>` 是单 task 精确清理模式
- 单 task 模式下不允许再组合 `--older-than-seconds` 或 `--include-failed`
- 单 task clean 只允许用于终态 task：`succeeded` / `failed` / `cancelled`
- 若该 task 属于某个 batch，clean 会同步修正 batch 成员列表与摘要
- runtime log 删除是 best-effort：若 log 可定位且当前进程可访问，则一并删除；否则 CLI 会明确提示未删除

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

`qexp agent status` 是 agent 生命周期的权威解释入口。

状态语义：

- `active`：当前 machine 仍有 `queued` / `dispatching` / `starting` 责任
- `draining`：当前 machine 已无 launch backlog，但仍有 `running` 责任需要收敛
- `idle`：当前 machine 已无任何 active responsibility，正在等待 idle timeout 自动退出
- `stopped` / `stale` / `failed`：分别表示未运行、心跳失效、异常退出

`on_demand` agent 仍会自动退出，但退出条件不是“最近没 launch 新任务”，而是：

1. 当前 machine 不再承担 `queued` / `dispatching` / `starting` / `running` 责任
2. agent 进入 `idle`
3. `idle` 持续达到 `idle_timeout`
4. 然后才自动退出

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

与 clean 的关系：

- 若 clean 中途失败且您怀疑索引视图不一致，优先跑 `qexp doctor rebuild-index`
- **假设/未验证**：若后续实现提供更通用的 `qexp doctor repair`，则 single-task clean 的失败恢复应优先收敛到该入口
- `cleanup-locks` 和 `repair-orphans` 不负责补做 single-task clean 的 batch 修正或索引重建

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

### 项目内按实验计划管理任务

假设您的项目目录是：

```text
/mnt/share/myproject/
```

并且您现在有一个实验计划，要比较 `n=4` 和 `n=6`。

推荐做法不是为这个实验计划单独建一个 `shared root`，而是：

1. 整个项目共用一个 project 级 `shared root`
2. 用一个稳定的 `group` 表示这次实验计划
3. 每个具体 run 用单独 task 提交

推荐初始化：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

推荐提交方式：

```bash
qexp submit --group contract_n_4and6 --name n4 --gpus 1 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 --gpus 1 -- python train.py --n 6
```

如果是一批一起交：

```yaml
batch:
  name: contract-compare-round1
  group: contract_n_4and6

tasks:
  - name: n4
    group: contract_n_4and6
    command: ["python", "train.py", "--n", "4"]
  - name: n6
    group: regrouped_debug
    command: ["python", "train.py", "--n", "6"]
```

然后：

```bash
qexp batch-submit --file runs.yaml
```

后续如果您要补交同一实验计划的新任务，推荐继续复用：

- 同一个 `group`

而不是：

- 新建一个新的 `shared root`

推荐管理方式：

- 用 `group` 代表实验计划
- 用 `name` 区分具体 run
- 用 `batch` 组织一次批量提交
- 用 `task_id` 作为单 task 的唯一主键
- 用同一个 project 级 `shared root` 保持项目内资源与观察一致性

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
- 当前机器是否卡在 `active` / `draining`，以及 `workset` 是否仍显示 backlog
- 是否安装了 `tmux`
- 是否安装了 `libtmux`
- 当前机器是否能检测到 GPU

如果自动拉起失败，可以先手动执行：

```bash
qexp agent start
```

### 2. `qexp` 提示缺少 `--shared-root` 或 `--machine`

说明当前命令没有拿到必要的 machine 上下文。

解决方式：

- 显式传 `--shared-root` 和 `--machine`
- 或设置 `QEXP_SHARED_ROOT` / `QEXP_MACHINE`

### 3. 我能不能为每个实验计划单独建一个 `shared root`

不推荐。

推荐：

- 整个项目共用一个 project 级 `shared root`
- 不同实验计划通过 `group` 区分

如果为每个实验计划拆一个新的 `shared root`，会出现这些问题：

- 同一项目内资源视图被拆碎
- 同一项目内的 `top` / `machines` / `list` 只能看到局部事实
- 切换 `qexp use` 后，新的 root 不会自动感知旧 root 中的 task 占用

只有当您明确要隔离成“另一套独立控制平面”时，才应使用另一个 `shared root`

### 4. 我以前用过旧版单机 qexp

旧版单机接口已移除。

当前应改用 project-root `.qexp` 控制目录，并显式提供 `shared_root` 与 `machine` 上下文。

### 5. `logs` 或任务执行时报本地路径不可写

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

当前命令支持这些全局参数：

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
- `--group <text>`：可选；项目内长期归组键；适合映射一个实验计划或一组长期相关任务；直接映射到 tmux session 名
- `--gpus <int>`：请求的 GPU 数量；默认是 `1`
- `-- <your command...>`：分隔符之后的内容会原样作为用户命令执行

### `retry` 参数

- `<task_id>`：必填；要重试的原 task
- `--group <text>`：可选；显式指定新 task 的 group；不传时默认继承原 task 的 group

### `top` 参数

- `--all`：显示所有机器的概览；不传时只看当前 machine

### `logs` 参数

- `-f` / `--follow`：持续跟随日志输出；不传时只打印当前已落盘内容

### `clean` 参数

- `--dry-run`：只展示将被清理的记录，不实际删除
- `--include-failed`：把失败和取消的终态任务也纳入清理范围
- `--older-than-seconds <int>`：只清理早于该秒数阈值的任务；默认 `604800`，即 7 天
- `--task-id <id>`：精确清理单个终态 task；与 `--older-than-seconds`、`--include-failed` 互斥

`clean` 的结果语义：

- batch clean 主要按终态范围和时间阈值筛选删除对象
- single-task clean 会删除该 task 的共享真相，并同步修正相关 batch 真相与索引
- single-task clean 成功后，`qexp inspect <task_id>` 与 `qexp logs <task_id>` 应表现为该 task 已不存在
- runtime log 删除不是跨机器强保证；若日志未删，CLI 必须明确报告

## 边界说明

`qexp` 负责轻量实验排队与调度，不负责：

- 训练指标体系设计
- artifact 托管
- 模型版本管理
- 数据版本管理
- 远程跨机器命令代理

如果您需要的是完整实验平台，`qexp` 不是那类系统；它更接近一个轻量共享队列和调度外壳。

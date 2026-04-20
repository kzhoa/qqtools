# qexp Product Spec

状态：草稿

更新时间：2026-04-17

## 目标

本文档只回答四个问题：

1. `qexp` 到底是什么
2. 用户平时怎么用
3. 哪些能力属于 `qexp`
4. 哪些能力不属于 `qexp`

实现细节、共享目录布局、锁、并发一致性不在本文档讨论，见配套实现文档：

- [qexp_runtime_spec.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md)

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

## 产品心智模型

### 术语

为避免混用，`qexp` 只使用四层术语：

- `machine`：一台执行 task 的机器或容器实例
- `group`：项目内长期归组键，用来把一组相关 task 放进同一个工作上下文
- `task`：一次具体提交与执行对象
- `batch`：一组一起提交、一起观察、一起重试的 task 集合

不再把 `job` 设为一等对象。

当前代码与新术语的映射关系：

- 当前 `qExpTask.task_id` = 新术语中的 `task_id`
- 当前 `qExpTask.name` = 新术语中的 `task name`
- 显式 task 标识统一使用 `--task-id`

术语关系必须明确：

- `group` 解决“这些 task 长期属于哪一组工作上下文”
- `batch` 解决“这些 task 是否是这一次一起提交的一批”
- 一个 `group` 内可以有多个 `batch`
- `batch` 不得替代 `group`

### 核心边界

#### 1. 默认只服务当前机器

`qexp submit` 默认只把 task 提交到当前机器。

不承诺：

- 从机器 A 执行命令，把 task 远程投给机器 B
- 从机器 A 远程唤起机器 B 的 agent

多台机器共享的是：

- 元数据视图
- 队列状态
- 批次信息

不是远程控制权。

#### 2. `shared_root` 是 project 级控制目录

`shared_root` 不应按单个实验计划切分，且正式形态固定为 `project_root/.qexp`。

推荐：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu2a
```

不推荐：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp/exp1/shared --machine gpu2a
qexp init --shared-root /mnt/share/myproject/.qexp/exp2/shared --machine gpu2a
```

原因：

- `shared_root` 代表一整套项目级控制平面
- 同一项目内的 task、batch、machine、索引和事件应共享同一套真相
- 如果按实验计划拆 root，同项目内的资源视图、队列视图和观察面会被打碎
- `qexp` 运行时会拒绝不符合 `project_root/.qexp` 约束的根路径

#### 3. 共享模式下 machine 名必须显式指定

由于同一 Docker 镜像可启动多个容器，`hostname` 不可靠。

因此共享模式下：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a
```

中的 `--machine` 必填。

不以 hostname 作为 machine 身份主键。

#### 4. agent 默认按需运行

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

#### 5. 日志边界只到调度事件

`qexp` 只记录：

- task 是否提交成功
- task 何时开始
- task 何时结束
- task 是否发生异常
- 异常类别是什么

训练任务具体输出什么日志、怎么记录、怎么存，完全由训练框架自身负责。

## 核心命令语义

### 1. `submit`

这是最高频入口，必须足够轻：

```bash
qexp submit -- python train.py --config configs/a.yaml
```

或者：

```bash
qexp submit --task-id qm9_seed_1 --name "qm9 seed 1" -- python train.py --config configs/a.yaml
```

**假设/未验证**：`group` 已进入产品推荐契约，但当前安装版本未必已经支持 `--group`；若未支持，请把下面示例理解为目标用法。

如果用户脑中有一个明确的实验计划，推荐把它映射成一个稳定的 `group`：

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

这里的含义是：

- `group=contract_n_4and6` 表示这两条 task 长期属于同一组工作上下文
- `name=n4` / `name=n6` 表示每次具体 run 的展示名
- 用户脑中的“实验计划”不需要变成单独对象，只需要稳定映射到一个 `group`

红线：

- 交一个 task 不允许强制写 yaml
- 不允许强制先构造 batch
- `submit` 的职责只能是“创建一个新 task”
- 若显式指定的 `task_id` 已存在，`submit` 默认必须报错，不允许静默覆盖旧 task

### 2. `batch-submit`

只有提交一组 task 时，才使用 manifest：

```bash
qexp batch-submit --file runs.yaml
```

manifest 只服务于 batch，不反向污染单任务入口。

**假设/未验证**：`batch.group` 已进入产品推荐契约，但当前安装版本未必已经支持该字段；若未支持，请把下面示例理解为目标 schema。

推荐形态：

```yaml
batch:
  name: contract-compare-round1
  group: contract_n_4and6

tasks:
  - name: n4
    command: ["python", "train.py", "--n", "4"]
  - name: n6
    command: ["python", "train.py", "--n", "6"]
```

这里的关系是：

- `batch.name` 表示“这次一起提交的一批”
- `batch.group` 表示“这批 task 长期属于哪个 group”
- 如果明天再补交一批同一实验计划的任务，推荐复用同一个 `group`，而不是强行复用同一个 `batch`

默认约束：

- manifest 中的 task 必须属于当前机器
- 若声明其他 machine，默认报错

### 3. `retry` / `resubmit` / `clean`

```bash
qexp cancel task_xxx
qexp retry task_xxx
qexp resubmit task_xxx -- python train.py --config configs/a.yaml
qexp batch-retry-failed batch_xxx
qexp batch-retry-cancelled batch_xxx
qexp clean
qexp clean --include-failed
qexp clean --older-than-seconds 259200
qexp clean --task-id task_xxx
qexp clean --task-id task_xxx --dry-run
```

要求：

- 不要求用户重写原 submit 命令
- 不要求用户回头找原 yaml
- 不要求用户先 clean 再 submit
- `retry` 必须生成新的 task，并保留 lineage
- `resubmit` 必须是显式独立入口，不允许由 `submit` 隐式触发

`submit / retry / resubmit` 的职责边界必须固定：

- `submit`：创建一个全新 task
- `retry`：保留旧 task 记录，再创建一个新的 task
- `resubmit`：删除一个指定终态旧 task 的正式记录后，使用同一个 `task_id` 重新提交

三者不得相互偷渡语义：

- 不允许 `submit` 隐式覆盖旧 task 来模拟 `retry`
- 不允许 `submit` 隐式 clean 旧 task 来模拟 `resubmit`
- 不允许 `retry` 复用原 `task_id`
- 不允许 `resubmit` 保留旧 task 记录

三种入口的正式语义：

- `submit`
  - 表示“我要创建一个新 task”
  - 若 `task_id` 已存在，默认报错
  - 不负责保留历史链路，也不负责删除旧历史
- `retry`
  - 表示“我要保留旧记录，再跑一次”
  - 只允许作用于终态 task
  - 必须生成新的 `task_id`
  - 必须写入 `lineage.retry_of = <old_task_id>`
- `resubmit`
  - 表示“我不要这条终态黑历史，但我还要沿用这个 `task_id`”
  - 只允许作用于终态 task
  - 必须先删除旧 task 正式记录，再用相同 `task_id` 创建新 task
  - 新 task 不保留 `lineage.retry_of`

`resubmit` 的边界规则：

- 默认只允许 `failed` / `cancelled`
- 默认不允许对 `queued` / `dispatching` / `starting` / `running` 执行
- 默认不允许对 batch 成员 task 执行
- 若未来要支持对 batch 成员 `resubmit`，必须单独扩展 batch 真相修正规则，不能隐式放开
- `resubmit` 成功后，用户不应再能观察到被替换前那条旧 task 记录
- `resubmit` 可复用 single-task clean 的删除子流程，但不是 `clean` 的别名命令，也不应被实现成简单串联 CLI `clean + submit`

设计取舍：

- `retry` 解决“保留历史再跑一次”
- `resubmit` 解决“抹掉黑历史后原位重提”
- `clean` 仍然保留为通用删除入口，但当用户的真实意图是“删除后立即重提同一个 task_id”时，优先使用 `resubmit`

`clean` 的产品契约：

- `qexp clean` 默认仍按现有批量规则清理旧终态 task
- `qexp clean --task-id <task_id>` 必须精确清理单个终态 task
- single-task clean 只允许 `succeeded` / `failed` / `cancelled`
- single-task clean 与 `--older-than-seconds`、`--include-failed` 互斥
- 若被清理 task 属于某个 batch，batch 视图必须同步反映删除后的正式成员与摘要
- runtime log 删除只承诺 best-effort，不承诺跨机器强一致删除

### 4. 观察命令

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

## 用户工作流

### 1. 首次接入

共享模式下，每台机器第一次都要执行：

```bash
qexp init --shared-root /path/to/myproject/.qexp --machine gpu2a
```

如果希望该机器 agent 常驻：

```bash
qexp init --shared-root /path/to/myproject/.qexp --machine gpu2a --agent-mode persistent
```

这里的目录形态不是任意示例，而是正式契约：

- 一个项目一个 `shared_root`
- 该目录固定命名为 `.qexp`
- 不允许一个实验计划一个 `shared_root`

### 2. 日常使用顺序

推荐的日常顺序是：

1. `qexp init`
2. `qexp submit` 或 `qexp batch-submit`
3. `qexp list` / `qexp inspect` / `qexp top`
4. 按需要执行 `cancel` / `retry` / `resubmit` / `clean`

## CLI Surface

### 高频命令

- `qexp init`
- `qexp submit`
- `qexp batch-submit`
- `qexp list`
- `qexp inspect`
- `qexp top`
- `qexp cancel`
- `qexp retry`
- `qexp resubmit`
- `qexp clean`
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
The local agent exits automatically after 10 minutes of true idleness unless
persistent mode was explicitly enabled.
True idleness means this machine has no remaining `queued`, `dispatching`,
`starting`, or `running` responsibilities. If only `running` tasks remain,
the agent stays alive in `draining` state until they converge.

Common commands:
  init            Initialize qexp on the current machine
  submit          Submit one task on the current machine
  batch-submit    Submit a batch manifest on the current machine
  list            List tasks
  inspect         Show one task
  top             Show live queue / machine overview
  cancel          Cancel one task
  retry           Retry one task
  resubmit        Replace one terminal task in place with the same task_id
  clean           Clean terminal task records
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
  qexp resubmit task_xxx -- python train.py --config configs/a.yaml
  qexp clean --dry-run
  qexp clean --task-id task_xxx
```

## 非目标命令

以下命令不应出现在主帮助第一页：

- `qexp task ...`
- `qexp clone`
- `qexp batch-export`
- `qexp machine inspect`
- `qexp agent wake`

如果未来确实保留，也只能是低频扩展。

## 验收标准

- 用户在共享模式下必须通过 `--machine` 显式声明当前机器身份。
- `shared_root` 的推荐粒度必须是 project 级，而不是实验计划级。
- 单任务提交只需要 `qexp submit -- ...`。
- 用户若有实验计划这类长期工作上下文，推荐通过 `group` 映射，而不是拆新的 `shared_root`。
- 批量提交才需要 `qexp batch-submit --file ...`。
- `group` 与 `batch` 的职责必须区分：`group` 负责长期归组，`batch` 负责一次批量提交。
- 默认不支持跨机器远程投递。
- 默认不支持长期常驻后台进程。
- `qexp` 不管理训练日志体系，只记录调度事件。
- `clean` 必须同时覆盖批量清理与单 task 精确清理，且单 task clean 后 batch 视图不能残留悬挂引用。

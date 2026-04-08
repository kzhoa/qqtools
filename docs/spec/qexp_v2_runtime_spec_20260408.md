# qexp v2 Runtime Spec

状态：草稿

更新时间：2026-04-08

## 目标

本文档只讨论实现层问题：

1. shared root 怎么组织
2. machine 怎么登记
3. 并发写怎么处理
4. agent 如何在 on-demand / persistent 两种模式下运行

产品定位、CLI、工作流不在本文档展开，见：

- [qexp_v2_product_spec_20260408.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_v2_product_spec_20260408.md)

## 系统平面

### 1. shared control root

示例：

```text
/mnt/share/myusername/qexp
```

这只是示例路径，不是强制约定。

职责：

- machine 元数据
- task 元数据
- batch 元数据
- 全局索引
- 锁
- 调度事件记录

### 2. local runtime root

建议：

```text
~/.qqtools/qexp-runtime
```

职责：

- agent pid
- local locks
- heartbeat
- wrapper / runtime 临时状态

## 共享目录布局

```text
<shared-root>/
  global/
    schema/
      version.json
    tasks/
      <task_id>.json
    batches/
      <batch_id>.json
    indexes/
      tasks_by_batch/
      tasks_by_state/
      tasks_by_machine/
    locks/
      submit/
      batch/
      migrate/
    events/
      <date>/
        <event_id>.json
  machines/
    <machine_name>/
      machine.json
      claims/
      events/
      state/
```

原则：

- `global/` 是共享真相
- `machines/<machine_name>/` 是该机器在共享目录中的私有区
- 本机只写自己的 machine 子目录
- 机器之间不直接写对方子目录

## 为什么每个机器要有自己的共享子目录

因为这样可以把高频写操作局部化。

否则所有机器都去争同一层目录，会带来：

- 锁竞争扩大
- 并发写冲突增多
- 故障排查困难

machine 子目录主要承载：

- 本机声明文件
- 本机事件
- 本机状态镜像
- 本机 claim 记录

### 每个 machine 子目录的固定布局

`machines/<machine_name>/` 应直接写死为以下结构：

```text
<shared-root>/machines/<machine_name>/
  machine.json
  state/
    agent.json
    gpu.json
    summary.json
  claims/
    active/
      <task_id>.json
    released/
      <task_id>.json
  events/
    <date>/
      <event_id>.json
```

各路径语义如下。

#### 1. `machine.json`

机器静态声明文件。

承载：

- `machine_name`
- `hostname`
- `shared_root`
- `runtime_root`
- `agent_mode`
- 机器标签

用途：

- 表明这台机器是谁
- 作为共享系统中的 machine 主声明

#### 2. `state/agent.json`

机器 agent 当前状态快照。

承载：

- `agent_state`
- `pid`
- `started_at`
- `last_heartbeat`
- `idle_timeout_seconds`
- `last_exit_reason`

用途：

- 让其他命令看到这台机器当前 agent 是否活着
- 区分 `on_demand` 自动退出和异常退出

#### 3. `state/gpu.json`

机器 GPU 当前状态快照。

承载：

- `visible_gpu_ids`
- `reserved_gpu_ids`
- `task_to_gpu_ids`
- `updated_at`

用途：

- 供 `qexp top --all`
- 供 `qexp machines`
- 供排查“这台机器为什么没拉新任务”

#### 4. `state/summary.json`

机器级聚合摘要。

承载：

- `queued_count`
- `running_count`
- `failed_count`
- `last_task_started_at`
- `last_task_finished_at`

用途：

- 提供轻量 machine 总览
- 避免每次都全量扫 task 文件

#### 5. `claims/active/<task_id>.json`

表示这台机器已成功 claim 某个 task，且该 claim 当前仍生效。

承载：

- `task_id`
- `machine_name`
- `claimed_at`
- `revision_at_claim`

用途：

- 帮助排查 dispatch 归属
- 为 orphan repair 提供额外证据

#### 6. `claims/released/<task_id>.json`

表示这台机器曾经 claim 过该 task，但 claim 已结束。

承载：

- `task_id`
- `released_at`
- `release_reason`

用途：

- 保留最小调度审计线索
- 辅助解释“为什么这台机器曾经处理过它”

#### 7. `events/<date>/<event_id>.json`

机器本地调度事件日志。

承载事件类型例如：

- `agent_started`
- `agent_stopped`
- `task_claimed`
- `task_started`
- `task_finished`
- `task_failed`
- `task_cancelled`
- `task_orphaned`

用途：

- machine 维度调度审计
- 不替代训练日志

### machine 子目录写入规则

为避免机器之间互相踩写，规则直接写死：

- 当前机器只能写自己的 `machines/<machine_name>/...`
- 当前机器不得写其他机器的子目录
- 其他机器状态只通过读取得知，不通过跨目录修改修复

### machine 子目录与 global 的关系

职责分层应明确：

- `global/tasks/*.json` 是 task 真相
- `machines/<machine>/state/*.json` 是该机器的状态镜像
- `claims/` 是调度证据
- `events/` 是 machine 维度事件审计

也就是说：

- `global/` 是 source of truth
- `machines/<machine>/...` 是 machine 私有的共享侧辅助视图

## machine 身份

### 1. 共享模式下 machine 名必填

由于同一 Docker 镜像可起多个容器，hostname 不可靠。

因此：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a
```

中的 `--machine` 必填。

### 2. 不以 hostname 作为主键

`hostname` 可以记录在 `machine.json` 里作为辅助信息，但不能作为 machine 主身份。

machine 主键必须是用户显式提供的 `machine_name`。

### 3. machine 对象

`machines/<machine_name>/machine.json` 至少包含：

```yaml
machine:
  machine_name: str
  hostname: str | null
  shared_root: str
  runtime_root: str
  agent_mode: enum[on_demand, persistent]
  agent_state: enum[stopped, starting, active, idle, stale, failed]
  last_heartbeat: str | null
  gpu_inventory:
    count: int
    visible_gpu_ids: list[int]
```

## task 对象

```yaml
task:
  task_id: str
  name: str | null
  batch_id: str | null
  machine_name: str
  attempt: int
  status:
    phase: enum[queued, dispatching, starting, running, succeeded, failed, cancelled, blocked, orphaned]
    reason: str | null
    category: str | null
  runtime:
    command: list[str]
    requested_gpus: int
    assigned_gpus: list[int]
  timestamps:
    created_at: str
    queued_at: str | null
    started_at: str | null
    finished_at: str | null
  result:
    exit_code: int | null
    terminal_reason: str | null
  lineage:
    retry_of: str | null
```

### `global/tasks/<task_id>.json` 最终字段格式

`global/tasks/<task_id>.json` 是 task 的真相文件。

建议固定为：

```yaml
meta:
  revision: int
  created_at: str
  updated_at: str
  updated_by_machine: str

task:
  task_id: str
  name: str | null
  batch_id: str | null
  machine_name: str
  attempt: int
  spec:
    command: list[str]
    requested_gpus: int
  status:
    phase: enum[queued, dispatching, starting, running, succeeded, failed, cancelled, blocked, orphaned]
    reason: str | null
    category: str | null
  runtime:
    assigned_gpus: list[int]
    process_group_id: int | null
    wrapper_pid: int | null
  timestamps:
    created_at: str
    queued_at: str | null
    started_at: str | null
    finished_at: str | null
  result:
    exit_code: int | null
    terminal_reason: str | null
  lineage:
    retry_of: str | null
```

字段约束：

- `meta.revision` 每次成功写入都必须递增
- `task.task_id` 必须与文件名一致
- `task.machine_name` 一旦写入后不得随意迁移
- `task.spec.command` 是 task 真正执行命令
- `task.status.phase` 是 task 当前唯一正式状态
- `task.lineage.retry_of` 仅在 retry 产生的新 task 上存在

设计原则：

- 这是唯一正式 task 真相
- indexes 和 machine 子目录都只是派生视图

### task 真相层最小写入原则

只有以下动作应改写 `global/tasks/<task_id>.json`：

- submit 创建 task
- agent dispatch task
- agent 启动 task
- task 正常结束
- task 失败
- task 取消
- orphan repair
- retry 生成新 task

任何机器级缓存、GPU 快照、辅助状态都不得反向覆盖 task 真相文件。

## batch 对象

```yaml
batch:
  batch_id: str
  name: str | null
  task_ids: list[str]
  source_manifest: str | null
  summary:
    total: int
    queued: int
    running: int
    succeeded: int
    failed: int
    cancelled: int
```

### `global/batches/<batch_id>.json` 最终字段格式

`global/batches/<batch_id>.json` 是 batch 的真相文件。

建议固定为：

```yaml
meta:
  revision: int
  created_at: str
  updated_at: str
  updated_by_machine: str

batch:
  batch_id: str
  name: str | null
  source_manifest: str | null
  machine_name: str
  task_ids: list[str]
  summary:
    total: int
    queued: int
    running: int
    succeeded: int
    failed: int
    cancelled: int
    blocked: int
    orphaned: int
  policy:
    allow_retry_failed: bool
    allow_retry_cancelled: bool
```

字段约束：

- `meta.revision` 每次成功写入都必须递增
- `batch.batch_id` 必须与文件名一致
- `batch.machine_name` 表示这次 batch-submit 的提交机器
- `batch.task_ids` 是这批 task 的正式成员列表
- `batch.summary.*` 是聚合字段，可由 task 真相重建

设计原则：

- batch 是组织容器，不是执行主体
- task 真相优先于 batch 摘要
- 若 batch 摘要损坏，可通过 task 列表重建

### batch 真相层最小写入原则

只有以下动作应改写 `global/batches/<batch_id>.json`：

- batch-submit 创建 batch
- batch 下新 task 创建完成后补全 `task_ids`
- batch-retry-failed 产生新 task 后更新摘要
- batch-retry-cancelled 产生新 task 后更新摘要
- rebuild-index / repair 重新计算摘要

不允许把 batch 当作 task 状态真相来源。

## shared root 真相层规则

shared root 下真正的 source of truth 只有：

- `global/tasks/<task_id>.json`
- `global/batches/<batch_id>.json`
- `machines/<machine_name>/machine.json`

其余内容都应视为派生层或辅助层：

- `global/indexes/`
- `global/events/`
- `machines/<machine_name>/state/`
- `machines/<machine_name>/claims/`
- `machines/<machine_name>/events/`

恢复顺序也应明确：

1. 先信任 task 真相
2. 再信任 batch 真相
3. 最后重建索引和 machine 侧辅助视图

## 并发写策略

### 1. 单对象原子写

共享对象写入必须使用：

1. 写 tmp
2. fsync
3. rename

### 2. revision CAS

所有共享对象必须带：

```yaml
meta:
  revision: int
  updated_at: str
  updated_by_machine: str
```

写入时：

- 先读 revision
- 写入时校验 revision
- 不匹配就重读重试

### 3. 锁只保护关键路径

建议只在这些路径上锁：

- submit
- batch-submit
- migrate
- machine init

锁文件建议放在：

```text
<shared-root>/global/locks/
```

### 4. 索引不是 source of truth

真相是主对象文件：

- tasks/
- batches/
- machines/

索引失败可以重建，不需要伪事务。

## agent 模型

### 1. on_demand

默认模式：

- 提交 task 时若本机 agent 不在，则可本机唤起
- 连续 600 秒无任务活动则自动退出

### 2. persistent

仅在用户显式选择时启用：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a --agent-mode persistent
```

或：

```bash
qexp agent start --persistent
```

### 3. 默认不做远程唤起

不支持：

- 从机器 A 远程拉起机器 B 的 agent
- 从机器 A 远程投递 task 到机器 B

因此 agent 唤起只讨论本机：

- `qexp submit`
- `qexp batch-submit`
- `qexp retry`

## 调度一致性

### 1. 当前机器只处理属于自己的 queued task

一旦 `task.machine_name` 写定：

- 只有对应机器能 dispatch 它

### 2. dispatch 前必须 CAS 改状态

agent dispatch task 时：

1. 读取 task
2. 确认 `phase == queued`
3. CAS 更新到 `dispatching`
4. 成功后才占 GPU

### 3. orphaned 检测

若 shared root 中某个 task 显示：

- `dispatching`
- `starting`
- `running`

但本机 runtime 已找不到对应执行证据，则应转为：

- `orphaned`

## 调度事件记录

`qexp` 只记录调度事件，不记录训练日志。

建议事件包括：

- submit_succeeded
- submit_failed
- task_started
- task_finished
- task_cancelled
- task_failed
- task_orphaned

事件可以存放在：

```text
<shared-root>/global/events/
<shared-root>/machines/<machine_name>/events/
```

## doctor 能力

建议保留：

- `qexp doctor`
- `qexp doctor rebuild-index`
- `qexp doctor repair-orphans`
- `qexp doctor cleanup-locks`

## 验收标准

- shared 模式下 machine 名必须显式提供
- 每个 machine 在 shared root 下都有自己的私有子目录
- 本机只写自己的 machine 子目录
- 默认不支持远程投递和远程唤起
- 默认 agent 为 on_demand，空闲 600 秒退出
- `qexp` 只记录调度事件，不负责训练日志

# qexp Runtime Spec

状态：草稿

更新时间：2026-04-17

## 目标

本文档只讨论实现层问题：

1. shared root 怎么组织
2. machine 怎么登记
3. 并发写怎么处理
4. agent 如何在 on-demand / persistent 两种模式下运行

产品定位、CLI、工作流不在本文档展开，见：

- [qexp_product_spec.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_product_spec.md)

## 文档边界与实现状态

实现状态说明：

- 本文档同时承载当前 runtime 契约与已接受但**尚未完全落地**的目标契约
- 若某项能力明确标注为 **假设/未验证**，表示它已进入推荐设计方向，但当前安装版本未必已经实现
- 当前最需要注意的未完全落地点包括：`group` 字段、`--group` CLI 参数、以及 `batch.group` manifest 字段

## Runtime 总览

### 系统平面

#### 1. shared control root

正式形态：

```text
/mnt/share/myproject/.qexp
```

- `shared_root` 必须按 project 级切分
- `shared_root` 必须直接命名为 `.qexp`
- `shared_root` 必须直接位于 project root 下

不推荐形态：

```text
/mnt/share/myproject/.qexp/exp1/shared
/mnt/share/myproject/.qexp/exp2/shared
```

原因：

- `shared_root` 代表一整套控制平面
- 同一项目内的 task / batch / machine / indexes / events 应共享同一套真相
- 不同实验计划、group、batch 不应各自拥有独立 `shared_root`
- `qexp` 运行时会把 `project_root/.qexp` 作为正式控制目录契约，而不是仅文档推荐

职责：

- machine 元数据
- task 元数据
- batch 元数据
- 全局索引
- 锁
- 调度事件记录

#### 2. local runtime root

建议：

```text
~/.qqtools/qexp-runtime
```

职责：

- agent pid
- local locks
- heartbeat
- wrapper / runtime 临时状态

### shared root 真相层规则

shared root 下正式 truth domains 固定为：

- `global/tasks/<task_id>.json`
- `global/batches/<batch_id>.json`
- `machines/<machine_name>/machine.json`
- `machines/<machine_name>/state/agent.json`
- `machines/<machine_name>/claims/active/<task_id>.json`

其余内容都应视为派生层或辅助层：

- `global/indexes/`
- `global/events/`
- `machines/<machine_name>/state/gpu.json`
- `machines/<machine_name>/state/summary.json`
- `machines/<machine_name>/claims/released/`
- `machines/<machine_name>/events/`

这条规则需要进一步明确到目录设计：

- 不允许为不同 experiment / group 建立独立真相目录
- 不允许出现 `groups/<group>/tasks/...` 或 `groups/<group>/batches/...`
- 不允许把 group 目录作为 submit / retry / clean / repair 的依赖真相

如果未来需要 group 级目录，最多只允许 machine-local 或离线可重建的派生产物，例如：

- **假设/未验证**：`derived/groups/<group>/summary.json`

这些目录必须满足：

- 删除后可完全重建
- 不得成为 source of truth
- 不得改变恢复顺序中的真相优先级

恢复顺序也应明确：

1. 先信任 task 真相
2. 再信任 batch 真相
3. 再信任 agent lifecycle truth 与 active claim truth
4. 最后重建索引和 machine 侧辅助视图

single-task clean 生效后，恢复规则补充如下：

- 若 task 真相已删除，则不允许为了“回滚 clean”而凭空重建该 task 文件
- 若 batch 真相或索引尚未修正完成，应通过 repair / rebuild 流程继续收敛，而不是把已删除 task 重新视为存在

### 共享目录布局

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
      tasks_by_state/
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

这里的目录组织原则必须写死为：

- 按对象类型组织
- 按 machine 私有区组织

禁止按 experiment / group 目录组织真相层，例如：

```text
<shared-root>/exp1/
<shared-root>/exp2/
<shared-root>/groups/<group>/tasks/
<shared-root>/groups/<group>/batches/
```

这些形态都会制造第二套真相组织维度，不属于正式 runtime layout。

原则：

- `global/` 是共享真相
- `machines/<machine_name>/` 是该机器在共享目录中的私有区
- 本机只写自己的 machine 子目录
- 机器之间不直接写对方子目录

## machine 模型

### 为什么每个机器要有自己的共享子目录

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

### machine 身份

#### 1. 共享模式下 machine 名必填

由于同一 Docker 镜像可起多个容器，hostname 不可靠。

因此：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a
```

中的 `--machine` 必填。

#### 2. 不以 hostname 作为主键

`hostname` 可以记录在 `machine.json` 里作为辅助信息，但不能作为 machine 主身份。

machine 主键必须是用户显式提供的 `machine_name`。

### machine 共享子目录

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

- `schema_version`
- `machine_name`
- `agent_mode`
- `agent_state`
- `pid`
- `started_at`
- `last_heartbeat`
- `last_transition_at`
- `idle_timeout_seconds`
- `idle_started_at`
- `idle_deadline_at`
- `drain_started_at`
- `last_exit_reason`
- `workset`

用途：

- 作为 agent 生命周期状态的唯一真相源
- 让其他命令看到这台机器当前 agent 是否活着
- 区分 `active`、`draining`、`idle` 等生命周期位置
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

- `machine_name`
- `counts_by_phase`
- `updated_at`

用途：

- 提供轻量 machine phase/count 总览
- 作为 `qexp top` / `qexp machines` 的展示缓存
- 避免每次都全量扫 task 文件

约束：

- `state/summary.json` 不是生命周期状态真相源
- 不得承载独立的 `agent_state` 或 `has_active_responsibility`
- 凡涉及 agent 生命周期判定，必须回到 `state/agent.json`

#### 5. `claims/active/<task_id>.json`

表示这台机器已成功 claim 某个 task，且该 claim 当前仍生效。

承载：

- `task_id`
- `machine_name`
- `claimed_at`
- `revision_at_claim`

用途：

- 帮助排查 dispatch 归属
- 作为 execution ownership truth
- 为 orphan repair 提供执行责任证据

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

- `global/tasks/*.json` 是 task truth
- `global/batches/*.json` 是 batch truth
- `machines/<machine>/state/agent.json` 是 agent lifecycle truth
- `machines/<machine>/claims/active/*.json` 是 execution ownership truth
- `machines/<machine>/state/summary.json` / `state/gpu.json` 是 machine 侧派生视图
- `machines/<machine>/claims/released/*.json` 是审计证据
- `events/` 是 machine 维度事件审计

也就是说：

- `global/` 承载 task truth 与 batch truth
- `machines/<machine>/...` 同时承载 machine 私有 truth objects 与 machine 侧派生/审计对象

### machine 对象

`machines/<machine_name>/machine.json` 至少包含：

```yaml
machine:
  machine_name: str
  hostname: str | null
  shared_root: str
  runtime_root: str
  agent_mode: enum[on_demand, persistent]
  gpu_inventory:
    count: int
    visible_gpu_ids: list[int]
```

`machines/<machine_name>/state/agent.json` 至少包含：

```yaml
agent:
  schema_version: "3.0"
  machine_name: str
  agent_mode: enum[on_demand, persistent]
  agent_state: enum[stopped, starting, active, draining, idle, stale, failed]
  pid: int | null
  started_at: str | null
  last_heartbeat: str | null
  last_transition_at: str | null
  idle_timeout_seconds: int
  idle_started_at: str | null
  idle_deadline_at: str | null
  drain_started_at: str | null
  last_exit_reason: str | null
  workset:
    queued_count: int
    dispatching_count: int
    starting_count: int
    running_count: int
    terminal_count: int
    has_launch_backlog: bool
    has_active_responsibility: bool
    updated_at: str
```

## 共享对象契约

### task 对象

```yaml
task:
  task_id: str
  name: str | null
  group: str | null
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
  group: str | null
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
- `task.group` 是 project 内长期归组键；允许为 null
- `task.machine_name` 一旦写入后不得随意迁移
- `task.spec.command` 是 task 真正执行命令
- `task.status.phase` 是 task 当前唯一正式状态
- `task.lineage.retry_of` 仅在 retry 产生的新 task 上存在

`group` 的额外语义约束：

- `group` 是工具层归组键，不是控制平面边界
- `group` 不得隐式映射到独立 `shared_root`
- `group` 不得要求在 `shared_root` 下拥有独立真相目录
- `group` 的缺省值为 `null`

设计原则：

- 这是唯一正式 task 真相
- indexes 只是派生视图
- machine 子目录中同时包含 machine-side truth objects、派生视图与审计对象

补充约束：

- `tasks_by_state` 只可作为 runtime 候选集，不得覆盖 `task.status.phase`
- 若 `tasks_by_state` 与 task truth 冲突，必须以 task truth 为准

### batch 对象

```yaml
batch:
  batch_id: str
  name: str | null
  group: str | null
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
  group: str | null
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
- `batch.group` 表示该 batch 默认归属的 project 内 group；允许为 null
- `batch.machine_name` 表示这次 batch-submit 的提交机器
- `batch.task_ids` 是这批 task 的正式成员列表
- `batch.summary.*` 是聚合字段，可由 task 真相重建

设计原则：

- batch 是组织容器，不是执行主体
- task 真相优先于 batch 摘要
- 若 batch 摘要损坏，可通过 task 列表重建

batch 与 group 的关系约束：

- `batch` 是一次批量提交形成的操作集合
- `group` 是长期归组键
- 一个 `group` 内允许存在多个 `batch`
- 一个 `batch` 通常应默认归属于一个 `group`
- `batch` 不得替代 `group`

## 运行时语义

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
- resubmit 删除旧 task 并创建同 `task_id` 新 task

任何机器级缓存、GPU 快照、辅助状态都不得反向覆盖 task 真相文件。

### `submit / retry / resubmit` 的真相层规则

`submit / retry / resubmit` 在 task 真相层的正式约束：

- `submit` 创建的新 task 必须满足：
  - `task.task_id` 是一个当前不存在于真相层的新主键
  - `task.lineage.retry_of = null`
- `retry` 创建的新 task 必须满足：
  - 使用新的 `task_id`
  - `task.lineage.retry_of = <old_task_id>`
  - 原 task 真相文件继续保留
- `resubmit` 创建的新 task 必须满足：
  - 复用被替换旧 task 的 `task_id`
  - `task.lineage.retry_of = null`
  - 新 task 写入前，旧 task 真相文件必须已被正式删除

因此：

- 同一个 `task_id` 在任一时刻最多只允许对应一条正式 task 真相
- 不允许通过 `submit` 静默覆盖已有 `task_id`
- 不允许通过 `retry` 复用原 `task_id`
- `resubmit` 是唯一允许复用旧 `task_id` 的正式入口

### resubmit 的真相层规则

`resubmit` 是一个显式双阶段动作，但对外语义必须稳定：

1. 校验目标 task 当前处于允许被 `resubmit` 的终态
2. 校验该 task 不属于受限场景
3. 持久化一条独立的 resubmit operation 真相
4. 删除旧 task 真相、相关索引项以及 best-effort runtime log
5. 用相同 `task_id` 创建一个新的 task 真相文件
6. 将 operation 推进为 committed，或在中断后由 `qexp doctor repair` 继续收敛

默认限制：

- 只允许 `failed` / `cancelled`
- 不允许 `queued` / `dispatching` / `starting` / `running`
- 不允许 batch 成员 task

一致性要求：

- 不允许在 `resubmit` 结束后同时留下“旧 task 内容”和“新 task 内容”的混合真相
- 不允许把旧 task 的 `lineage.retry_of`、终态 `status.reason`、`result.*` 直接带入新 task
- 新 task 必须表现为一次新的首次提交，而不是一次可见的 retry
- `resubmit` 可以复用 single-task clean 的底层删除逻辑，但必须由 `resubmit` 自己掌控完整流程与一致性边界，不能退化成两个独立公开命令的松散拼接
- 在 `creating_new` 阶段，repair 必须按 prepared replacement snapshot 识别已创建的新 task，而不是把运行中 replacement task 误判为残留旧真相

### batch 真相层最小写入原则

只有以下动作应改写 `global/batches/<batch_id>.json`：

- batch-submit 创建 batch
- batch 下新 task 创建完成后补全 `task_ids`
- batch-retry-failed 产生新 task 后更新摘要
- batch-retry-cancelled 产生新 task 后更新摘要
- single-task clean 删除 batch 成员后同步修正 `task_ids` 与摘要
- rebuild-index / repair 重新计算摘要

不允许把 batch 当作 task 状态真相来源。

## 派生索引族分层

### Tier A: runtime-critical projection

- `global/indexes/tasks_by_state/*.json`

约束：

- 允许 runtime 用于 phase 候选集枚举
- 读到候选后必须回读 task truth
- 发现 phase/index 不一致时，应优先执行单任务纠偏

### Tier B: removed persistent membership projections

以下索引族不再属于长期正式架构：

- `global/indexes/tasks_by_batch/*.json`
- `global/indexes/tasks_by_machine/*.json`
- `global/indexes/tasks_by_group/*.json`
- `global/indexes/batches_by_group/*.json`

正式要求：

- 不再由运行时写路径维护
- 不再作为 verify / rebuild 的治理对象存在
- batch / group / machine listing 与 summary 直接从 truth objects 计算

### Tier C: offline recovery substrate

- `qexp doctor rebuild-index`

约束：

- 必须可从 truth 全量重建 `tasks_by_state`
- 应清理历史遗留的 Tier B index 目录
- 不得反向推断或覆盖 truth objects

### single-task clean 对 batch 的修正规则

当 `global/tasks/<task_id>.json` 对应 task 带有 `batch_id` 时，single-task clean 必须同步改写对应 batch 真相。

规则固定为：

1. 从 `batch.task_ids` 中移除被删除的 `task_id`
2. 基于删除后的正式成员集合重算 `batch.summary.*`
3. 保留 `batch.policy`、`batch.name`、`batch.machine_name` 等非聚合字段
4. 若删除后该 batch 不再包含任何 task，batch 真相仍然保留，不自动删除 `global/batches/<batch_id>.json`

约束：

- 不允许留下仍引用已删除 `task_id` 的 `batch.task_ids`
- 不允许只修索引而不修 batch 真相
- batch 摘要必须视为可重建聚合值，不得把已删除 task 继续计入摘要

### clean 运行时语义

#### 1. single-task clean 的对象边界

首版 single-task clean 只允许直接修改或删除以下共享层对象：

- 删除 `global/tasks/<task_id>.json`
- 改写相关 `global/indexes/...` 派生条目
- 若存在 `batch_id`，改写 `global/batches/<batch_id>.json`

首版不应把以下对象纳入 clean 的强契约：

- `machines/<machine_name>/events/...`
- `machines/<machine_name>/claims/...`
- 训练框架自身业务日志、checkpoint、artifact

#### 2. runtime log 语义

runtime log 不属于 shared root 真相层。

因此 single-task clean 对 runtime log 的契约固定为：

- 若实现能从 task 元数据稳定定位到 log 路径，且当前进程对该路径可访问，则按 best-effort 删除
- 若路径不存在、机器不可达、权限不足或路径无法确定，不得阻断 shared root 中 task/batch/index 的 clean 主流程
- CLI 或 API 返回值必须区分“task 已清理”与“runtime log 未删除”

这意味着：

- clean 成功的判定以 shared root 真相与派生视图收敛成功为准
- runtime log 删除失败不构成 task 真相删除的回滚条件

## 读取路径规则

### runtime critical readers

- `scheduler`
- `runner` 相关 phase/liveness 判断
- orphan / stale recovery

规则：

- 仅 `tasks_by_state` 可进入 runtime critical 候选集路径
- task phase 判定回到 `global/tasks/*.json`
- agent lifecycle 判定回到 `machines/<machine>/state/agent.json`
- execution ownership 判定回到 `machines/<machine>/claims/active/*.json`

### query and listing readers

- `observer`
- list / show / status APIs

规则：

- batch / group / machine 结果直接从 truth objects 计算
- 不再依赖持久化 membership indexes
- 最终展示结果必须以 truth objects 为准

### summary builders

- machine workset builder
- machine summary / gpu summary builders

规则：

- 直接从 task truth 聚合
- `workset` / `summary` 不得反向定义 agent lifecycle truth

## Legacy Membership Index Cleanup

历史 `tasks_by_batch` / `tasks_by_machine` / `tasks_by_group` / `batches_by_group`
目录若仍存在，只视为待清理遗留物，不再属于正式治理面。

恢复要求：

- 新写路径不得继续写入这些目录
- `doctor rebuild-index` 应清理这些目录

- 允许短暂漂移
- 不允许长期明显不准
- 不允许无人知晓地长期失真

#### 3. single-task clean 的失败模型

single-task clean 应分为两个阶段：

前置校验阶段：

- 读取 task 真相
- 校验终态
- 若有 `batch_id`，读取 batch 真相
- 预计算 batch 修正结果与索引修正结果

这一阶段失败时，必须不写入任何对象。

执行阶段：

1. 写入 batch 真相修正结果（若存在）
2. 写入索引修正结果
3. 删除 `global/tasks/<task_id>.json`
4. best-effort 删除 runtime log

允许的中间状态：

- task 已删除，但 machine 本地 log 尚未删除
- **假设/未验证**：若执行阶段在 batch/索引修正后半途失败，系统可暂时依赖 repair / rebuild-index 继续收敛

不允许的持久状态：

- batch 真相仍保留对已删除 task 的正式成员引用
- 索引长期把已删除 task 暴露为有效对象

### agent 模型

#### 1. on_demand

默认模式：

- 提交 task 时若本机 agent 不在，则可本机唤起
- 连续 600 秒无任务活动则自动退出

#### 2. persistent

仅在用户显式选择时启用：

```bash
qexp init --shared-root /path/to/shared --machine gpu2a --agent-mode persistent
```

或：

```bash
qexp agent start --persistent
```

#### 3. 默认不做远程唤起

不支持：

- 从机器 A 远程拉起机器 B 的 agent
- 从机器 A 远程投递 task 到机器 B

因此 agent 唤起只讨论本机：

- `qexp submit`
- `qexp batch-submit`
- `qexp retry`

### 调度一致性

#### 1. 当前机器只处理属于自己的 queued task

一旦 `task.machine_name` 写定：

- 只有对应机器能 dispatch 它

#### 2. dispatch 前必须 CAS 改状态

agent dispatch task 时：

1. 读取 task
2. 确认 `phase == queued`
3. CAS 更新到 `dispatching`
4. 成功后才占 GPU

#### 3. orphaned 检测

若 shared root 中某个 task 显示：

- `dispatching`
- `starting`
- `running`

但本机 runtime 已找不到对应执行证据，则应转为：

- `orphaned`

### doctor 能力

建议保留：

- `qexp doctor`
- `qexp doctor verify`
- `qexp doctor repair`
- `qexp doctor rebuild-index`
- `qexp doctor repair-orphans`
- `qexp doctor cleanup-locks`

与 single-task clean 的关系：

- `rebuild-index` 负责收敛 clean 后的派生索引视图
- `repair` 负责收敛中断中的 metadata repair operation，并对 repair 触达的 batch truth / targeted stale task indexes 做定点收敛
- `repair` 不应吸收 `rebuild-index` 的核心语义；二者语义必须保持区分
- `repair-orphans` 不负责修复 clean 产生的 batch 成员引用问题
- `repair-orphans` 必须联合 task truth、agent lifecycle truth、active claim truth 做 orphan 判定
- 历史 membership index 目录若仍残留，应通过 `rebuild-index` 清理，而不是继续扩大对这些遗留对象的依赖

`doctor verify` 的输出应至少包含：

- `ok`
- `severity`
- `issues`
- `messages`
- `index_drift`
- `recommended_actions`
- `diagnosis`
- `issue_count_by_category`
- `issue_count_by_code`

其中：

- `severity` 用于区分 truth corruption、metadata gap、derived drift 的处理优先级
- `issues` 必须是结构化对象数组，而不是仅字符串数组；每项至少包含 `code`、`category`、`severity`、`message`
- `messages` 是给人读的投影层，不得作为唯一机器语义接口
- `recommended_actions` 必须是结构化对象数组；每项至少包含 `action_code`、`command`、`blocking`、`reason`
- `recommended_actions` 必须明确建议是执行 `repair`、`rebuild-index`、`repair-orphans`，还是进入人工修复
- `blocking` 用于区分“必须先处理的阻塞性治理动作”和“可延后汇总处理的非阻塞动作”
- `recommended_actions` 的生成规则必须来自显式 policy table；正式语义应是 `issue_code -> action_code` 映射，而不是散落在代码中的文本匹配或隐式分支
- `diagnosis` 应至少暴露 truth 是否可读、是否仅为 index drift、是否存在 resubmit gap、是否存在 batch truth drift
- `issue_count_by_category` / `issue_count_by_code` 应服务于 CI、巡检与治理报表聚合

`doctor verify` 还必须支持面向治理系统的策略接口：

- `qexp doctor verify`：默认 observation mode，只输出诊断结果，不因发现问题而把命令本身视为执行失败
- `qexp doctor verify --strict`：把任意治理发现（`low` 及以上）映射为非零策略退出
- `qexp doctor verify --fail-on <severity>`：按阈值触发策略退出，阈值集合固定为 `low` / `medium` / `high`
- `qexp doctor verify --jsonl`：输出 line-delimited structured records，至少包含 `verify_summary`、`verify_issue`、`verify_recommendation`、`verify_result`
- `verify_issue` record 必须显式输出 `issue_code`、`category`、`severity`，禁止要求下游系统解析 message 文本推断治理语义
- `verify_recommendation` record 必须显式输出 `action_code`、`blocking`、`command`，禁止要求下游系统解析 reason 文本推断动作语义

策略退出约定：

- 退出码 `0`：诊断命令执行成功，且未触发所请求的 fail policy
- 退出码 `2`：诊断命令执行成功，但结果触发了 `--strict` 或 `--fail-on` 的治理策略
- 退出码 `1` 或其他常规非零码：CLI/运行时错误，而不是治理判定本身

## 一致性与写入机制

### 并发写策略

#### 1. 单对象原子写

共享对象写入必须使用：

1. 写 tmp
2. fsync
3. rename

#### 2. revision CAS

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

#### 3. 锁只保护关键路径

建议只在这些路径上锁：

- submit
- batch-submit
- migrate
- machine init

锁文件建议放在：

```text
<shared-root>/global/locks/
```

#### 4. 索引不是 source of truth

真相是对应 truth objects：

- `global/tasks/`
- `global/batches/`
- `machines/<machine>/machine.json`
- `machines/<machine>/state/agent.json`
- `machines/<machine>/claims/active/`

索引失败可以重建，不需要伪事务。

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

## 验收标准

- shared 模式下 machine 名必须显式提供
- 每个 machine 在 shared root 下都有自己的私有子目录
- 本机只写自己的 machine 子目录
- 默认不支持远程投递和远程唤起
- 默认 agent 为 on_demand，空闲 600 秒退出
- `qexp` 只记录调度事件，不负责训练日志

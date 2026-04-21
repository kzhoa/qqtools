# qexp Truth Domains And Derived Views

状态：已实现归档

更新时间：2026-04-21

完成时间：2026-04-20

## 背景

当前 `qexp` 的运行时状态系统同时依赖多类数据：

1. task 真值文件：`global/tasks/<task_id>.json`
2. machine agent 状态：`machines/<machine_name>/state/agent.json`
3. 执行归属证据：`machines/<machine_name>/claims/active/<task_id>.json`
4. 状态索引文件：`global/indexes/tasks_by_state/<phase>.json`

按长期设计，至少有三类对象应被区分：

- task truth：回答“task 现在是什么状态”
- agent lifecycle truth：回答“这台机器上的 agent 现在是否活着、处于什么生命周期位置”
- execution ownership truth：回答“这台机器是否仍持有某个 task 的执行责任证据”

状态索引只是派生缓存，用于加速 `queued` / `dispatching` / `starting` / `running` 等视图与扫描。

当前实现已经建立了部分保护，例如：

- 已有统一的单任务状态索引收敛原语
- `scheduler reconcile` 已按“索引枚举 + 真值确认 + 不匹配先纠偏”工作

但系统层面的真相边界仍未被完整冻结。只要仍有其他路径把“索引中的 phase”直接当成可信输入，或在新增逻辑时绕开既有原语，索引残留、漏删、重复写或并发更新异常仍可能把系统带回错误分支。

这说明当前问题不是“某一处漏了 defensive check”，而是：

- 真相层与派生层的主从关系没有在代码里被强制表达
- 统一的单任务收敛原语虽然已经出现，但尚未被完整推广为唯一正式写路径
- repair 逻辑与正常运行逻辑之间没有明确的职责边界

## Scope Boundary

本 pitch 的主目标是冻结 `qexp` 的真相域模型与派生视图边界。

因此它会完整规范：

- `task truth`
- `agent lifecycle truth`
- `execution ownership truth`
- `tasks_by_state` 这一类 phase index 的运行时收敛原则

它也会覆盖其他派生索引族，但只覆盖到以下层级：

- 它们是不是真相
- 它们当前属于哪种维护等级
- 它们是否允许参与运行时关键决策

它不会在本文件里完整展开 `tasks_by_batch` / `tasks_by_machine` / `tasks_by_group` / `batches_by_group` 的在线收敛协议、单对象同步原语、调用者职责和迁移步骤。

这些内容应由独立 pitch 继续回答，因为那已经属于“派生索引族的维护架构设计”，不再只是“真相模型定义”。

## 问题定义

当前状态系统仍存在三个结构性风险。

### 1. 索引被错误地当成状态机输入真相

像 `scheduler reconcile` 这类逻辑，本质上应该根据 task 真值决定：

- 任务现在真实处于哪个 phase
- 这个 phase 允许执行什么检查
- 是否应进入下一步处理

但如果逻辑直接按 `load_index(..., phase)` 的 phase 分支执行，就会把派生缓存提升成行为真相。

### 2. 派生索引缺少幂等收敛能力

历史上常见的更新模式是：

- 从 `old_phase` 删除
- 向 `new_phase` 添加

这种写法只能表达“我相信自己知道旧值”，不能表达“把这个 task 在所有错误位置都清掉，再只保留真实位置”。一旦历史上已有脏残留，后续正常相位迁移也不会自动修复。

### 3. repair 与 runtime 没有统一协议

当前系统里：

- runtime 会维护索引
- `doctor rebuild-index` 会全量重建索引
- `doctor repair` 会收敛未完成 metadata repair operation
- observer / scheduler / api 未来若继续各自形成隐含规则，仍会放大职责漂移

这说明当前仍需要把“一致性维护”进一步收敛成单一协议，而不是继续依赖局部习惯。

## 目标

### 必须

1. 明确 `task truth`、`agent lifecycle truth`、`execution ownership truth` 的职责边界
2. task 状态机决策只以 task 真值为主输入
3. agent 生命周期判定只以 agent 真值为主输入
4. 执行归属子问题只以 claim/ownership 真值为主输入；orphan repair 必须联合 task / agent / claim 三域显式判定
5. 状态索引降级为纯派生缓存，允许脏、允许缺、允许重建
6. 正常运行路径必须具备单任务级自修复能力
7. rebuild-index 工具必须能在全局层面把派生层收敛回真值

### 应该

1. 为所有状态索引操作提供统一原语，而不是散落手写增删
2. 将“索引不一致”明确定义为可恢复状态，而不是异常真相
3. 让 scheduler / observer / api / doctor 共享同一套一致性规则
4. 为未来新增 phase、operation、派生视图提供稳定扩展面

### 可以

1. 引入索引健康度观测指标
2. 将派生层收敛事件纳入 event stream
3. 为批量 rebuild 引入更细粒度的统计与审计输出

## 第一性原理结论

状态系统必须回答三个根问题：

1. 哪个对象定义“任务现在是什么状态”
2. 哪个对象定义“某台机器上的 agent 现在处于什么生命周期”
3. 哪个对象定义“某台机器是否仍持有某个 task 的执行责任”

在 `qexp` 里，答案必须固定为：

1. `global/tasks/<task_id>.json` 定义真实 phase
2. `machines/<machine_name>/state/agent.json` 定义 agent lifecycle truth
3. `machines/<machine_name>/claims/active/<task_id>.json` 定义 execution ownership truth

而以下对象只应作为派生或审计视图存在：

1. `tasks_by_state/*.json`
2. `state/summary.json`
3. `state/gpu.json`
4. `claims/released/*.json`

因此所有正式设计都必须满足：

- 允许索引错，但不允许因为索引错而做出错误状态转移
- 允许索引缺，但不能因此丢失 task 真值
- 允许 machine summary 滞后，但不能因此误判 agent 生命周期
- 允许 released claim 仅作审计，但不能用它反向定义 active ownership
- ownership claim 可以定义“谁当前持有执行责任”，但不能单独完成 orphan repair 判定
- rebuild-index 的职责是“让派生重新贴合真值”，而不是“猜测真值”

## 设计原则

1. 多真相域分治：task、agent、ownership 各自只回答本域问题
2. 禁止跨域偷推断：任何真相域不得被其他域的派生视图反向定义
3. 派生可重建：任意状态索引删除后都能从 task 真值重建
4. 决策看真值：状态机分支必须读取对应真相域，而不是读取缓存
5. 联合判定显式化：跨域判断必须显式 join task truth、agent truth、ownership truth
6. 正常路径自修复：运行时发现不一致时，应先纠偏再继续
7. 全局 rebuild 兜底：后台存在全量重建路径，但不依赖人工日常介入
8. 幂等优先：重复执行同一次索引收敛，不应产生额外副作用

## 推荐最终形态

### 1. 明确真相域与派生层

#### 真相域 A：Task Truth

- `global/tasks/<task_id>.json`

回答：

- task 当前 phase 是什么
- task 属于哪台机器
- task 下一步允许执行什么状态转移
- task 最终结果是什么

#### 真相域 B：Agent Lifecycle Truth

- `machines/<machine_name>/state/agent.json`

回答：

- 该 machine agent 是否存活
- agent 当前处于 `stopped` / `starting` / `active` / `draining` / `idle` / `stale` / `failed` 哪个位置
- 该 machine 是否仍承担 agent 生命周期责任

#### 真相域 C：Execution Ownership Truth

- `machines/<machine_name>/claims/active/<task_id>.json`

回答：

- 该 machine 是否仍持有某个 task 的 active execution claim
- 该 claim 在哪个 revision 上建立
- orphan / repair 时这台机器是否仍拥有执行责任证据

#### 其他持久化真相对象

- `global/batches/<batch_id>.json`
- `machines/<machine_name>/machine.json`

#### 派生层

- `global/indexes/tasks_by_state/*.json`
- `global/indexes/tasks_by_batch/*.json`
- `global/indexes/tasks_by_machine/*.json`
- `global/indexes/tasks_by_group/*.json`
- `machines/<machine_name>/state/summary.json`
- `machines/<machine_name>/state/gpu.json`
- `machines/<machine_name>/claims/released/*.json`

状态索引必须明确标注为：

- 非真相
- 可缺失
- 可重建
- 不参与真值定义

当前实现里，这些派生索引还需要区分两种收敛等级：

- `tasks_by_state` 已有正式单任务收敛原语与运行时纠偏职责
- `tasks_by_batch` / `tasks_by_machine` / `tasks_by_group` 当前仍以 `doctor rebuild-index` 的全量重建为正式维护路径

这意味着长期上有两条可接受路径，但文档必须显式二选一：

- 要么后续为其补齐单对象收敛原语与运行时职责边界
- 要么继续把它们定义为“仅离线 rebuild 维护”的派生索引

其中：

- `claims/active/*.json` 是 ownership truth，不是普通缓存
- `claims/released/*.json` 是审计线索，不是 active ownership truth
- `state/summary.json` / `state/gpu.json` 是 machine 侧投影，不是 lifecycle truth

### 2. 引入统一索引收敛原语

当前正式落地的统一索引收敛原语只覆盖 phase/state index：

- `sync_task_state_index(cfg, task_id, actual_phase)`

其正式语义应是：

1. 以 `actual_phase` 作为唯一真相输入
2. 将该 `task_id` 从所有非 `actual_phase` 的状态索引中移除
3. 将该 `task_id` 添加到 `actual_phase` 的状态索引
4. 整个过程允许重复执行，结果保持不变

`update_index_on_phase_change(old_phase, new_phase)` 可以保留为兼容包装，但内部必须退化为对 `sync_task_state_index(...)` 的调用。

**假设/未验证**：若未来要把 `tasks_by_batch` / `tasks_by_machine` / `tasks_by_group` 也纳入在线收敛，则应分别补齐各自的单对象同步原语，而不是继续复用 state index 语义硬套。

### 3. 运行时组件一律按“索引枚举 + 真值确认”工作

任何按状态索引遍历 task 的逻辑，都必须遵循统一模板：

1. 通过索引拿到候选 `task_id`
2. 加载 task 真值
3. 校验 task 真值是否匹配当前索引语义
4. 若不匹配，先修正索引
5. 仅在真值匹配时，才执行该 phase 对应的业务逻辑

形式上必须从：

- “按索引 phase 处理 task”

改为：

- “用索引找候选 task，再由 task 真值决定如何处理”

### 4. 跨真相域联合判定规则

工业级长期模型里，运行时经常需要跨域回答更复杂的问题，例如：

- 这个 task 还算不算 running responsibility
- 这台机器挂了之后，哪些 task 应该被判为 orphaned
- 一个 claim 还是否有效

这类问题不得由单一缓存或单一真相域硬推断，而必须显式 join：

1. task truth
2. agent lifecycle truth
3. execution ownership truth

典型规则：

1. 判断 task 业务 phase，只看 task truth
2. 判断 agent 是否存活，只看 agent lifecycle truth
3. 判断 machine 是否仍持有 task 执行责任，只看 active claim truth
4. 判断 orphan / stale recovery，必须联合看 task truth + agent truth + active claim truth

换句话说：

- ownership 子问题可以只由 claim truth 回答
- orphan repair 不是 ownership 子问题的同义词，它是三域联合判定

典型示例：

1. task truth 显示 task 为 `running`
2. agent truth 显示对应 machine 已 `stale`
3. active claim 仍存在，且没有正常 release
4. 此时系统可以把该 task 视为 orphan candidate，并进入显式 repair/orphan 判定流程

这里真正做决定的不是任何单个缓存，而是三个真相域的联合结果。

### 5. 禁止的反向推断

以下推断路径在长期模型里应被明确禁止：

1. 不能用 `tasks_by_state/running.json` 推断 agent 仍然活着
2. 不能用 `state/summary.json` 推断 machine 仍承担 active responsibility
3. 不能用 `agent.json` 反向定义 task 必然仍是 `running`
4. 不能用 `claims/released/*.json` 反向定义 active claim 已合法结束所有业务后果
5. 不能用索引、summary、审计日志去覆盖 task truth、agent truth、ownership truth

### 6. 收敛机制分层

最终形态应有两层收敛机制。

#### A. 在线单任务收敛

由 runtime 组件在日常路径中完成，例如：

- `scheduler`
- `runner`
- `api`
- 未来其他活跃状态消费者

职责：

- 发现当前 task 的索引不一致
- 立即做单任务级纠偏
- 不依赖全局 rebuild

#### B. 离线全局重建

由 `doctor rebuild-index` 或等价工具负责。

职责：

- 扫描全部 task 真值
- 完整重建所有派生索引
- 处理历史脏数据、孤儿索引、缺失索引
- 输出重建统计

在线修复负责降低日常漂移，离线 rebuild 负责全局兜底与版本迁移收口。

### 7. 把“索引不一致”定义为正常可恢复状态

系统不应把索引不一致视为“真值可疑”，而应视为“派生层待收敛”。

因此正式语义应写死：

- task 真值优先级高于任意状态索引
- 状态索引冲突不触发 task phase 改写
- rebuild 索引时，不得反向推断并覆盖 task 真值

## 关键运行时规则

### Scheduler

`reconcile_running_tasks()` 一类逻辑必须满足：

1. active index 只用于发现候选 task
2. 真正的 liveness 检查依据 `task.status.phase`
3. 当 `indexed_phase != task.status.phase` 时，先纠偏索引，再结束当前错误分支
4. 不允许按 stale phase 执行 timeout / crash / fail 分支

若需要判断 task 是否已失去 machine 执行责任，应进一步结合：

1. 对应 machine 的 `state/agent.json`
2. 对应 task 的 active claim 是否仍存在

### Runner

`runner` 在 `starting -> running`、`running -> terminal` 等迁移中：

1. 先写 task 真值
2. 再调用统一索引收敛原语
3. 不假设旧索引一定干净

### Observer / API

所有面向查询的逻辑都应遵循：

1. 索引可用于缩小候选集
2. 展示值以 task 真值为准
3. 若索引与真值冲突，查询结果返回真值，不返回索引幻觉

### Doctor

`doctor rebuild-index` 的目标不是“补丁式修复某个现象”，而是：

- 重新从真值构造派生层
- 校验真值与派生层职责边界
- 保证 rebuild 可重复执行且结果稳定

`doctor repair` 的职责应继续保持为：

- 收敛未完成 metadata repair operation
- 继续被中断的 `resubmit`、single-task clean 等修复流程
- 不承担“重建全部状态索引”的主语义

`doctor repair-orphans` 一类能力在长期模型里应遵循：

- task phase 判定回到 task truth
- machine 存活性判定回到 agent lifecycle truth
- ownership 证据判定回到 active claim truth
- 最终 orphan 化是显式联合判定结果，不是某个单文件推断结果

当前文档应再明确一层 CLI/维护边界：

- `doctor rebuild-index` 负责全量重建 `tasks_by_state`、`tasks_by_batch`、`tasks_by_machine`、`tasks_by_group` 等派生索引
- `doctor repair` 不应吸收 `rebuild-index` 的核心语义
- 若未来希望提供“一键恢复”体验，应新增更高层编排命令，而不是让 `repair` 改名承接所有索引重建语义

## 建议的数据契约

### 状态索引协议

状态索引文件继续保持轻量结构即可，例如：

```json
{
  "task_ids": ["t1", "t2", "t3"]
}
```

但必须补充明确语义：

- `task_ids` 列表只是某一时刻的派生快照
- 列表存在残留或缺项不构成 task 真值变化
- 任何消费者都必须二次读取 task 真值

### Active Claim 协议

`claims/active/<task_id>.json` 长期应被视为 execution ownership truth，而不是普通调试痕迹。

其最低语义应包括：

- `task_id`
- `machine_name`
- `claimed_at`
- `revision_at_claim`

正式语义：

- active claim 存在，表示该 machine 当前仍声明自己持有该 task 的执行责任
- active claim 消失，必须有显式 release 或等价的收敛动作解释
- ownership 判定不能只看 `task.machine_name`，应同时看 active claim 是否仍成立

### Agent State 协议

`state/agent.json` 长期应被视为 agent lifecycle truth。

正式语义：

- 它定义的是 machine agent 生命周期，不是 task phase
- 它可以参与 orphan / stale / idle 等联合判定
- 它不得反向覆盖 task truth

### 可选增强

**假设/未验证**：如果后续需要更强的一致性观测，可考虑为索引文件加上：

- `rebuilt_at`
- `source_revision`
- `generator`

但这些字段仅用于审计与诊断，不改变“task 真值优先”的根约束。

## 迁移策略

### 第一步：统一原语

先把所有状态索引写路径收敛到统一 API：

- 正式写路径用 `sync_task_state_index(...)`
- 历史 `update_index_on_phase_change(...)` 变为兼容包装

### 第二步：修正活跃 runtime

优先改以下组件：

1. `scheduler`
2. `runner`
3. `api cancel/retry/resubmit` 相关状态写路径
4. orphan / repair 路径上的跨域联合判定
5. 其他直接消费 active state index 的模块

### 第三步：补强 doctor

让 `doctor rebuild-index` 输出至少以下信息：

- 重建了多少状态索引文件
- 移除了多少孤儿索引项
- 修复了多少 phase 不一致项
- 是否检测到无法解释的 task 真值异常

### 第四步：冻结契约

在 runtime spec 中明确写死：

- task 文件是 phase 真相
- `state/agent.json` 是 agent lifecycle truth
- `claims/active/*.json` 是 execution ownership truth
- 索引只能派生，不能反驱动状态机
- 跨域恢复必须显式 join 多个真相域，禁止单域硬推断

## 测试要求

至少覆盖以下场景。

### 单元测试

1. `sync_task_state_index()` 会清理多个 stale phase 残留
2. 正常 phase 迁移仍能保持索引正确
3. 重复执行索引收敛不会产生重复项或副作用

### 跨域判定测试

1. task 为 `running` 且 agent 为 `active` 且 active claim 存在时，不得误判 orphaned
2. task 为 `running` 且 agent 为 `stale` 且 active claim 残留时，应进入 orphan candidate 判定
3. task 为 terminal 时，即使存在历史 released claim，也不得反向恢复 active ownership

### 运行时回归测试

1. task 真值已是 `running`，但 `starting` 索引残留时，不得被误判 `startup_timeout`
2. task 真值已是 terminal，但 active 索引残留时，不得重复触发失败逻辑
3. `scheduler` 发现 phase/index 不一致后，下一轮扫描不再重复看到同一脏索引

### Rebuild 测试

1. 删除全部状态索引后能完全从 task 真值重建
2. 混入伪造 task_id、重复 task_id、错误 phase 残留后，rebuild 能收敛到稳定结果
3. rebuild 重复执行两次，结果一致

这里的 `rebuild` 指全局索引重建语义，正式命令应对应 `doctor rebuild-index`，不是 `doctor repair`。

## 不推荐的方案

以下做法都不应作为最终修复。

1. 只在 `scheduler` 某个分支里额外补一个 `if task.status.phase != phase: continue`
2. 继续把 `old_phase -> new_phase` 当成唯一索引更新模型
3. 遇到索引脏数据时只依赖人工执行 `qexp doctor rebuild-index`
4. 在 observer / api 展示层继续直接信任索引 phase
5. 用索引反向猜测并覆盖 task 真值

这些做法最多只能缓解个别症状，不能稳定建立真相层与派生层的边界。

## 影响面

本提案会影响以下模块：

- [src/qqtools/plugins/qexp/indexes.py](/mnt/c/Users/Administrator/proj/qqtools/src/qqtools/plugins/qexp/indexes.py)
- [src/qqtools/plugins/qexp/scheduler.py](/mnt/c/Users/Administrator/proj/qqtools/src/qqtools/plugins/qexp/scheduler.py)
- [src/qqtools/plugins/qexp/runner.py](/mnt/c/Users/Administrator/proj/qqtools/src/qqtools/plugins/qexp/runner.py)
- [src/qqtools/plugins/qexp/observer.py](/mnt/c/Users/Administrator/proj/qqtools/src/qqtools/plugins/qexp/observer.py)
- [src/qqtools/plugins/qexp/api.py](/mnt/c/Users/Administrator/proj/qqtools/src/qqtools/plugins/qexp/api.py)
- [src/qqtools/plugins/qexp/doctor.py](/mnt/c/Users/Administrator/proj/qqtools/src/qqtools/plugins/qexp/doctor.py)
- [docs/spec/qexp_runtime_spec.md](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md)

其中以下能力已经落地：

- `sync_task_state_index(...)` 已存在
- `update_index_on_phase_change(...)` 已退化为兼容包装
- `scheduler.reconcile_running_tasks()` 已具备状态索引不一致时的单任务纠偏路径

## 假设与未验证项

以下内容目前属于**假设/未验证**：

1. **假设/未验证**：当前除 `scheduler` 外，是否还存在其他 runtime 路径会把状态索引 phase 直接当成行为分支输入
2. **假设/未验证**：是否需要把 batch/group/machine 索引也统一升级为同样的“按真值收敛”原语体系
3. **假设/未验证**：未来是否需要为 active task 单独维护更强语义的 workset 快照，以减少对状态索引文件的直接消费
4. **假设/未验证**：机器侧 `state/summary.json`、`state/gpu.json` 是否也需要进一步形式化为“可重建派生层”，同时保持 `state/agent.json` 继续作为 agent 生命周期真相

## 最终建议

长期标准方案应明确为：

- `global/tasks/*.json` 定义 task truth
- `machines/<machine>/state/agent.json` 定义 agent lifecycle truth
- `machines/<machine>/claims/active/*.json` 定义 execution ownership truth
- 状态索引、machine summary、gpu summary、released claim 只是可重建投影或审计
- runtime 一律按“候选集由投影提供，正式决策由对应真相域或多真相域 join 决定”工作
- `doctor rebuild-index` 负责全局索引收敛，不负责定义真相
- `doctor repair` 继续负责未完成 metadata repair operation 的收口，不承担索引重建主语义

这不是为了修某一个 `startup_timeout` 误判，而是为了让 `qexp` 的状态系统在新增 phase、repair、resubmit、跨机观测与未来派生视图扩展时，仍然保持稳定、可解释、可恢复。

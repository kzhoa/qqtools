# qexp Derived Index Families And Convergence Tiers

状态：已实现归档

更新时间：2026-04-21

完成时间：2026-04-21

## 背景

`qexp` 曾同时持久化多类派生索引：

- `global/indexes/tasks_by_state/*.json`
- `global/indexes/tasks_by_batch/*.json`
- `global/indexes/tasks_by_machine/*.json`
- `global/indexes/tasks_by_group/*.json`
- `global/indexes/batches_by_group/*.json`

旧设计试图把它们分成 Tier A / Tier B / Tier C 三层治理。

但如果目标不是“在当前实现上做成本受限的收敛”，而是直接走向长期最终产品形态，那么这个前提本身就需要被推翻：

- `tasks_by_state` 有明确 runtime critical 价值
- 其余 membership-style indexes 只是在重复表达 task truth / batch truth
- 这些重复表达会持续放大 stale、误用、治理和恢复表面积

因此长期最终形态不应是“保留 Tier B 并加强治理”，而应是：

- 保留唯一必要的持久化运行时索引
- 删除其余持久化 membership indexes
- 让 batch / group / machine 视图直接从 truth objects 计算

## 第一性原理

任何持久化 projection 都必须回答一个问题：

它是否值得成为 shared-root 中长期存在的正式对象？

判断标准固定为：

1. 若该 projection 直接参与 runtime critical phase scanning，并且没有它会显著恶化调度与恢复路径，它可以保留
2. 若该 projection 只是把已有 truth 重新编码成 membership 文件，它不应长期持久化
3. 若该 projection 的错误不会污染正式状态机，只会误导 listing / summary / operator judgment，那么最优方案通常不是“治理 stale”，而是“删除该 projection”

据此，`qexp` 的最终答案应是：

- `tasks_by_state` 保留
- `tasks_by_batch` / `tasks_by_machine` / `tasks_by_group` / `batches_by_group` 删除

## 结论

### Tier A: Runtime-Critical Persistent Projection

对象：

- `tasks_by_state`

正式要求：

- 继续作为唯一持久化 runtime index
- 必须具备单对象级在线纠偏能力
- runtime 读取方必须遵循“索引枚举 + truth confirm + mismatch repair”
- `doctor rebuild-index` 必须能从 task truth 全量重建它

### Tier B: Removed Persistent Membership Projections

从最终架构中删除：

- `tasks_by_batch`
- `tasks_by_machine`
- `tasks_by_group`
- `batches_by_group`

正式要求：

- 不再作为 shared-root 正式对象存在
- 不再参与写路径维护
- 不再参与 verify drift 治理
- 不再由 `doctor rebuild-index` 重建

对应能力改为：

- batch listing 直接读取 `global/batches/*.json`
- group filtering 直接读取 `global/tasks/*.json` / `global/batches/*.json`
- machine listing / workset / summary 直接读取 `global/tasks/*.json`

### Tier C: Offline Recovery Substrate

对象：

- `doctor rebuild-index`

正式要求：

- 只负责从 truth 重建 `tasks_by_state`
- 清理历史遗留的 Tier B 索引目录
- 不反向推断 truth

## 为什么旧版 Tier B 方案不是最终形态

旧版方案的问题不在于它“不够强”，而在于它长期上保留了不值得保留的对象。

### 1. `tasks_by_batch` 是重复表达

它重复表达的是：

- `global/tasks/<task_id>.json.batch_id`
- `global/batches/<batch_id>.json.task_ids`

长期保留只会制造第三份 membership 语义。

### 2. `tasks_by_machine` 容易被误当成 lifecycle / ownership 证据

但 machine 真相其实来自：

- task truth 中的 `machine_name`
- `machines/<machine>/state/agent.json`
- `machines/<machine>/claims/active/*.json`

把 `tasks_by_machine` 留在 shared-root 里，只会长期诱导错误推断。

### 3. `tasks_by_group` 与 `batches_by_group` 都没有独立真相价值

它们只是在把已有 truth 做按字段分桶。

这类文件最常见的长期结局不是“治理成熟”，而是：

- 写路径散落
- rebuild 依赖增强
- listing 暂时变快
- 但整体语义越来越脏

### 4. “受治理的 stale”不如“根本没有这类对象”

如果一个 projection 的唯一价值是加速非关键查询，而代价是：

- 长期 drift detection
- misuse boundary
- stale budget
- rebuild semantics
- code review 成本

那么在不计成本、只追求最终形态的前提下，最佳答案通常是删除它。

## 最终目标架构

- `tasks_by_state` = 唯一持久化 derived runtime index
- `global/tasks/*.json` = task truth
- `global/batches/*.json` = batch truth
- `machines/<machine>/state/agent.json` = agent lifecycle truth
- `machines/<machine>/claims/active/*.json` = execution ownership truth
- `summary.json` / `gpu.json` = machine-local projection

明确删除：

- `global/indexes/tasks_by_batch/`
- `global/indexes/tasks_by_machine/`
- `global/indexes/tasks_by_group/`
- `global/indexes/batches_by_group/`

## 写路径所有权矩阵

### task submit / retry / resubmit / delete / clean

必须维护：

- task truth
- batch truth（若相关）
- `tasks_by_state`

不再维护：

- batch membership index
- machine membership index
- group membership index
- batch-group membership index

### batch submit / batch repair / batch clean

必须维护：

- batch truth
- 受影响 task truth
- `tasks_by_state`（仅当 task phase 变化）

不再维护：

- `batches_by_group`

## 读取路径约束

### runtime critical readers

对象：

- scheduler
- runner-adjacent liveness readers
- orphan / stale recovery

规则：

- 只允许使用 `tasks_by_state` 做 phase candidate scan
- 正式判断回读 task / agent / claim truth

### query and listing readers

对象：

- observer
- list / show / status APIs

规则：

- batch / group / machine 结果直接从 truth 计算
- 不再依赖持久化 membership index

### summary builders

对象：

- `build_machine_workset()`
- machine summary builders

规则：

- 直接从 task truth 聚合
- 不允许再通过 `tasks_by_machine` 推断 machine responsibility

## Failure Modes And Recovery Policy

### F1. state index 漂移

影响：

- 可能把 runtime 引向错误 phase 分支

严重度：

- 高

恢复策略：

- 读时单对象纠偏
- `doctor rebuild-index` 全量重建

### F2. 历史 Tier B 索引目录残留

影响：

- 误导 operator 以为这些对象仍受正式支持

严重度：

- 中

恢复策略：

- `doctor rebuild-index` 主动删除历史目录
- 新实现不得继续写入这些目录

## 命令边界

### `doctor rebuild-index`

职责：

- 从 truth 全量重建 `tasks_by_state`
- 删除历史 Tier B index 目录

不负责：

- 修复 truth corruption
- 推断 batch / machine / group membership files，因为这些文件不再属于正式架构

### `doctor repair`

职责：

- 收敛 resubmit / clean / batch truth 等未完成 metadata operation

不负责：

- 重建已被删除出架构的 Tier B 索引族

## 正式同步原语

已接受为正式原语：

- `sync_task_state_index(cfg, task_id, actual_phase)`

不再接受为长期原语：

- `sync_task_batch_membership_index(...)`
- `sync_task_machine_membership_index(...)`
- `sync_task_group_membership_index(...)`
- `sync_batch_group_index(...)`

这些接口若曾存在，只应视为历史过渡实现，不再属于最终架构。

## 迁移阶段

### Stage 0: 文义修正

- 纠正“Tier B 长期保留”的旧结论

### Stage 1: 写路径删改

- 删除 Tier B 索引写入

### Stage 2: 读取路径 truth-only 化

- batch / group / machine 读取全部回到 truth objects

### Stage 3: rebuild 语义收口

- rebuild 只保留 `tasks_by_state`
- rebuild 清理历史 Tier B 目录

### Stage 4: 测试与文档冻结

- 所有回归与文档统一以“只有 `tasks_by_state` 为持久化 runtime index”为准

## 验收标准

只有满足以下条件，才算进入工业级最终形态：

1. `tasks_by_state` 成为唯一正式持久化 derived runtime index
2. Tier B membership index families 被从 shared-root 正式架构中删除
3. batch / group / machine 读取路径全部 truth-driven
4. `doctor rebuild-index` 只重建 state index，并清理历史遗留 Tier B 目录
5. verify / repair / runtime 行为不再把 Tier B 当作治理对象或恢复目标

## 最终建议

最稳妥的长期终局不是“把所有派生索引分级治理”，而是：

- 保留唯一真的值得持久化的 `tasks_by_state`
- 删除其余持久化 membership indexes
- 让所有非 runtime-critical 视图回归 truth computation

这比旧版 Tier B 治理方案更简洁、更强、更接近产品最终形态。

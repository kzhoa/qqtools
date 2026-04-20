# qexp batch-submit transaction semantics

状态：已实现归档

更新时间：2026-04-20

完成时间：2026-04-20

关联文档：

- [qexp Product Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_product_spec.md)
- [qexp Runtime Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md)
- [qexp shared-root scope](/mnt/c/Users/Administrator/proj/qqtools/docs/archive/pitch/002-qexp-shared-root-scope.md)
- [qexp group-to-session mapping](/mnt/c/Users/Administrator/proj/qqtools/docs/archive/pitch/001-qexp-experiment-session-grouping.md)

## 目标

本文档只讨论一件事：

为 `qexp batch-submit` 引入正式的组合提交事务语义。

本文档要回答的核心问题是：

1. `batch-submit` 是否应该被视为原子操作
2. `batch-submit` 的成功与失败应如何定义
3. 多对象写入中允许出现哪些内部中间态
4. 系统在崩溃、异常退出、部分写入后应如何自动收敛

本文档不讨论：

- 短期止血补丁
- 单任务 `submit` / `retry` / `resubmit` 的实现细节
- 训练日志体系
- artifact 管理
- 远程跨机器投递

## 背景

当前 `qexp` 已经把以下对象视为共享真相的一部分：

- `task`
- `batch`
- `machine`

其中：

- `task` 是单次执行对象
- `batch` 是“一起提交、一起观察、一起重试”的一组 task 集合

这意味着 `batch-submit` 在产品语义上并不是：

- “帮用户循环调用几次 `submit`”

而是：

- “创建一个 batch 对象，并一次性创建其全部成员 task”

一旦接受这个前提，就必须承认一个事实：

- `batch-submit` 是多真相对象组合创建操作

它天然不同于：

- 单文件原子写
- 单对象 CAS 更新

如果系统允许 `batch-submit` 在中途失败后留下半批 task，就会出现如下撕裂状态：

- 某些 task 已正式存在
- 但对应 `batch` 可能尚未存在，或只存在部分真相
- 观察面无法稳定判断这些 task 是否属于一个有效 batch
- 修复逻辑只能通过猜测残片关系来补救

这会直接破坏 `qexp` 当前强调的真相层边界。

## 问题定义

本问题的本质不是：

- “怎么让一个 for-loop 更稳”

而是：

- `qexp` 是否承认“组合提交”是一等系统能力

如果答案是否定的，那么 `batch-submit` 就只能是：

- 若干独立 `submit` 的薄封装

那样的后果是：

- `batch` 失去真相层地位
- `batch-submit` 失去“整体创建一批任务”的语义
- 用户无法合理期待“一起提交”的结果真的一起存在或一起失败

如果答案是肯定的，那么系统就必须给出正式承诺：

- `batch-submit` 对外必须呈现为原子操作
- 系统内部允许短暂中间态
- 但这些中间态必须可识别、可恢复、可收敛

因此，本提案的正式问题定义是：

- 为 `qexp` 设计一套“对外原子、对内可恢复”的 batch 组合提交事务模型

## 设计原则

### 1. `batch-submit` 必须是组合提交，不是命令糖

`batch-submit` 的语义不能退化为：

- “顺序提交 task，最后顺手写个 batch”

正式语义必须是：

- “尝试创建一个 batch 及其全部成员”

因此：

- 成功条件不能只看个别 task 是否写入成功
- 失败语义也不能接受半批对象长期残留

### 2. 用户可见结果必须原子

从用户视角看：

- 要么整批任务提交成功，并可被正式观察
- 要么本次批量提交失败，并且不应留下可被正式枚举的半批结果

这条原则是产品层心智稳定性的基础。

### 3. 内部允许中间态，但中间态必须显式建模

只要是多对象提交，就不能假装底层不存在中间态。

真正稳的做法不是“祈祷不中断”，而是：

- 承认中间态存在
- 把中间态写进真相模型
- 为中间态设计统一恢复规则

### 4. 索引必须继续是派生层

`qexp` 的正式真相仍然应该是：

- `global/tasks/<task_id>.json`
- `global/batches/<batch_id>.json`
- `machines/<machine_name>/machine.json`

因此：

- 索引不能成为事务成功条件
- 索引失败不应否定真相提交结果
- 索引收敛应由 `rebuild-index` / `repair` 负责

### 5. 恢复逻辑必须制度化，而不是事后猜测

如果系统在提交过程中崩溃、被 kill、或发生部分 I/O 失败，恢复逻辑不应依赖：

- 人工判断
- 猜哪些 task 属于哪次失败提交
- 模糊启发式

系统必须让 `doctor repair` 能直接从真相层识别未完成事务并收敛。

## 推荐结论

本提案建议正式把 `batch-submit` 定义为：

- 一次 batch-centric transaction

即：

- `batch` 是提交容器
- `task` 是该容器内的成员对象
- 事务状态由 `batch` 真相显式承载

这意味着长期正确模型不应是：

1. 先写若干 task
2. 最后补一个 batch

而应是：

1. 先创建一个带事务状态的 batch 容器
2. 再写全部成员 task
3. 最后把 batch 提升为正式 committed 状态

## 推荐真相模型

### 1. `batch` 应显式承载提交状态

建议在 `batch.json` 中加入事务状态字段：

```yaml
commit_state: preparing | committed | aborted
expected_task_count: int
task_ids:
  - ...
```

建议语义如下：

- `preparing`
  - 表示该 batch 已进入正式提交流程
  - 但整体提交尚未完成
- `committed`
  - 表示该 batch 及其成员 task 已构成正式成功真相
- `aborted`
  - 表示该 batch 提交已被明确放弃
  - 该状态主要服务于恢复和审计

**假设/未验证**：是否最终保留 `aborted` 作为持久状态，还是在 repair 后直接删除失败 batch 真相，可在实现阶段再定；但 `preparing` 和 `committed` 两态必须进入正式模型。

### 2. `batch` 是事务主锚点

系统不应通过“扫描某些孤立 task 并猜测关系”来恢复批量提交。

恢复、观察、修复都应首先信任：

- `global/batches/<batch_id>.json`

然后再据此判断：

- 该批次应有哪些成员
- 当前成员是否齐全
- 当前事务应继续提交、回滚还是标记中止

### 3. `task` 继续是正式真相对象

本提案不建议把 task 内联到 `batch.json`，原因是：

- task 仍然需要独立生命周期管理
- task 仍然是取消、日志、调度、重试、inspect 的主对象
- task 独立文件有利于延续现有真相层设计

因此长期模型仍然应保留：

- 独立 `task.json`
- 独立 `batch.json`
- 由 `batch.commit_state` 承载组合提交状态

## 两阶段提交协议

本提案建议为 `batch-submit` 引入轻量两阶段协议。

这里不是分布式数据库意义上的完整 2PC，而是：

- 在当前文件真相架构下，为多对象提交定义正式阶段边界

### 阶段 A：Prepare

这一阶段必须完成：

- 解析 manifest
- 校验 batch 元数据
- 校验全部 task 元数据
- 校验所有 `task_id` 合法且彼此不冲突
- 校验所有 `task_id` 与现有真相不冲突
- 生成完整 task 快照

这一阶段失败时，必须满足：

- 不写入任何真相对象

### 阶段 B：Commit

这一阶段建议采用以下顺序：

1. 写入 `batch.json(commit_state=preparing)`
2. 写入全部 `task.json`
3. CAS 更新 `batch.json(commit_state=committed)`
4. 更新索引与派生视图

这样设计的原因是：

- 一旦进入写阶段，系统就总能通过 `batch` 锚点识别这是一笔未完成或已完成的组合提交
- 不会再出现“task 已经存在，但系统不知道这些 task 是否属于一笔合法 batch 提交”的真相撕裂

### 失败语义

若失败发生在不同阶段，建议语义如下：

- Prepare 阶段失败：
  - 对外完全无副作用
- Commit 阶段、`committed` 之前失败：
  - 允许留下 `preparing` 中间态
  - 但该中间态不得被普通用户视图当作正式成功 batch
- `committed` 之后、索引更新失败：
  - 事务仍视为成功
  - 后续仅需收敛派生层

## 用户可见语义

### 1. 什么叫 batch-submit 成功

正式成功定义建议写死为：

- 存在 `batch.json`
- 且 `batch.commit_state == committed`
- 且 `batch.task_ids` 对应的 task 真相齐全

只有满足这三个条件，普通用户命令才应把该 batch 视为正式存在。

### 2. 什么叫 batch-submit 失败

正式失败定义建议写死为：

- 不存在 committed batch 真相

此时即使系统内部暂存了 `preparing` 批次，也不得把它当作：

- 成功 batch
- 可正常观察的已提交任务组

### 3. `preparing` 不是用户态对象

`preparing` 只能是：

- 内部恢复态
- 运维诊断态

不能是：

- 普通 `list`
- 普通 `top`
- 普通 `batch inspect`

默认展示的正式对象。

**假设/未验证**：是否保留一个 `doctor inspect-transaction` 或高级调试入口来查看 `preparing` 批次，可在后续实现时决定。

## 索引层规则

本提案建议进一步把以下原则写死：

- 索引不是 source of truth
- 索引不是事务成功条件
- 索引只负责加速观察，不负责定义提交真假

因此：

- `batch-submit` 不应因为索引尾部写失败而回滚已 committed 的真相
- `doctor rebuild-index` 应能够从 task / batch 真相完全重建派生索引

这条规则的长期收益在于：

- 事务边界被限制在真相层
- 派生层故障不会污染提交语义
- 恢复逻辑更清晰

## doctor / repair 的事务收敛职责

长期方案不能只定义 happy path，还必须定义崩溃恢复。

建议 `doctor repair` 明确承担以下职责：

### 1. 识别未完成事务

当发现：

- `batch.commit_state == preparing`

则应把该 batch 视为未完成组合提交，而不是普通 batch。

### 2. 校验事务完整性

repair 应检查：

- `expected_task_count` 是否满足
- `task_ids` 对应 task 是否齐全
- task 元数据是否与 batch 所声明范围一致

### 3. 决定收敛方向

建议规则如下：

- 如果 task 集合齐全且只差 batch 提交收口：
  - 可补做 commit
- 如果 task 集合不齐全，或事务真相已损坏：
  - 应执行 abort / cleanup

### 4. 修复派生层

若 batch 已 committed，但索引不一致，则 repair 只需要：

- 重建索引
- 刷新摘要

不应回滚真相提交。

## 与其他命令的关系

### 1. 与 `submit`

本提案不要求单任务 `submit` 立即引入同等级事务外壳。

原因：

- 单任务提交本身只有一个主真相对象
- 它不面临 batch 级组合提交的多对象一致性问题

### 2. 与 `retry`

本提案不直接讨论 `retry` 的当前实现，但长期上：

- 若未来引入 batch 级 retry 操作
- 则应优先复用同一事务模型

### 3. 与 `resubmit`

本提案不直接讨论 `resubmit` 的落地细节，但长期上：

- 若 `resubmit` 未来支持 batch 级语义
- 则同样应复用“对外原子、对内可恢复”的组合提交流程

## 非目标

本提案明确不追求：

- 跨机器远程分布式事务
- 跨 shared-root 的统一提交
- GPU 级资源预占事务
- 把训练进程生命周期也纳入同一个提交事务

本提案只解决：

- shared-root 内 batch 真相与 task 真相的一致性提交问题

## 验收标准

若本提案被采纳，长期实现完成后，系统应满足：

- `batch-submit` 不会再留下长期可见的半批正式结果
- `batch` 真相显式承载事务状态
- `preparing` 批次不会被普通用户视图误认成成功 batch
- `committed` 才是正式成功定义
- 索引失败不会否定已 committed 真相
- `doctor repair` 能识别并收敛未完成 batch 提交
- 修复逻辑不再依赖人工猜测某些孤立 task 是否属于同一次失败 batch-submit

## 最终建议

本提案建议 `qexp` 明确接受以下长期判断：

- `batch-submit` 不是一组独立 `submit`
- `batch` 是组合提交容器，而不是事后补写的附属记录
- 多对象提交必须拥有正式事务语义
- 真相层应允许中间态，但中间态必须显式建模并可恢复

只有在这个前提下，`qexp` 才能把 batch 从“方便用户的批量入口”升级为：

- 一个有严格一致性边界的正式系统对象

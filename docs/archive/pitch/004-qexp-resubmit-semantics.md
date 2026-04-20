# qexp resubmit semantics

状态：草稿

更新时间：2026-04-20

关联文档：

- [qexp Product Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_product_spec.md)
- [qexp Runtime Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md)
- [qexp batch-submit transaction semantics](/mnt/c/Users/Administrator/proj/qqtools/docs/archive/pitch/003-qexp-batch-submit-transaction-semantics.md)

## 背景与目标

`docs/spec` 已经把 `qexp resubmit` 定义为正式产品语义：

- 它不是 `submit` 的隐式覆盖
- 它不是 `retry`
- 它也不是公开 CLI 层面的 `clean + submit`

当前缺口是：

- 这一语义还没有实现
- 还没有一份可直接指导实现的 `docs/pitch`

本提案的目标只有一个：

- 为 `qexp resubmit` 给出一套工业级、可恢复、可验证的正式落地方案

本提案必须回答：

1. `resubmit` 到底替换什么，不替换什么
2. 哪些状态允许执行 `resubmit`
3. 真相层如何保证同一时刻只有一条正式 `task_id` 记录
4. 中途失败、崩溃、部分删除后，系统如何稳定收敛

## 非目标与范围

本提案不讨论：

- `submit` 的新功能扩展
- `retry` 的语义修改
- batch 成员 task 的 `resubmit`
- 训练日志内容、artifact 管理、远程投递
- `batch-submit` 事务本身的实现细节

本提案只覆盖：

- single-task `resubmit`
- CLI 入口语义
- task 真相删除与重建协议
- 索引、事件、runtime log 的收敛规则

## 场景

### 场景 1：用户想保留原 `task_id`，但不想保留失败黑历史

用户已经有一个稳定的 `task_id`，例如某个实验固定名字。第一次运行失败后，用户希望：

- 继续使用同一个 `task_id`
- 让外部观察面把它视为“这条 task 被重新正式提交了”
- 不再看到此前那条失败正式记录

此时应该使用：

```bash
qexp resubmit task_xxx -- python train.py --config configs/a.yaml
```

而不是：

- `retry`
- `submit` 复用旧 `task_id`
- 手工 `clean` 后再 `submit`

### 场景 2：用户只想再跑一次，但必须保留历史链路

这不是 `resubmit` 的场景，而是：

- `retry`

因为用户要的是：

- 保留旧 task 正式记录
- 创建新的 `task_id`
- 在 lineage 中留下可见关系

### 场景 3：用户试图对运行中任务原位重提

这必须失败。因为 `resubmit` 的本质是：

- 用一条新的首次提交真相，替换一条已经终态的正式记录

运行中 task 并不满足“可被替换的封闭历史”这一前提。

## 问题定义

`resubmit` 的本质不是“重跑命令”，而是：

- 在同一个 `task_id` 上执行一次受限的正式真相替换

因此真正需要解决的问题是：

- 如何让旧 task 被完整移出正式真相
- 如何在替换过程中不暴露长期撕裂状态
- 如何让失败恢复不依赖人工判断“删到哪一步了”

如果这件事只被实现成：

1. CLI 调 `clean`
2. CLI 再调 `submit`

那就会出现三个问题：

- 中途失败时，系统无法知道这是一次未完成的 `resubmit`
- 恢复逻辑只能猜测用户意图
- CLI 公开语义和真相层一致性边界发生分裂

所以 `resubmit` 必须是一个独立的一等操作。

## 设计原则

### 1. `resubmit` 必须是显式替换，不是命令糖

系统必须把 `resubmit` 识别为独立动作，并由专门流程负责：

- 前置校验
- 旧真相删除
- 新真相创建
- 中断恢复

### 2. 用户可见结果必须单态

从用户视角看，`resubmit` 完成后只能出现以下两种稳定结果之一：

- 旧 task 仍存在，且本次 `resubmit` 失败
- 新 task 已正式存在，且旧 task 不再可见

不允许长期存在：

- 旧 task 和新 task 同时可见
- 同一 `task_id` 没有正式 task，但系统却声称成功

### 3. 真相层必须先于索引层

正式真相仍然只能以 task 真相文件为准。

因此：

- 索引修正失败不能否定已经成功完成的 `resubmit`
- 但 `resubmit` 主流程必须负责把真相层切换到正确状态
- 派生索引可由 repair / rebuild-index 继续收敛

### 4. 删除与新建之间的空窗必须显式建模

`resubmit` 天然存在一个危险区间：

- 旧 task 已删除
- 新 task 尚未正式写入

工业级方案不能假装这个窗口不存在，而必须：

- 把该中间态写入显式操作记录
- 让恢复流程可识别、可继续、可回滚到稳定状态

### 5. 新 task 必须表现为一次新的首次提交

`resubmit` 成功后的新 task 必须是：

- 同一个 `task_id`
- 一次新的首次提交

因此旧 task 的以下字段都不得透传：

- `lineage.retry_of`
- 终态 `status`
- `result.*`
- 终态失败原因
- 运行时产物路径中的旧 run 身份

## 推荐方案

本提案建议把 `resubmit` 实现为：

- 一次 task-centric replace transaction

也就是：

1. 先创建一条 `resubmit` 操作记录，标记该 `task_id` 正在被替换
2. 再执行受控的旧真相删除
3. 再用同一 `task_id` 创建新的 task 真相
4. 最后把操作记录标记为 `committed`

这不是数据库级事务，但在当前文件真相架构下，已经足以提供：

- 清晰的阶段边界
- 稳定的恢复依据
- 对外可解释的一致性模型

## 推荐真相模型

### 1. 新增 `resubmit` 操作记录

建议在共享真相目录中新增一类操作文件：

```text
global/operations/resubmit/<task_id>.json
```

建议字段：

```yaml
operation_type: resubmit
task_id: task_xxx
state: preparing | deleting_old | creating_new | committed | aborted
old_task_snapshot_path: global/tasks/task_xxx.json
new_submission:
  command: [...]
  name: ...
  group: ...
  machine_name: ...
created_at: ...
updated_at: ...
```

设计意图：

- `task` 仍然是最终正式真相
- `operation` 只负责承载进行中的替换事务状态
- repair 通过它识别未完成的 `resubmit`

**假设/未验证**：当前仓库尚未存在通用 `global/operations/` 目录；若实现阶段已有更统一的事务记录容器，也可复用该容器，但“显式持久化操作状态”这一原则不能省略。

### 2. `task.json` 不新增长期事务态

不建议把 `task.status = resubmitting` 暴露为正式 task 生命周期状态。

原因：

- `resubmit` 的对象是一条旧 task 记录，而不是该 task 在运行中的自然状态
- 若把事务态混入 task 生命周期，会污染观察模型
- 一旦旧 task 被删除，这个中间状态本身也失去长期承载对象

因此更好的边界是：

- task 真相只表达正式 task
- operation 真相表达替换中的过程态

### 3. 可选锁文件只做并发保护，不做真相

建议继续允许实现层使用锁来保护并发，例如：

- task 级 replace lock

但锁只能做互斥，不得承担恢复语义。恢复必须依赖持久化 operation 记录，而不是依赖锁还在不在。

## 状态机与执行协议

### 1. 前置校验

进入 `resubmit` 前，系统必须完成以下检查：

1. `task_id` 对应旧 task 真相存在
2. 旧 task 状态属于允许集合：`failed` 或 `cancelled`
3. 旧 task 不属于 batch 成员
4. 当前没有其他针对同一 `task_id` 的未完成 replace / clean / retry 冲突操作
5. 新提交参数可被解析为一条合法的 single-task submit 请求

任一失败都必须直接报错，且不产生部分删除。

### 2. 阶段 A：Prepare

系统创建 `operation` 记录，状态为：

- `preparing`

同时持久化：

- 目标 `task_id`
- 新提交快照
- 旧 task 关键元数据摘要

此阶段完成后，即使进程崩溃，repair 也能知道：

- 这是一次未完成的 `resubmit`
- 它针对哪个 `task_id`
- 它原本想创建怎样的新 task

### 3. 阶段 B：Delete Old

系统把操作状态推进到：

- `deleting_old`

然后受控删除旧 task 的正式可见真相，至少包括：

- `global/tasks/<task_id>.json`
- 相关任务索引项
- best-effort runtime log 引用

如果旧 task 仍残留任何正式 task 真相文件，则不得进入下一阶段。

### 4. 阶段 C：Create New

系统把操作状态推进到：

- `creating_new`

然后按 `submit` 的正式创建规则写入新的 `global/tasks/<task_id>.json`。

新 task 必须满足：

- `task_id` 与旧 task 相同
- `lineage.retry_of = null`
- 生命周期从首次提交开始
- 创建时间、调度信息、结果字段全部重新生成

### 5. 阶段 D：Commit

当新 task 真相文件已经正式写入后，系统把操作记录标记为：

- `committed`

随后可以：

- 删除操作记录
- 或保留短期审计窗口后由清理流程回收

**假设/未验证**：操作记录在 `committed` 后立即删除还是保留审计期，取决于后续是否要建设统一操作审计面；本提案不强制二选一，但要求 `committed` 语义明确。

## 用户可见语义

### 1. 成功语义

只有满足以下条件时，`resubmit` 才能对外返回成功：

- 新 task 真相已存在
- 同 `task_id` 的旧正式 task 已不可见
- 主流程内的必要真相切换已完成

### 2. 失败语义

以下任一情况都应视为失败：

- 前置校验失败
- 旧 task 删除未完成
- 新 task 创建失败
- 流程中断，尚未收敛到稳定单态

若主流程失败但存在未完成 operation 记录，则错误信息必须明确提示：

- 当前存在未完成 `resubmit`
- 可通过 `qexp doctor repair` 继续收敛

### 3. 观察语义

建议：

- `qexp inspect <task_id>` 在发现未完成 operation 且正式 task 不存在时，返回“该 task 正处于未完成 resubmit 收敛中”的可解释提示
- `qexp list` 不应把 operation 记录当作正式 task 枚举

这样用户能理解：

- 正式 task 真相和恢复中间态是两层对象

## 并发与冲突控制

同一 `task_id` 上，以下操作必须互斥：

- `resubmit`
- `clean --task-id`
- `retry`
- `cancel`

推荐策略：

- 在进入 `resubmit` 主流程前获取 task 级互斥锁
- 锁释放前不得允许其他写操作介入

同时必须保证：

- 锁丢失不会导致恢复信息丢失
- 进程死亡后 repair 仍能仅凭 operation 记录推进收敛

## 恢复与 repair 责任

`qexp doctor repair` 应明确承担 `resubmit` 收敛职责。

### 1. 发现未完成 `resubmit`

repair 扫描：

- `global/operations/resubmit/*.json`

若发现状态不是 `committed` / `aborted` 的记录，则进入收敛逻辑。

### 2. 收敛规则

对于每条未完成 operation，repair 必须基于“旧 task 是否存在、新 task 是否存在、operation 当前阶段”进行判定。

推荐规则：

- 若状态为 `preparing`，且旧 task 仍存在
  - 说明尚未开始正式替换
  - 可以安全重试删除阶段
- 若状态为 `deleting_old`，且旧 task 仍存在
  - 继续完成删除
- 若状态为 `deleting_old`，且旧 task 已不存在
  - 进入创建新 task 阶段
- 若状态为 `creating_new`，且新 task 不存在
  - 继续创建新 task
- 若状态为 `creating_new`，且新 task 已存在
  - 直接推进为 `committed`

### 3. 不允许的异常态

repair 若发现以下情况，必须报高优先级错误，不得静默猜测：

- 同一 `task_id` 出现两条正式 task 真相
- 旧 task 内容和新 task 内容同时长期可见
- operation 记录缺失关键新提交快照，无法重建新 task

此类情况说明：

- 真相层已经超出提案假设边界

## 与其他命令的边界

### 1. 与 `submit`

`submit` 仍然只负责：

- 创建全新 `task_id`

它不得因为用户显式传入已存在的 `task_id` 就退化为 `resubmit`。

### 2. 与 `retry`

`retry` 仍然只负责：

- 保留旧 task
- 创建新 `task_id`
- 写入 lineage

它不得复用旧 `task_id`。

### 3. 与 `clean`

`clean --task-id` 仍然只负责删除，不负责重提。

`resubmit` 可以复用其底层删除子流程，但必须：

- 自己掌控完整状态机
- 自己写操作记录
- 自己负责恢复语义

### 4. 与 batch

本期继续保持：

- batch 成员 task 默认禁止 `resubmit`

原因不是 CLI 限制，而是真相修正规则尚未定义完全。若未来放开，必须同步设计：

- batch 成员引用修正
- batch 摘要重算
- batch 观察语义

## 非功能性要求

### 1. 一致性

- 任一时刻最多只允许一条正式 `task_id` 真相
- 不允许 `resubmit` 完成后残留旧终态字段污染新 task

### 2. 幂等恢复

- 对同一条未完成 operation，多次执行 repair 的最终结果必须一致
- repair 不得因为重复运行而再创建额外 task 或额外 lineage

### 3. 可观察性

- CLI 错误信息要能区分“普通失败”和“存在待收敛事务”
- 关键阶段应写调度事件，便于排查 replace 过程

### 4. 性能

- `resubmit` 的额外成本应限制在一次 operation 文件写入和少量状态推进写入
- 不应引入全局扫描依赖作为主流程前提

## 验收标准

1. 对 `failed` task 执行 `resubmit` 成功后，`inspect <task_id>` 只能看到新 task，不再看到旧终态内容。
2. 对 `cancelled` task 执行 `resubmit` 成功后，新 task 的 `lineage.retry_of` 为空。
3. 对 `running` task 执行 `resubmit` 必须明确失败，且旧 task 不受影响。
4. 对 batch 成员 task 执行 `resubmit` 必须明确失败，且不产生部分删除。
5. 在“旧 task 已删、新 task 未写”阶段强制中断后，`doctor repair` 能把该 `task_id` 收敛为单一稳定结果。
6. 任意恢复流程不得产生第二条同 `task_id` 的正式 task 真相。
7. 索引修正失败时，真相层仍保持单一正式结果，后续可由 repair / rebuild-index 收敛。

## 实施方案

### Phase 1：落地操作真相与状态机
- [ ] 在 `src/qqtools/` 中为 `resubmit` 新增显式 operation 持久化结构与读写封装。
- [ ] 为 single-task `resubmit` 建立状态机常量、序列化结构和阶段推进接口。
- [ ] 为同一 `task_id` 的 replace 冲突建立互斥保护与统一错误码。

### Phase 2：实现受控删除与同 ID 重建
- [ ] 抽离 single-task clean 的底层删除子流程，确保可被 `resubmit` 复用而不暴露为松散命令拼接。
- [ ] 实现 `resubmit` 主流程：前置校验、operation 创建、旧真相删除、新 task 创建、commit。
- [ ] 确保新 task 创建严格复用 `submit` 的合法字段生成逻辑，但屏蔽旧终态字段透传。

### Phase 3：接入观察与恢复
- [ ] 为 `doctor repair` 增加未完成 `resubmit` 识别与收敛逻辑。
- [ ] 为 `inspect` / 相关错误提示补充未完成 `resubmit` 的可解释输出。
- [ ] 为关键阶段补调度事件，便于审计和故障排查。

### Phase 4：补齐验证与文档同步
- [ ] 在 `tests/` 中覆盖正常流、非法状态、强制中断恢复、索引收敛等场景。
- [ ] 实现完成后回写 `docs/spec` 状态说明，去掉“已定义未实现”的文档漂移。

## 验证用例

| ID | 测试场景 | 输入/操作 | 预期结果 |
| :--- | :--- | :--- | :--- |
| TC-01 | 正常替换 failed task | 对终态 `failed` task 执行 `qexp resubmit <task_id> -- ...` | 命令成功；同 `task_id` 只剩新 task；新 task 无旧终态字段 |
| TC-02 | 正常替换 cancelled task | 对终态 `cancelled` task 执行 `resubmit` | 命令成功；`lineage.retry_of = null`；旧 task 不可见 |
| TC-03 | 非法状态拦截 | 对 `running` task 执行 `resubmit` | 命令失败；旧 task 原样保留；无 operation 残留或残留可安全回收 |
| TC-04 | batch 成员保护 | 对带 `batch_id` 的 task 执行 `resubmit` | 命令失败；batch 真相与 task 真相均不变 |
| TC-05 | 删除后中断恢复 | 在 `deleting_old` 后、`creating_new` 前强制中断 | `doctor repair` 可继续创建新 task 并收敛到单一稳定结果 |
| TC-06 | 创建后未提交恢复 | 新 task 已写入但 operation 尚未 `committed` 时中断 | `doctor repair` 识别新 task 已存在并将 operation 推进为完成 |
| TC-07 | 索引派生失败 | 真相切换成功后人为制造索引缺失 | 正式 task 真相保持正确；后续 repair / rebuild-index 可补齐索引 |

## 验收清单

- [ ] `resubmit` 被实现为独立状态机，而不是 CLI 级 `clean + submit`
- [ ] 同一 `task_id` 的正式真相替换具备可恢复的持久化 operation 记录
- [ ] 恢复流程不依赖人工猜测用户意图
- [ ] 用户可见语义与 `docs/spec` 当前定义完全一致
- [ ] batch 成员默认禁用 `resubmit` 的限制在实现中被强制执行

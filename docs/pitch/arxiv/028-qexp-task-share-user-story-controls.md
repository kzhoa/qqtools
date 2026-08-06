---
doc_type: pitch
status: archived
updated_at: 2026-08-06
archived_at: 2026-08-06
---

# qexp Task 分享与收回控制

## 背景

qexp 已将 placement 拆分为用户 policy、运行时 queue scope 和当前 active claim。这个拆分
对协议正确性必要，但用户不应为了表达“让其他机器帮忙”而理解并手动组合
`sharing_mode`、`fallback_constraint`、`offer_after_seconds` 和 `queue_scope`。

Runtime Spec 已允许 queued、unclaimed Task 由用户显式执行 placement mutation，但当前
CLI 只提供 `qexp task offer`。它要求 Task 提交时已经是 spillover，无法覆盖一个常见故事：

> 我先把任务留给本机；等了一段时间后，决定让 Group 里的其他机器帮忙。

本需求提供故事化控制命令，将用户意图原子转换成内部 policy 和 runtime 状态。

## 用户故事与命令

### 让其他机器现在帮忙

```bash
qexp task share <task-id>
```

用户语义：在 home machine 仍可领取的前提下，也允许 Group 内符合条件的
其他机器执行，并立即让它们可领取。`share` 是扩大候选机器集合，不是把 Task
从 home machine 移交出去。

原子效果：

- sharing policy 变为 spillover；
- fallback 默认为当前 Group Worker Set；
- queue scope 变为 shared；
- home machine 仍保留 claim 资格，不进入 fallback helper 过滤；
- 记录用户授权、操作时间和执行机器；
- 不创建 claim，也不指定最终执行机器。

### 先留给本机，稍后再让其他机器帮忙

```bash
qexp task share <task-id> --after 10m
```

用户语义：从本次命令成功开始，再给 home machine 十分钟独占候选时间；届时仍未
被领取，将符合条件的 Group helpers 加入候选集合，home machine 仍保留资格。

内部保持 home queue，并记录规范化 deadline、创建者 clock bound 和 observation 引用。
`--after` 接受明确的 `s`、`m`、`h` 单位，禁止无单位数字和负值。

### 只允许指定机器帮忙

```bash
qexp task share <task-id> --with g2 --with g3
```

用户语义：除 home machine 外，只允许列出的 Group workers 帮忙。`--with` 可重复；省略时
表示 Group 中所有符合状态要求的 workers。`--with` 只限制 helper 集合，不排除
home machine。用户不需要理解 fallback constraint。

### 分享后的竞争语义

Task 进入 shared queue 后，home agent 与符合 Worker Set、fallback 和运行状态要求的
remote agents 可以并发竞争同一 Task。Group → Task 锁序、Task CAS 和 fencing token 保证
只有一个 active claim；`share` 不预选胜出机器。

- home agent 不因 queue scope 变为 shared 而失去资格；
- remote agent 仅在 Task 已 shared 且自身符合 Worker Set/fallback 时获得资格；
- claim 胜出后，所有其他候选机器都必须服从 active claim 和 fencing token；
- authority 模式由胜出 agent 与 Task 的 home 关系及时钟能力决定，不由
  queue scope 决定。

### 收回到本机队列

```bash
qexp task keep-local <task-id>
```

用户语义：该 Task 之后只允许由其 home machine 领取。

原子效果：

- sharing policy 变为 private；
- queue scope 重置为 home；
- 清除当前 offer 状态和自动分享 deadline；
- 不修改 Task 的 home machine。

## 状态边界

`share` 和会改变状态的 `keep-local` 只接受满足以下条件的 Task：

- projection 为 queued；
- 没有 active claim；
- 没有 cleanup、取消或其他互斥控制操作；
- submission 已 committed；
- grouped Task 的 home machine 仍是 active、非 draining 的 Group worker。

`share` 额外要求 Task 已属于一个 Group。`keep-local` 对原本就是 private/home 的 standalone
Task 是只读幂等成功，不进入 Group 锁路径。

grouped Task 的操作在 Group lock → Task lock 下重新读取并验证；standalone 的幂等
`keep-local` 只取得 Task lock。任何 concurrent claim、offer、取消、drain 或 remove 只能
有一个状态变更获胜。

对 claimed、starting、running、blocked、orphaned 或 terminal Task，命令必须拒绝，不得创建
“下一个 Attempt 才生效”的隐藏 override。用户若要停止当前 Attempt，必须使用 cancel；
placement 命令不迁移、不召回也不终止进程。

## Standalone Task

standalone Task 没有 Group Worker Set，不能安全表达“其他机器”。执行 `share` 时返回：

```text
Task '<id>' is local-only because it does not belong to a Group.
Submit the work to a Group to let other machines help.
```

本需求不提供将既有 standalone Task 原地附加到 Group 的能力。Group membership sequence、
取消 barrier 和 Worker Set authority 使该操作成为独立的迁移协议，不能隐藏在 `share`
命令中。

`keep-local` 对已经 private 的 standalone Task 可以作为幂等成功返回，也可以在 CLI 层提示
“already local-only”；不得创建 Group 或修改持久化语义。

## 重复调用与已有分享状态

- 已经使用相同 helper 范围处于 shared queue 的 Task 再执行即时 `share`，幂等成功。
- 已经是 private/home 的 Task 再执行 `keep-local`，幂等成功且不增加 revision。
- `share --after` 只接受 home queue；对已经 shared 的 Task 拒绝，并提示先执行
  `keep-local`，避免一次命令暗中收回再重新分享。
- 对 queued、unclaimed Task 重新执行 `share --with ...` 可以原子替换 helper 范围；不得
  追加形成不可见的历史并集。
- 重复 `--with`、指定 home machine 自身或指定非 active Group worker 必须返回可定位错误。

## 与无可信时钟降级的关系

- `keep-local` 不需要可信时钟。
- `share` 不依赖本机时钟计算 deadline 时，可以持久化用户授权。立即 shared 后，
  home agent 仍可领取：无可信时钟时使用 `holder_bound`，有可信时钟时使用
  `bounded_lease`。符合 Worker Set/fallback 的 remote agent 也可领取：无可信时钟时使用
  `holder_bound`，有可信时钟时使用 `bounded_lease`。
- `share --after` 需要签发可比较的 wall-clock deadline，并持久化 creator clock bound、
  provider 和 observation ID；发起机器没有合格时钟能力时拒绝，并建议用户稍后执行不带
  `--after` 的即时 share。
- `offer_due`、deadline 扫描和 elapsed offer 只由 Task home agent 执行。执行时必须取得
  当前合格 clock observation，并证明 evaluator 当前时间下界不早于 deadline 上界；能力
  缺失、证据过期、provider 冲突或 bound 缺失时保持 `queued_home`，保留 deadline 等待
  能力恢复。
- 即时 `share` 和显式 `task offer` 以用户操作作为授权，不依赖 elapsed-offer 时钟门禁。
- 一个已经以 `holder_bound` authority 运行的 Attempt 不满足 queued/unclaimed
  前置条件，因此不能被 share 或 keep-local 修改。

## `task offer` 的兼容处理

`qexp task offer <task-id>` 是现有公开命令。实现本需求后：

- `task share` 成为推荐的用户入口；
- 对已经 spillover 且 queued_home 的 Task，`task offer` 保持现有幂等行为；
- `task offer` 不得自行把 private 改为 spillover；
- `task offer` 必须复用与 `share`、elapsed offer 相同的 Group → Task 原子 transition、
  状态复验、审计和幂等路径；
- CLI 帮助和文档将 `offer` 标注为已有 policy 下的立即 offer 操作；
- 是否在未来移除 `offer` 必须经过独立的兼容性决定，本需求不删除它。

## 输出与错误语义

成功输出首先描述用户结果，而不是内部字段。例如：

```text
Task task-17 is now available to eligible workers in Group stage-c1.
Task task-18 will stay on g1 for 10m, then become available to eligible Group workers.
Task task-19 is now restricted to its home machine g1.
```

机器可消费结果仍应包含稳定字段，例如 action、task_id、group、home_machine、eligible helper
machines、effective_at 和 resulting state，但不要求用户理解 queue scope 或 fallback 的内部
名称。

错误必须给出下一步，例如：

- Task 正在运行：说明 placement 不能改变正在运行的 Attempt；
- Task 不属于 Group：说明需要提交到 Group；
- 指定机器不在 Worker Set：列出无效机器；
- Group paused：命令可以成功记录分享意图，但输出明确 Task 要等 Group resume 后才会被领取；
- `--after` 缺少可信时钟：建议即时 share 或先配置合格时钟 provider。

## 权威状态变更

`share`、`keep-local`、显式 `task offer` 和 elapsed offer 必须复用同一个 Task availability
transition，不得各自实现不同的锁序或状态检查。四种 action 的差异固定为：

| action | policy 变化 | queue scope 变化 | 额外门禁 |
| :--- | :--- | :--- | :--- |
| `share_now` | private/spillover 变为 spillover，并提交 helper 范围 | home → shared | 当前用户授权 |
| `keep_local` | 变为 private，并清除 offer policy/deadline | shared/home → home | 当前用户授权 |
| `manual_offer` | 不变；必须已经是 spillover | home → shared | 当前用户授权 |
| `elapsed_offer` | 不变；必须已经是 spillover | home → shared | creator/evaluator clock evidence 证明 deadline 已过 |

提交时必须：

1. 取得 Group → Task 锁；
2. 重新验证 submission committed、Task queued/unclaimed、Group、Worker Set、取消、cleanup
   和其他控制 barrier；
3. 按 action 重新验证既有 sharing policy、queue scope、helper 范围和时钟证据；
4. 计算完整候选 placement policy/runtime；
5. 以一次 Task revision 更新提交 policy 与 queue scope；
6. 写入包含旧值、新值、action 和 clock evidence（如适用）的审计事件；
7. 更新或删除 advisory offer-deadline index；
8. 释放共享锁后触发 agent 唤醒；agent 唤醒不是提交成功条件。

任何失败都保持原 Task 不变。索引和 agent activation 失败由 doctor 重建或重试，不能回滚
已提交的 Task truth。已经处于目标状态且 action 语义相同的调用幂等成功，不增加 revision；
concurrent claim、cancel、drain、share、manual offer 和 elapsed offer 只能有一个 transition
提交成功。

## 非目标

- 不提供运行中 Attempt 的迁移或召回。
- 不允许 agent 自动扩大 private policy。
- 不把 standalone Task 隐式转换为 singleton Group。
- 不提供自动 overload、heartbeat-based share 或容量预测。
- 不让 `share` 绕过 Group pause、Worker drain/remove 或时钟能力门禁。
- 不提供任意内部 placement 字段编辑器。

## 文档与接口影响

实现时必须同步：

- Product Spec 的 Task placement 用户故事和 CLI 命令；
- Runtime Spec 的 queued placement mutation 与 offer 协议；
- CLI 帮助、README 和人类可读输出；
- manifest 文档中“提交时配置”与本命令“排队期间改变意图”的关系；
- CHANGELOG 中新增命令和 `task offer` 的定位。

## 验收标准

- [ ] grouped private queued Task 执行 `share` 后立即可由合格 Group worker claim。
- [ ] `share` 使 Task 进入 shared queue 后，home machine 仍在候选集合中，不被 helper 过滤排除。
- [ ] home/no-clock 或符合 Worker Set/fallback 的 remote/no-clock agent 可以从 shared queue
  胜出并创建 `holder_bound` claim；具备合格时钟能力的 winner 创建 `bounded_lease` claim。
- [ ] `share --after 10m` 持久化 creator clock evidence，并仅在 home agent 以当前合格
  observation 证明 deadline 肯定已过后允许共享 claim。
- [ ] `share --with` 只允许指定且仍合格的 Group workers。
- [ ] `keep-local` 将 queued_shared Task 原子收回 home 并改为 private。
- [ ] concurrent share、keep-local、manual offer、elapsed offer、claim、cancel 和 worker
  drain 通过同一 Group → Task transition，只有一个状态变更获胜。
- [ ] claimed/running/blocked/terminal Task 的 placement 修改全部被拒绝且不改变当前 Attempt。
- [ ] standalone Task 的 share 返回面向用户的可行动错误，不创建隐式 Group。
- [ ] Group paused 时可记录用户分享意图，但 Task 在 resume 前不可被 claim。
- [ ] 无可信时钟时即时 share 和显式 `task offer` 可记录授权，`share --after` 被拒绝，
  已记录 deadline 的 Task 保持 `queued_home`，local-safe Attempt 不受影响。
- [ ] `task offer` 对既有 spillover Task 保持兼容且不能扩大 private policy。
- [ ] Task revision、审计事件和 offer-deadline index 在崩溃窗口中可幂等对账。
- [ ] CLI 单测、Task/Group 并发测试和真实多进程测试覆盖全部状态边界。

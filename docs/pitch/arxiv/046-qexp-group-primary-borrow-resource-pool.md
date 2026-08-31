---
doc_type: pitch
status: archived
updated_at: 2026-08-31
archived_at: 2026-08-31
---

# qexp Group Primary/Borrow 动态资源池

关联文档：

- [qexp_product_spec.md](../spec/qexp_product_spec.md)
- [qexp_runtime_spec.md](../spec/qexp_runtime_spec.md)
- [043-qexp-machine-agent-multi-project-scheduling.md](arxiv/043-qexp-machine-agent-multi-project-scheduling.md)

## 背景与问题

qexp 当前允许用户以 Group 为单位维护 Worker Set，并通过 Group machines 操作动态增加、
drain 和移除机器。machine agent 在单台机器上统一调度多个已注册项目，但跨项目选择采用
无权重 round-robin，Group Worker Set 只表达机器是否有资格 claim，没有表达该机器对 Group
而言是主要资源还是仅可借用的空闲资源。

典型场景包含两个项目：

- A 是当前最重要的项目，希望主要使用 g1、g2、g3；
- B 需要以 g3 作为稳定的主要资源，避免长期饿死；
- 当 A 在 g1、g2 上没有需求时，B 应能借用这些空闲 GPU；
- B 对 g1、g2 的借用量可以分别受限，也可以不设上限；
- A 后续重新产生需求时，已经启动的 B 训练不得被中断，新释放的 GPU 应优先回到主要任务；
- 用户应在提交 B Group 时一次性描述上述意图，不应再逐台机器配置跨项目调度策略。

如果 B 只把 g3 加入 Worker Set，A 完成后 g1、g2 无法看到 B 的工作，造成资源空转。如果
直接把 g1、g2 作为普通 Worker 加入 B，B 又会在 A 有积压时与 A 等权竞争。因此需要在
Group Worker Set 中增加主要与借用角色，并让 machine agent 以该角色执行本机跨项目仲裁。

## 目标

本需求必须实现以下用户可见结果：

1. Group 的每台 Worker 可以声明为 `primary` 或 `borrow`。
2. `borrow` Worker 可以省略 GPU 上限，表示该 Group 在符合借用条件时可以使用该机器的
   全部可用 qexp GPU。
3. `borrow` Worker 也可以设置正整数 GPU 上限，限制该 Group 在该机器上的并发借用卡数。
4. 用户可以在一次 `batch-submit` 中创建 Group、声明 primary/borrow 资源池并提交 Tasks；
   单个 `submit --group` 只向既有 Group 追加 Task。
5. Group 创建后仍可动态增加、修改、drain 和移除 Worker，也可动态调整角色与借用上限。
6. 所有调整只影响新的 claim；已经 `starting` 或 `running` 的训练不被终止、暂停或迁移。
7. 当本机存在 primary 需求时，machine agent 不启动新的 borrow 工作；没有 primary 需求时，
   borrow 工作可以使用空闲容量。
8. primary 大任务仅因当前空闲 GPU 不足而等待时，agent 不得继续启动会阻止 GPU 自然聚合的
   borrow 工作。
9. 已在目标 machine agent 注册的项目在 Group 提交或 Worker 变更后自动观察 shared Group truth
   并执行后续调度；日常 Task 提交不要求再次注册或远程激活 agent。
10. 实现继续使用共享文件、原子替换和现有锁协议，不引入数据库、中心调度服务或训练抢占。

## 非目标

本需求不包含：

- 中断、暂停、checkpoint 后迁移或杀死正在运行的借用任务；
- 承诺 primary Task 在提交后立即获得 GPU；
- 根据预计运行时长执行 Slurm 式 backfill；
- 为项目或 Group 保证最小 GPU 份额；
- 将借用上限解释为预留额度或独占额度；
- 引入跨项目数据库、消息中间件、RPC scheduler 或 SSH 编排；
- 第一阶段引入项目权重、Task 优先级或历史 GPU 时间计费；
- 让 Group Worker Set 绕过 Task 自身的 placement/fallback 约束。
- 自动发现从未在本机注册的项目、远程激活 agent、分发项目文件或改变 MachineRuntime registry
  的信任边界；这些能力如有需要，另立需求和 ADR。

## 核心术语

| 术语 | 含义 |
| :--- | :--- |
| primary Worker | Group 正常使用的机器；在该机器上属于 primary 调度层 |
| borrow Worker | 仅在该机器没有 primary 需求时才允许产生新 claim 的 Worker |
| borrow GPU limit | 一个 Group 在一台 borrow Worker 上可同时占用的最大 GPU 数；`null` 表示不设该上限 |
| borrow usage | 该 Group 在该机器上由 provisional、claimed、starting 和 running 工作占用的 GPU 并集/总数 |
| non-preemptive return | primary 需求出现后停止新的 borrow claim，等待既有借用任务自然完成后再把释放容量交给 primary |
| Worker Set epoch | Group Worker Set 的并发控制版本；Worker 增删、状态、角色或借用上限变化均使其递增 |

## 用户场景

### 一次提交声明资源池

项目 A 的 Group 使用 g1、g2、g3 作为主要资源。`runs-a.yaml` 显式使 Task 可在 Group 内
spill over；Worker role 不会自行改变 Task placement：

```bash
qexp batch-submit \
  --group a-sweep \
  --file runs-a.yaml
```

```yaml
group:
  workers:
    primary: [g1, g2, g3]
defaults:
  placement:
    home_machine: g1
    sharing:
      mode: spillover
      fallback_machines: group
      offer:
        after_seconds: 0
```

项目 B 以 g3 为主要资源，最多借用 g1 的 1 张卡和 g2 的 2 张卡：

```bash
qexp batch-submit \
  --group b-sweep \
  --file runs-b.yaml
```

```yaml
group:
  workers:
    primary: [g3]
    borrow:
      g1: 1
      g2: 2
defaults:
  placement:
    home_machine: g3
    sharing:
      mode: spillover
      fallback_machines: group
      offer:
        after_seconds: 0
```

项目 B 也可以不限制借用卡数：

```bash
qexp batch-submit \
  --group b-sweep \
  --file runs-b.yaml
```

`runs-b.yaml` 的 Worker Set 为 `primary: [g3]` 与 `borrow: [g1, g2]`。单个 `batch-submit`
仅接受 manifest 中的 `group.workers` 作为初始 Worker Set 输入，避免 CLI 与 manifest 出现两个
语义相同但可能冲突的来源。同一机器重复出现、同时声明为 primary 和 borrow、上限为零、负数或
非整数时必须拒绝整个 Submission，不得部分创建 Group 或 Tasks。

### Manifest 输入

无上限 borrow 可以使用机器列表：

```yaml
group:
  workers:
    primary: [g3]
    borrow: [g1, g2]
```

带上限或混合配置使用映射，值为正整数或 `null`：

```yaml
group:
  workers:
    primary: [g3]
    borrow:
      g1: 1
      g2: 2
      g4: null
```

两种 `borrow` 输入归一化为同一个内部结构：

```yaml
g1:
  scheduling_role: borrow
  borrow_limit_gpus: null
```

列表写法中的每台机器都归一化为 `borrow_limit_gpus: null`。`null` 表示没有 Group 级借用
上限，不表示忽略机器物理容量、machine-wide reservation、primary 需求或 Task 的
`requested_gpus`。

### 后续追加 Task

Group 已存在后，新增 Task 仍必须显式声明可执行 placement；`--group` 不会让 private Task 自动
借用其他 Worker：

```bash
qexp submit --group b-sweep --home-machine g3 --sharing spillover --offer-after-seconds 0 \
  -- python train.py --config b3.yaml
```

普通无 Group 提交继续保持当前 private/local 默认语义，不受本需求影响。

## 调度语义

### 本机选择顺序

每当 machine agent 形成可信的空闲 GPU 快照时：

1. 对本机已注册项目执行无副作用的 primary-demand probe，扫描 active primary Group 候选。
2. primary 候选可以立即运行时，在 primary 层内使用确定性公平调度。
3. primary 候选仅因当前空闲 GPU 不足而等待，但其请求不超过机器可管理总容量时，停止新的
   borrow admission，让运行任务自然结束并聚合 GPU。
4. 没有 primary 需求时，扫描 active borrow 候选。
5. borrow 层内继续使用确定性公平调度，并跳过达到本机 Group 借用上限的候选。
6. Worker role 只在 Task 已通过 placement/fallback 授权到达本机后参与本机排序；候选最终仍必须
   通过现有 Group、Task、placement、claim、lease、fencing 和 launch gate
   校验。

primary-demand probe 必须使用独立的 advisory probe cursor；它不得推进 ordinary dispatch ready
cursor、创建 reservation 或改变 Task 状态。probe 必须返回下列互斥结果：

- `runnable_now`：存在 active primary Worker 上已获 placement 授权、元数据可读、working
  directory 可用且请求 GPU 不超过当前空闲容量的 Task；
- `waiting_for_aggregation`：存在同样合法的 Task，请求 GPU 不超过本机可管理总容量但超过当前
  空闲容量；
- `no_primary_demand`：不存在上述 Task。被 placement 拒绝、请求超过总容量、working directory
  不可用或 ready 元数据损坏的候选必须附带诊断原因，不得阻塞 borrow。
- `unresolved`：本轮扫描预算耗尽、ready index 在扫描期间变更，或任一项目的 primary
  候选范围无法可信读取。`unresolved` 不是“没有需求”，必须阻止新的 borrow admission。

`runnable_now`、`waiting_for_aggregation` 与 `unresolved` 都阻止新的 borrow admission。probe
按项目公平推进自己的 advisory cursor；只有完整遍历候选范围，且遍历起止的 ready-index revision
一致时，才能报告 `no_primary_demand`。结构性不可运行候选在记录诊断后跳过，避免永久阻塞
borrow。probe 不得通过消费普通 dispatch 的 next-ready 操作来推断“没有 primary 需求”。

一次 borrow admission 以完成可信 probe 的容量快照为线性化观察点：该快照之后新提交或新
offer 的 primary Task 属于后续调度决策，不追溯撤销已经获得 reservation 的 borrow Attempt。

以下情况不构成阻止 borrow 的 primary 容量需求：

- Group 已 paused；
- Worker 不再 active；
- Task placement 不允许当前机器；
- Task 请求超过机器可管理 GPU 总量；
- working directory 缺失或不可访问；
- Task/ready 元数据损坏或项目调度已 fail closed。

这些情况必须产生明确诊断，不能因为一个永久或结构性不可运行的 primary 候选让整台机器
长期空闲。

### 无借用上限

`borrow_limit_gpus: null` 只移除该 Group 的借用额度门槛。实际可启动数量仍受以下条件约束：

- 当前空闲 qexp GPU；
- 本机是否存在 primary 需求；
- Task 的 `requested_gpus`；
- 其他 Group/项目的公平调度；
- machine-wide provisional/active reservation；
- Group Worker 与 Task placement 的最终授权。

因此，无上限 borrow 是 work-conserving 借用，不是机器独占。

### 有借用上限

对于 g2 上 `borrow_limit_gpus: 2` 的 B Group：

```text
borrow_usage(B, g2) + requested_gpus <= 2
```

才允许新的 borrow claim。额度按 GPU 数而非 Task 数计算。一个请求 4 张 GPU 的 Task 不能在
上限为 2 的 g2 上以 borrow 身份启动，即使机器当前有 4 张空闲 GPU；它仍可在允许且容量
足够的其他 Worker 上运行。

usage 必须覆盖尚未 attach 的有效 provisional reservation、claimed、starting 和 running
Attempt，防止连续或并发 claim 突破上限。终态、已释放和已过期 provisional reservation
不得继续计入。

### 非抢占式归还

如果 B 已借用 g1/g2，随后 A 产生 primary 需求：

1. B 的 starting/running Attempt 保持不变；
2. agent 停止在对应机器为 B 创建新的 borrow claim；
3. B Attempt 自然完成后释放 reservation；
4. 新释放的容量在下一调度决策中优先满足 primary；
5. 不发送终止信号，不暂停、迁移或自动重试 B。

因此系统只能承诺任务边界上的资源归还，不能承诺 primary 的立即启动时间。

## 动态资源池

primary/borrow 角色和借用上限必须扩展现有 Group Worker Set 动态能力，而不是冻结在 Group
创建时。

建议的 CLI 语义：

```bash
# 新增无上限 borrow Worker
qexp group machines add b-sweep g4 --role borrow

# 新增最多借用 2 张卡的 Worker
qexp group machines add b-sweep g5 --role borrow --max-gpus 2

# 修改借用上限
qexp group machines set b-sweep g2 --max-gpus 4

# 清除借用上限
qexp group machines set b-sweep g2 --max-gpus unlimited

# 动态修改角色
qexp group machines set b-sweep g1 --role primary
qexp group machines set b-sweep g3 --role borrow --max-gpus 1

# 延续现有安全退出流程
qexp group machines drain b-sweep g1
qexp group machines remove b-sweep g1

# 查询规范化角色、usage、limit 与注册观察
qexp group machines list b-sweep
```

本期公共 CLI 固定采用 `add`、`set`、`drain`、`remove` 与 `list`；不得要求创建新 Group 或
重提已运行 Task。

每次 Worker 增删、drain、角色变化或 limit 变化都必须：

- 在 Group lock 下线性化；
- 递增 `worker_set_epoch`；
- 更新对应 Worker 的 `state_epoch`；
- 影响后续 claim 和 launch authorization；
- 保持已 starting/running Attempt 的执行权，不隐式产生终止意图；
- 由已注册 machine agent 从 Group shared truth 观察耐久变更；不写入新的跨项目 assignment。

### 下调额度

若 g2 当前 borrow usage 为 2，用户将上限下调为 1：

```text
role: borrow
usage: 2
limit: 1
state: over_limit
```

修改应成功提交，而不是要求中断任务或等待 CLI 阻塞：

- 当前 Attempt 继续运行；
- usage 高于或等于新上限期间不得产生新 borrow claim；
- usage 自然下降到零后，新的单卡任务才可再次启动；
- 状态与观察命令必须显示 `over_limit`，不能把配置收敛误报为调度故障。

### 从 borrow 提升为 primary

角色提升从下一次调度决策生效。borrow GPU limit 不再参与 primary admission；内部规范化值应
变为 `null`。已经运行的 Attempt 不重启，也不更换 claim 身份。

### 从 primary 降为 borrow

角色降级必须显式给出 limit 或采用无上限默认。已经运行的 primary Attempt 继续完成，并计入
当前 Group 在该机器上的观察使用量。若当前占用超过新借用上限，状态显示 `over_limit`，并
禁止新的 borrow claim，直至自然收敛。

## 权威数据模型

Group Worker Set 建议扩展为：

```yaml
worker_set:
  g1:
    state: borrow
    scheduling_role: borrow
    borrow_limit_gpus: 1
    state_epoch: 3
    added_at: "..."
    added_by_operation: "..."
    drain_requested_at: null
    remove_requested_at: null
    terminate_running: false

  g2:
    state: borrow
    scheduling_role: borrow
    borrow_limit_gpus: null
    state_epoch: 5

  g3:
    state: active
    scheduling_role: primary
    borrow_limit_gpus: null
    state_epoch: 2
```

验证不变量：

- `state` 只能是 `active | borrow | draining | removing`；`active` 与 `borrow` 都是新 agent
  可 claim 的 lifecycle state，后两者不可产生新 claim；
- `scheduling_role` 只能是 `primary | borrow`；
- `primary` 的 `borrow_limit_gpus` 必须为 `null`；
- `borrow_limit_gpus` 只能是 `null` 或正整数；
- 同一 Group 中同一 machine name 只能出现一次；
- Task fallback constraint 不能超出 Group Worker Set；
- 角色与 limit 是 Group truth，不是 ready marker 或 machine-local cache 的最终权威；
- machine-local reservation 必须携带足以按 `(project_id, group_name, machine_name)` 统计
  usage 的身份；当前 Worker 为 borrow 时，该 Group 在本机的所有有效 reservation 都计入
  usage，包括角色变更前已获 primary admission 的 Attempt。

claim 与 Attempt 应记录 claim 时观察到的：

- `group_worker_set_epoch`；
- `worker_state_epoch`；
- `worker_scheduling_role`；
- `borrow_limit_gpus`；
- 本次 Attempt 是否以 borrow 身份 admitted。

这些字段用于审计与 launch gate 复核，不把历史快照提升为当前 Group 权威。role/limit 的后续
变化不撤销已经成功获得的 claim；drain、remove、pause、取消和 fencing 仍按现有 launch gate
规则处理。

## Agent 注册前置条件

参与某 Group 的每个目标 machine 必须存在该项目的 enabled machine-agent binding，并能访问该
项目 shared root 与 Task working directory。`qexp init` 会创建当前代 binding；只有缺失或被
移除的 binding 才使用 `qexp agent add-project` 恢复。现有 machine-local project registry 是本期
唯一的项目发现与本机资源信任边界。

binding 存在后，Group submit、Worker 增删和 role/limit 修改都只写 Group shared truth；正在运行的
agent 在后续调度循环中读取该 truth。on-demand agent 可能已因空闲退出，qexp 不远程唤醒目标
machine；此时操作者必须在目标 machine 本地启动或唤醒 agent。agent 离线不使 Submission 回滚，
恢复后按 Group、Task、reservation truth 重建状态。可观察性只报告 agent heartbeat/可读性，不引入
assignment acknowledgement 协议。

**假设/未验证**：g1、g2、g3 均存在上述 enabled binding，并具有稳定 machine identity；其 shared
root 和 working directory 均可访问。自动接入未注册项目不属于本期。

## 并发与安全约束

完整链路为：

```text
Group submit/update
  -> Group lock 下提交 Worker role/limit/epoch
  -> 已注册 machine agent 读取 primary/borrow 候选
  -> 形成可信 machine-wide reservation 快照
  -> primary demand gate
  -> Group + Task 权威复核
  -> reservation lock 下 borrow limit admission + provisional reservation
  -> Attempt claim
  -> launch gate 再次复核
  -> starting/running
  -> terminal publication
  -> reservation release
```

实现必须处理：

- 两个 Task 连续 claim 争用最后一个 borrow GPU 名额；
- limit 更新与 provisional reservation 并发；
- role 更新与 claim/launch authorization 并发；
- drain/remove 与 borrow claim 并发；
- agent 在 reservation 后、Task claim 前崩溃；
- agent 重启后从 active/provisional reservation 重建 usage；
- reservation 已释放但观察快照陈旧；
- Group shared root 暂时不可读时保持现有 fail-closed 和 reservation 隔离语义。

borrow admission 必须在 Group -> Task 权威锁顺序下重读 role、limit 与 epoch；随后调用唯一的
`reserve_admitted` reservation API。该 API 在 machine reservation lock 内按
`(project_id, group_name, machine_name)` 聚合有效 provisional/active reservation，校验
`usage + requested_gpus <= limit`，并原子创建携带身份、role 与 epoch 的 provisional reservation。
machine reservation lock 内不得执行 shared Group/Task I/O。claim 与 Attempt 写入成功后，role/limit
更新不会撤销该 claim；launch gate 只继续复核现有 pause、drain/remove、取消、lease 与 fencing
约束。该协议保证不会产生超过当时有效上限的新 admission，不能以展示层统计或无锁预检查替代。

## 可观察性

```bash
qexp group machines list b-sweep
```

应至少显示：

```text
MACHINE  ROLE     GPU_USAGE  GPU_LIMIT  STATE       AGENT
g3       primary  4          -          active      registered
g2       borrow   2          2          full        registered
g1       borrow   1          unlimited  active      registered
g4       borrow   2          1          over_limit  registered
g5       borrow   0          2          active      registered
```

状态解释：

- `full`：usage 已达到有限上限；
- `over_limit`：动态下调或角色变化后，既有非抢占任务使 usage 暂时超过上限；
- `registered`：目标 machine 已具备该项目的本机 agent 注册；该列不表示跨机器配置已推送；
- `unlimited`：没有 Group 借用上限，不表示没有其他调度约束。

结构化输出必须同时提供规范化的 `borrow_limit_gpus: int | null`，脚本不得依赖人类展示字符串
`unlimited`。

## 兼容性与迁移

- 现有 Group Worker 缺少 `scheduling_role` 时读取为 `primary`。
- 现有 Worker 缺少 `borrow_limit_gpus` 时读取为 `null`。
- 现有 manifest 的 `group.workers: [g1, g2]` 保持含义不变，全部归一化为 primary。
- 新的 `group.workers.primary`/`borrow` 是显式选择新调度语义的入口。
- 当前 `add/drain/remove` 行为继续有效；未指定 role 的 `machines add` 默认为 primary，以保持
  兼容。
- borrow Worker 使用兼容 wire encoding：`state: borrow` 与
  `scheduling_role: borrow` 同时写入。新 agent 将其作为可 claim 的 borrow Worker；旧 agent 仅将
  `state: active` 视为可 claim，因此会跳过 borrow Worker。旧 agent 最多造成借用容量暂不可用，
  不得把 borrow Worker 当作 primary Worker 调度。
- 角色、limit、epoch 与 `state: borrow` 的组合属于同一次 Group lock 写入。agent heartbeat 可以
  展示版本观察，但不是写入 borrow 配置的安全前提。

**假设/未验证**：所有已发布旧 agent 的 claim、recovery 与 doctor 路径都会将未知
`state: borrow` 视为非 claimable。实现前必须以旧 agent 二进制回归验证；任何一路若将未知 state
当作 `active`，本期改为 schema cutover，禁止采用该兼容编码。

兼容性决策：本期不改变 `qexp init -> qexp submit` 或 `qexp init -> qexp agent start` 的受保护工作流；
远程机器仍只要求既有 enabled binding 与本地 agent 启动，不增加远程唤醒或 `add-project` 前置步骤。

## 验收标准

### 输入与归一化

- [ ] `borrow: [g1, g2]` 被接受并归一化为两个 `borrow_limit_gpus: null` Worker。
- [ ] `borrow: {g1: 1, g2: 2, g3: null}` 被接受并保持逐机 limit。
- [ ] manifest 接受 unlimited、limited 和混合 borrow 声明。
- [ ] 重复 Worker、角色冲突、零/负数/非整数 limit 在写入前被拒绝。
- [ ] primary 与 borrow Worker 都进入同一动态 Group Worker Set。

### 调度行为

- [ ] A 在 g1/g2/g3 为 primary，B 在 g3 为 primary、g1/g2 为 borrow 时，A 有需求期间
  g1/g2 不产生新的 B borrow claim。
- [ ] A 无需求后，B 在 g1/g2 存在 enabled binding 且目标 agent 正在运行时自动使用 g1/g2；B 的
  Task placement 已明确允许这两台机器。
- [ ] 无上限 borrow 能使用当前所有可借空闲 GPU，但不越过 primary demand gate。
- [ ] g1 limit=1、g2 limit=2 时，B 的有效 usage 分别不因新 admission 超过 1 和 2。
- [ ] 单 Task requested_gpus 超过本机 borrow limit 时跳过该机器，不影响其他机器候选。
- [ ] primary 大任务等待 GPU 聚合时不启动会延迟它的 borrow Task。
- [ ] primary 永久不可运行候选不会让 borrow 资源永久闲置。

### 动态修改与非抢占

- [ ] 运行期间可以新增 primary/borrow Worker。
- [ ] 可以在有限 limit 与 unlimited 之间动态切换。
- [ ] 可以在 primary 与 borrow 之间动态切换。
- [ ] 下调 limit 低于当前 usage 时进入 `over_limit`，运行 Attempt 不被中断。
- [ ] primary 需求回流时，既有 borrow Attempt 正常完成，不再产生新的 borrow claim。
- [ ] drain/remove 保留现有 queued-work 与 active-attempt 安全检查。
- [ ] role/limit/worker state 竞争在 claim admission 中按 epoch 线性化；role/limit 更新不会撤销
  已成功 claim，drain/remove 仍由 launch gate 处理。

### 恢复与观察

- [ ] agent 重启从 reservation truth 恢复 borrow usage，不依赖进程内计数。
- [ ] 已注册 agent 离线恢复后按 Group、Task 与 reservation truth 恢复并参与调度。
- [ ] human 与 structured 输出区分 unlimited、full、over-limit 和 registered。
- [ ] shared-state 不确定时 fail closed，不把无法验证的 GPU 当成可借容量。

## 实施前已确定契约

- primary-demand probe 使用独立 advisory cursor；完整且 revision-stable 的遍历才可得出
  `no_primary_demand`，否则 `unresolved` 并 fail closed。
- 唯一的 `reserve_admitted` API 在 reservation lock 内完成 usage/limit 检查与 provisional
  reservation 创建；Group -> Task -> reservation 是 admission 的锁顺序。
- borrow 使用 `state: borrow` 的 fail-closed 兼容编码，不建立全 Worker capability acknowledgement
  协议。
- 公共 CLI 固定为 `machines add|set|drain|remove|list`。

## 实施方案

### Phase 1：契约、schema 与兼容门禁

- [ ] 扩展 Group Worker schema，默认旧 Worker 为 `primary`，并校验 role/limit 不变量。
- [ ] 扩展 Worker lifecycle state 与兼容读取：新 agent 识别 `state: borrow`，旧 agent 对其 fail
  closed。
- [ ] 将稳定的输入、迁移与兼容契约同步到 `docs/spec/qexp_product_spec.md` 与
  `docs/spec/qexp_runtime_spec.md`。

### Phase 2：placement 保持不变的本机调度

- [ ] 实现无副作用 primary-demand probe，明确四态结果、独立 cursor、revision-stable 完整遍历与
  `unresolved` fail-closed 行为。
- [ ] 将 machine agent 的本机项目 round-robin 改为 primary 层优先、borrow 层公平的 admission；
  不改变 Task placement、project registry 或远程 agent 生命周期。

### Phase 3：原子额度、动态变更与观察

- [ ] 为 reservation 增加 Group 身份与 admission snapshot，实现 `reserve_admitted` 的 borrow
  limit 原子 admission；role/limit 更新不得撤销已成功 claim。
- [ ] 扩展 `group machines` 的 role/limit 变更和结构化/human 观察输出，保留 drain/remove
  的现有安全语义。

### Phase 4：验证

| ID | 测试场景 | 输入/操作 | 预期结果 |
| :--- | :--- | :--- | :--- |
| TC-01 | placement 边界 | private B Task 配置 borrow Worker | B 不在该 Worker claim；显式 spillover 后才可进入 borrow 层 |
| TC-02 | primary 回流 | A 有 `waiting_for_aggregation` demand，B 有可运行 borrow Task | 不创建新的 B borrow claim |
| TC-03 | 并发额度 | 两个 Task 同时争用最后一个 borrow GPU | 至多一个获得 launch authorization |
| TC-04 | 动态下调 | usage=2 时将 limit 改为 1 | 状态为 `over_limit`，运行 Attempt 不终止，后续 borrow admission 停止 |
| TC-05 | 混合版本 | 旧 agent 读取 `state: borrow` Worker | 不产生该 Worker 的新 claim；primary Worker 行为不变 |

### 验收清单

- [ ] 核心输入、placement、调度、动态变更与恢复测试通过。
- [ ] 原子 admission 与 launch gate 的锁顺序有可复核测试证据。
- [ ] 混合版本 fail-closed 规则已验证，旧 agent 不会获得 borrow-as-primary 的机会。
- [ ] 产品与运行时规格已同步；本期未引入未注册项目自动发现。

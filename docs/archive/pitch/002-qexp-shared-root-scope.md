# qexp shared-root scope

状态：已实现归档

更新时间：2026-04-17

完成时间：2026-04-20

关联文档：

- [qexp Manual](/mnt/c/Users/Administrator/proj/qqtools/docs/manual/qexp.md)
- [qexp Product Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_product_spec.md)
- [qexp Runtime Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md)
- [qexp group-to-session mapping](/mnt/c/Users/Administrator/proj/qqtools/docs/archive/pitch/001-qexp-experiment-session-grouping.md)

## 目标

本文档只讨论一件事：

定义 `qexp shared_root` 的正确隔离粒度与使用边界。

本文档要回答的核心问题是：

1. `shared_root` 应该按什么粒度切分
2. 哪个粒度内 `qexp` 应承诺队列与 GPU 视图一致性
3. 哪个粒度外 `qexp` 不承诺互斥隔离
4. 项目级 shared root 是否会导致 metadata 无限膨胀

本文档不讨论：

- group 到 tmux session 的映射细节
- 训练日志体系
- artifact 管理
- 全机级 GPU 强互斥实现

## 背景

当前 `qexp v2` 的设计是：

- 一个 `shared_root` 承载一套 task / batch / machine / index 真相
- agent、observer、scheduler 都只看当前 `shared_root`
- `qexp use` 切换的本质是 CLI 默认上下文切换，不是全机级运行态合并

这会带来一个实际后果：

- 如果同一台机器在 `shared_root A` 下已有运行中 task
- 用户再切到 `shared_root B`
- `shared_root B` 的 agent 和调度视图不会自动感知 `shared_root A` 中的 task

因此，`shared_root` 的切分粒度如果定义得太细，就会直接决定：

- 哪些 task 能互相看见
- 哪些 task 能参与同一套 GPU 占用判断
- 哪些 task 能被 `top` / `machines` / `list` 统一观察

## 用户场景

一个典型场景如下：

- 用户有一个项目：`/mnt/share/myproject/`
- 项目下有两组实验计划：`exp1/` 和 `exp2/`
- 两组实验虽然属于不同工作上下文，但仍然共享同一项目的机器资源池和同一项目的观察面需求

用户可能会直觉地这样做：

```text
/mnt/share/myproject/.qexp/exp1/shared/
/mnt/share/myproject/.qexp/exp2/shared/
```

并分别把它们作为两套 `shared_root`。

这个做法的问题是：

- `exp1` 和 `exp2` 会变成两个彼此隔离的控制平面
- 同一台机器在 `exp1` 中已占用的 GPU，不会自然出现在 `exp2` 的保留视图里
- 项目级 `top` / `machines` / queue 观察会被打碎成两套
- 用户需要自己记住“现在在哪个 root 下看到了哪部分真相”

而用户真正需要的通常是：

- 项目内所有 task 共用一套控制真相
- 项目内不同实验计划通过 `group`、`name`、`batch` 等工具层字段区分
- 同一项目内的资源占用、队列观察、重试与清理在一个 root 下闭环

因此，更合理的目录形态应是：

```text
/mnt/share/myproject/.qexp/
```

然后：

- `exp1`、`exp2` 作为该 root 下的 task 归组语义存在
- 而不是各自拆成独立 `shared_root`

## 问题定义

本问题的本质不是“metadata 放哪更顺手”，而是：

- `shared_root` 到底代表一套什么边界内的系统真相

如果边界定义错误，会同时破坏三件事：

1. 资源隔离语义
2. 观察语义
3. 运维语义

因此，`shared_root` 必须被定义成：

- 一个完整控制平面的边界

而不是：

- 随手给某个实验计划单独开个元数据目录

## 隔离粒度定义

本提案建议正式定义 5 层粒度：

### 1. host 级

表示同一台物理机或容器宿主上的全部 GPU 资源。

本提案结论：

- `qexp` 当前版本不承诺 host 级全局隔离

也就是说：

- 如果两套互不关联的 `shared_root` 恰好跑在同一台机器上
- `qexp` 不保证它们彼此感知，也不保证全局 GPU 强互斥

### 2. project 级

表示同一个项目的统一控制平面。

例如：

```text
/mnt/share/myproject/.qexp
```

本提案结论：

- `shared_root` 的正式推荐粒度应是 project 级

在这个粒度内，`qexp` 应承诺：

- task 真相统一
- batch 真相统一
- machine 视图统一
- 同项目内的 GPU 占用判断统一
- 同项目内的观察与清理入口统一

### 3. group 级

表示同一项目下的一组相关 task run。

例如：

- `exp1`
- `exp2`
- `contract_n_4and6`

本提案结论：

- `group` 是 project 内的逻辑归组键
- `group` 不应升级为独立 `shared_root`

### 4. batch 级

表示“一起提交、一起观察、一起重试”的一组 task。

本提案结论：

- batch 也不应升级为独立 `shared_root`
- batch 是 project root 内的操作级集合，不是控制平面边界
- batch 也不应替代 `group`

补充定义：

- `group` 是 project 内长期归组键
- `batch` 是一次批量提交形成的操作集合
- 一个 `group` 内允许存在多个 `batch`
- 同一个 `batch` 默认应归属于一个 `group`

### 5. task 级

表示单次执行对象。

本提案结论：

- task 永远是项目控制平面内部的基本对象
- 不能拿 task 粒度去切 `shared_root`

## 推荐结论

正式推荐如下：

### 1. `shared_root` 应按 project 级切分

推荐：

```text
/mnt/share/myproject/.qexp
```

不推荐：

```text
/mnt/share/myproject/.qexp/exp1/shared
/mnt/share/myproject/.qexp/exp2/shared
```

原因：

1. per-experiment shared root 会把同项目内的资源与观察真相打散
2. 同项目不同实验计划之间无法参与同一套 GPU 占用判断
3. `top` / `machines` / `list` 只能看到 root 内局部事实，无法提供项目总览
4. 用户会频繁陷入“当前命令看到的是哪套宇宙”的认知负担

### 2. `group` 才是实验计划的正确承载层

若用户脑中有：

- `exp1`
- `exp2`

这样的实验计划，应转化为：

- 同一 project shared root 下的不同 `group`

而不是：

- 不同 `shared_root`

同时也不是：

- 把每个实验计划硬映射成一个独立 `batch`

原因：

- `batch` 更接近“一次提交动作”
- `group` 更接近“一个持续存在的工作上下文”
- 用户完全可能在同一个实验计划下分多轮 batch-submit

### 3. 当前版本只承诺 project 内隔离，不承诺 host 全局隔离

也就是说，本提案明确接受：

- 同一台机器上，不同项目之间若用了不同 `shared_root`，`qexp` 之间可以彼此不感知

但至少要承诺：

- 同一项目内部，不同实验计划不应因为切 root 而丢失资源占用感知

### 4. project root 内 metadata 必须按对象类型维护，而不是按 experiment 目录维护

`shared_root` 下应维护的是一套统一对象真相，而不是“每个 experiment 一棵子树”。

推荐的 metadata 类别只有这些：

- task 真相
- batch 真相
- machine 真相
- 全局索引
- 锁
- 调度与审计事件
- machine 侧状态镜像与 claim

换句话说，metadata 的组织维度应是：

- 按对象类型

而不是：

- 按 experiment / group 目录

## Metadata 契约

为了避免后续继续把 `shared_root` 用成“随手分目录的收纳盒”，这里需要进一步约定 metadata 到底维护什么。

### 1. task 级 metadata

每个 task 至少应维护：

- `task_id`
- `name`
- `group`
- `batch_id`
- `machine_name`
- 执行命令与资源请求
- 当前 phase / reason
- runtime 运行态字段
- timestamps
- lineage
- revision / updated_by_machine 这类写入控制字段

这部分是真相层对象。

正确承载方式是：

- `global/tasks/<task_id>.json`

而不是：

- `groups/<group>/tasks/<task_id>.json`
- `exp1/tasks/<task_id>.json`

### 2. batch 级 metadata

每个 batch 至少应维护：

- `batch_id`
- `name`
- `group` 或默认 group 语义
- `machine_name`
- `task_ids`
- summary
- policy
- revision / updated_by_machine

这部分也是真相层对象。

正确承载方式是：

- `global/batches/<batch_id>.json`

### 3. machine 级 metadata

每个 machine 至少应维护：

- `machine.json`
- `state/agent.json`
- `state/gpu.json`
- `state/summary.json`
- `claims/*`
- `events/*`

这部分是 machine 私有共享侧辅助视图。

正确承载方式是：

- `machines/<machine_name>/...`

### 4. index 级 metadata

索引层只应维护派生视图，例如：

- `tasks_by_state`
- `tasks_by_batch`
- `tasks_by_machine`

如果后续增加 `tasks_by_group`、`batches_by_group` 这类索引，也仍然应属于：

- 全局派生索引

而不是：

- 某个 group 自己的一套真相目录

### 5. event / audit 级 metadata

事件层只应维护审计记录：

- task lifecycle events
- agent lifecycle events
- machine-side audit events

它们用于追踪和排障，不应承担 group 真相。

## 不同 exp / group 是否需要独立目录

本提案的明确结论是：

- 不需要
- 不应该默认拥有

更严格地说：

- 不同 exp / group 不应在 `shared_root` 下拥有自己的独立真相目录

不推荐的形态包括：

```text
<shared-root>/groups/exp1/tasks/
<shared-root>/groups/exp1/batches/
<shared-root>/groups/exp2/tasks/
<shared-root>/groups/exp2/batches/
```

或者：

```text
<shared-root>/exp1/
<shared-root>/exp2/
```

原因如下：

### 1. 这会制造第二套真相组织维度

一旦 group 拥有独立目录，系统就会开始面临两个问题：

- 真相究竟以 `global/tasks/*.json` 为准
- 还是以 `groups/<group>/...` 为准

工业级系统不能接受这种双路径真相。

### 2. 这会让 clean / retry / repair 复杂度爆炸

如果 task 与 batch 既存在全局真相，又被 group 子目录组织一次，那么：

- clean 需要双写或双删
- retry 需要双处登记
- repair 需要决定先修哪边

这会直接破坏“task 真相优先，batch 真相次之，索引可重建”的当前收敛原则。

### 3. 这会让用户误以为 group 是控制平面边界

一旦看见目录：

```text
groups/exp1/
groups/exp2/
```

用户自然会理解成：

- exp1 和 exp2 各自有一套局部控制面

这与本提案要强调的 project-root 单真相边界正面冲突。

## 是否应该允许 group 拥有任何目录

默认不应允许 group 拥有独立真相目录。

如果未来确有需要，最多只应允许两类“纯派生、可重建”的 group 级目录：

1. 展示缓存
2. 导出产物

例如：

```text
<shared-root>/derived/groups/<group>/summary.json
<shared-root>/exports/groups/<group>/manifest.snapshot.json
```

但即便如此，也必须满足三个硬约束：

- 这些目录不是 source of truth
- 删除后必须可完全重建
- 不得成为 submit / retry / clean / repair 的必经依赖

因此，结论仍然是：

- 可以在未来谨慎允许 group 级派生目录
- 但不应让 group 拥有自己的 metadata 真相目录

## 为什么不建议 per-experiment shared root

### 1. 它会把控制平面切碎

`shared_root` 不是一个普通配置项目录。

它承载的是：

- task 真相
- batch 真相
- machine 真相
- indexes
- locks
- events

把它按实验计划切开，本质上就是把项目切成多个小调度系统。

### 2. 它会弱化项目内最重要的互斥边界

用户当前可以接受：

- 不同项目之间不做 host 级全局隔离

但无法接受：

- 同一项目里，只因为切了不同实验计划目录，就把 GPU 占用判断拆成两套

如果 `shared_root` 细到 experiment 级，这种问题一定发生。

### 3. 它会让运维观察面失真

当 root 被切碎后：

- `qexp top`
- `qexp machines`
- `qexp list`

看到的都只是某个实验计划自己的局部事实，而不是项目整体事实。

这不符合 `qexp` 作为项目内轻量队列控制面的定位。

## Metadata 膨胀问题

您提出的担心是合理的：

- 如果整个项目都共用 `/mnt/share/myproject/.qexp`
- 那么这个目录是不是会无限膨胀

答案是：

- 会增长
- 但这不构成把 root 切成 per-experiment 的充分理由

真正应解决的是：

- project root 内 metadata 的生命周期管理

而不是：

- 用错误的 shared_root 粒度规避增长

### 1. 为什么 project root 增长是可接受的

因为 `qexp` 当前保存的是轻量控制元数据，而不是训练产物本身：

- task json
- batch json
- indexes
- events
- machine state

这些对象的体积远小于：

- checkpoint
- tensorboard log
- artifact
- dataset cache

因此，project 级 root 的增长通常是“管理问题”，不是“架构上必须拆 root”的问题。

补充说明：

- 把 metadata 分散到多个 experiment 目录里，不会消除总量
- 它只会把同一项目的 metadata 分片化，增加查找、修复、观察和生命周期治理成本

### 2. 应该如何控制增长

工业级正确做法是：

1. 保持 project 级单 root
2. 用 `clean` 管终态 task 生命周期
3. 用索引重建能力保证派生视图可恢复
4. 必要时为旧 events / 旧 task 做归档或 retention 策略

而不是：

1. 为每个实验计划新开一个 root
2. 把控制真相天然碎片化
3. 牺牲项目内资源与观察一致性

### 3. 当前版本最少需要承认的约束

为了避免“无限膨胀”的担心演化成误用，本提案建议明确写入产品边界：

- `shared_root` 是项目级控制目录，不是一次性实验目录
- 用户应定期清理旧终态 task
- `qexp` 未来可增加 project-root 级 retention / archive 能力
- 但不应鼓励用户通过拆分 `shared_root` 规避 metadata 增长

## 工业级推荐方案

### 1. 正式定义 `shared_root` 为 project control root

建议在 product spec 和 manual 中明确写死：

- `shared_root` 的推荐粒度是 project 级
- 一个项目通常只应有一个活跃 `shared_root`
- 同项目的多个实验计划、group、batch、task 都进入这同一个 root

这些契约现已冻结在正式 spec 中；本节保留是为了说明该冻结形态背后的设计理由，而不是表示 spec 仍待补写。

### 2. 正式定义 `group` 为 project 内归组键

建议在术语层明确：

- `group` 用于项目内归组
- `group` 不是控制平面边界
- `group` 不应隐式映射到独立 `shared_root`

### 3. 正式承认“跨 root 不做资源一致性承诺”

必须明确写出来：

- 不同 `shared_root` 之间，`qexp` 不承诺共享 workset
- 不同 `shared_root` 之间，`qexp` 不承诺 GPU 占用互斥
- 用户若希望同一项目内保持资源一致性，必须复用同一个 `shared_root`

### 4. 引入 project-root 生命周期治理，而不是拆 root

后续应优先沿这条路线演进：

- 强化 `clean`
- 增加 retention policy
- 增加事件归档或裁剪策略
- 增加 root 体积与对象计数观测

而不是推动：

- experiment-root proliferation

## 全链路推演

### 推荐路径

输入：

- 项目目录：`/mnt/share/myproject/`
- 共享控制目录：`/mnt/share/myproject/.qexp`
- 两组实验计划：`exp1`、`exp2`

处理：

1. 用户对项目执行一次：

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
```

2. `exp1` 的 task 以 `group=exp1` 提交
3. `exp2` 的 task 以 `group=exp2` 提交
4. agent、tracker、observer 都在同一个 project root 内工作

结果：

- `exp1` 与 `exp2` 共享同一项目控制真相
- 同项目内的 GPU 占用判断基于同一套 task 真相
- 观察面不被拆碎

### 不推荐路径

输入：

```text
/mnt/share/myproject/.qexp/exp1/shared
/mnt/share/myproject/.qexp/exp2/shared
```

处理：

1. `exp1` 和 `exp2` 各自初始化自己的 root
2. 每套 root 只看到自己的 task / machine / events / indexes

结果：

- 同项目内资源视图被拆开
- 同项目内实验之间的 GPU 占用感知断裂
- 用户切 root 就像切到另一套平行宇宙

## 验收标准

- 文档正式定义 `shared_root` 的推荐粒度为 project 级
- 文档明确禁止把单个实验计划默认映射为独立 `shared_root`
- 文档明确 `group` 是 project 内归组键，不是控制平面边界
- 文档明确不同 `shared_root` 之间不承诺 GPU 占用一致性
- 文档明确 project-root metadata 增长应通过 lifecycle 治理解决，而不是通过拆 root 规避

## 非目标

- 不在本提案中实现 host 级全局 GPU lease
- 不在本提案中引入跨 project 的统一资源调度器
- 不在本提案中设计完整的 metadata archive 子系统
- 不要求所有用户只能有一个全局唯一 `shared_root`

## 需要特别确认的点

- **假设/未验证**：当前用户是否真的存在“同一项目拆多个 shared_root 仅为减少 metadata”的高频习惯；若存在，应在 user-facing 文档中继续强化纠偏措辞
- **假设/未验证**：目前 `clean`、`doctor`、`rebuild-index` 是否已足以支撑 project-root 长期运行；若不足，需要单独立项补 lifecycle 治理
- **假设/未验证**：未来是否需要 project-root 级指标，例如 task 总数、event 总数、索引体积和旧终态占比，以帮助用户判断何时清理

## 文档落位说明

该提案的目标形态已经冻结在正式文档中：

- `docs/manual/qexp.md`
- `docs/spec/qexp_product_spec.md`
- `docs/spec/qexp_runtime_spec.md`

因此，本文档不再承担“推动 spec 补写”的职责，而只承担以下作用：

- 记录为什么 `shared_root` 必须定义为 project 级控制平面
- 记录为什么 `group` 是归组键而不是控制平面边界
- 记录为什么 metadata 增长问题应通过 lifecycle 治理而不是拆 root 解决

当前剩余的后续工作应理解为实现与运维跟进，而不是 spec 跟进，主要包括：

- 验证 CLI 与 observer 是否完全收敛到冻结后的 `group` / `batch.group` 契约
- 评估 `clean`、`doctor`、`rebuild-index` 是否足以支撑 project-root 长期运行
- 视需要补充 retention、archive、体积观测等 lifecycle 治理能力

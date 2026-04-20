# qexp group-to-session mapping

状态：已实现归档

更新时间：2026-04-17

完成时间：2026-04-20

关联文档：

- [qexp Manual](/mnt/c/Users/Administrator/proj/qqtools/docs/manual/qexp.md)
- [qexp Product Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_product_spec.md)
- [qexp Runtime Spec](/mnt/c/Users/Administrator/proj/qqtools/docs/spec/qexp_runtime_spec.md)

## 目标

本文档只讨论一件事：

为 `qexp` 引入“group 级 session 分组”能力。

目标行为是：

- 一个 group 对应一个 tmux session
- 同一 group 下的多个 task run 对应该 session 下的多个 tmux window
- 用户可以在提交 task 时显式声明该 task 属于哪个 group

本文档不讨论：

- 训练日志体系
- artifact 管理
- batch 重试策略重构
- tmux 之外的执行后端替换

## 背景

当前 `qexp` 的实际行为是：

- task run 默认进入固定 tmux session：`experiments`
- 每个 task run 在该 session 中创建一个新 window
- agent / daemon 进入独立的内部 session：`qqtools_internal`

这套行为能满足“单 session 多 window”的基本执行模型，但缺一个稳定的工具级归组维度：

- 用户实际会先定义一组要一起观察和管理的 task run
- 然后在该组下发起多个 task run
- 用户希望 tmux 视图直接反映这个归组关系，而不是把所有 task run 混在一个全局 `experiments` session 里

典型场景：

- 用户自己在业务上把“比较参数 `n=4` 和 `n=6`”视为一个实验计划
- 在调用 `qexp` 时，将其映射成一个 group：`contract_n_4and6`
- 用户希望：

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

产生的 tmux 结构是：

```text
session: contract_n_4and6
  window 1: n4 / task_xxx
  window 2: n6 / task_yyy
```

而不是：

```text
session: experiments
  window ...: task_xxx
  window ...: task_yyy
  window ...: 其他实验的 task
```

## User Story

一个典型的日常使用场景如下：

- 用户脑中先有一个明确的实验计划
- 例如，这次要比较参数 `n=4` 和 `n=6` 的行为差异
- 用户在纸面、笔记、PRD 或个人工作流里，把这件事理解为一次完整实验

但在调用 `qexp` 时，用户不需要把“实验计划”这个业务概念输入到工具里。

用户只需要做一层简单心智转化：

- 选一个稳定的 `group` 名
- 用这个 `group` 把本次相关 task run 归在一起

例如：

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

用户对这两个命令的预期不是：

- `qexp` 理解“实验计划”这四个字

而是：

- `qexp` 把同一个 `group` 下的 task run 放进同一个 tmux session
- 用户进入这个 session 后，能同时看到本次相关 run 的多个 window
- 用户可以围绕这个 session 做 attach、观察、比对、排障

这套工作流的价值在于：

- 用户保留业务侧命名自由
- `qexp` 保持工具层抽象克制
- tmux 结构仍然能稳定映射到用户当下的一组工作上下文

因此，本提案虽然不把“实验计划”写成正式字段，但明确承认它是常见用户心智来源之一。

`qexp` 的职责不是理解业务语义本身，而是提供一个稳定的 `group` 机制，让用户把这类业务语义自行投影到运行时归组结构里。

## 问题定义

当前缺的不是：

- “单个 task 是否能在 tmux 中运行”

而是：

- `qexp` 没有把“group”建模成一等归组对象

结果是：

- 工具层没有稳定字段承载归组信息
- 调度器无法根据 group 选择 tmux session
- 观察面只能看到 task 级运行，不能直接看到“这个 task 属于哪个 group”

因此，本需求的本质是：

- 为 `qexp` 增加稳定的 group-to-session 契约

而不是：

- 临时把 `name` 偷当成 session 名
- 让用户手工预先创建 tmux session
- 为每个 task 单独创建一个 session

## 名词对齐

本提案要求 `qexp` 明确区分三层概念：

### 1. `group`

表示用户在工具层声明的一组 task run。

例如：

- `contract_n_4and6`
- `qm9_seed_sweep`
- `ablation_dropout_202604`

语义要求：

- `group` 是用户显式声明的归组键
- 一个 `group` 对应一个 tmux session
- 一个 `group` 下可以包含多个 task run
- “实验计划 -> group” 的心智映射由用户自己管理，不进入 `qexp` 的正式术语

### 2. `task run`

表示一次具体提交执行。

例如：

- `python train.py --n 4`
- `python train.py --n 6`

语义要求：

- 每个 task run 仍然保留独立 `task_id`
- 每个 task run 在 tmux 中对应一个 window
- task run 是调度、取消、重试、日志、状态跟踪的基本对象

### 3. `task_id`

表示系统内部唯一标识。

语义要求：

- `task_id` 不等于 `group`
- `task_id` 不承担归组语义
- `task_id` 继续作为 `cancel`、`retry`、`logs`、`inspect` 的主键

### 4. `batch`

表示一次批量提交产生的 task 集合。

语义要求：

- `batch` 是操作级集合，不是长期归组键
- `batch` 主要服务于“一起提交、一起观察、一起重试”
- `batch` 不应替代 `group`

关系约束：

- 一个 `group` 可以包含多个 `batch`
- 一个 `batch` 通常应默认落在一个 `group` 下
- `batch` 和 `group` 可以相同名，但不能强制绑定为同一概念

## 设计原则

### 1. 归组键必须显式化

group 不能继续停留在用户心智里，必须进入 task 元数据。

否则：

- 调度器无法稳定复原 session 归属
- observer 无法展示 group 维度
- retry / clean / inspect 无法在输出中保留归组信息

### 2. 业务语义优先于 tmux 细节

tmux session 只是运行时投影。

真正应被建模的是：

- task 属于哪个 group

而不是：

- 让业务代码直接操作 tmux session 名

最终映射关系应写死为：

- `group -> tmux session`
- `task run -> tmux window`

### 3. 默认路径必须稳定明确

本需求不做 v1 兼容设计，只面向当前正式演进方向实现。

因此：

- 未显式声明 `group` 的 task，继续回落到默认 session：`experiments`
- `group = null` 是正式且稳定的未归组语义

### 4. 展示名与归组键必须分离

`name` 适合作为人类可读标签。

它不应承担：

- tmux session 主键
- 归组主键
- 稳定业务标识

否则会把“显示名”和“分组键”混成一个字段，后续必然失控。

### 5. 一次收敛到长期稳定契约

本需求不适合做“先偷塞 session_name 参数、以后再说”的临时方案。

工业级最终态应直接做到：

- CLI 有显式入口
- 模型有正式字段
- 调度有稳定选择逻辑
- observer / inspect / top 有一致输出

## 工业级推荐方案

### 1. 新增正式字段 `group`

`qexp task` 模型必须新增：

```python
group: str | None
```

推荐约束：

- 允许为空，表示“未归组”
- 非空时必须通过与 `task_id` 同等级别的安全字符校验
- 禁止路径分隔符、空白折叠歧义和 tmux 不稳定字符

推荐首版允许字符集：

- 字母
- 数字
- `.`
- `_`
- `-`

不推荐首版允许空格。原因：

- tmux session 名、CLI 参数、脚本拼接、日志输出都会受影响
- 首版优先稳定，不引入 quoting 复杂度
- 首版优先避免“用户输入值”和“实际 session 名”发生隐式变形

额外约束：

- `group` 必须区分大小写，系统不做大小写折叠
- `group` 必须禁止保留名：`experiments`
- `group` 必须禁止保留名：`qqtools_internal`
- `group` 必须禁止以 `.` 或 `-` 开头，避免和 shell/tmux 观察习惯冲突
- `group` 长度上限固定为 64 个字符
- `group` 校验通过后，`tmux_session = group`，禁止再做二次 sanitize 或隐式重写

这样保证：

- task 真相中的 `group` 与运行态 `tmux_session` 一一对应
- 用户看到的 group 名和 tmux session 名完全一致
- 不会出现两个不同 group 在 sanitize 后映射到同一个 session 的冲突

### 2. CLI 显式新增 `--group`

`submit` 系列命令必须支持：

```bash
qexp submit --group <group> -- ...
```

若存在 `batch-submit` 或 manifest 提交，也必须允许每个 task 或整批默认 group 声明。

推荐规则：

1. `--group` 为可选参数
2. 未传入时，task 的 `group = null`
3. 传入时，写入 task 真相
4. `retry` 的正式接口为 `qexp retry <task_id>` 与 `qexp retry <task_id> --group <group>`
5. `qexp retry <task_id>` 默认继承原 task 的 `group`
6. `qexp retry <task_id> --group <group>` 使用显式覆盖值，不继承原 group

### 3. 调度器按 `group` 选择 tmux session

调度 task 时，必须先计算“目标 session 名”：

```text
if task.group is not None:
    session_name = task.group
else:
    session_name = "experiments"
```

然后：

- 在该 session 中创建 task 对应的 window
- 将实际使用的 `tmux_session` 写回 task 运行态元数据
- task 终态后不主动销毁该 session
- 只有用户手工清理 tmux session，或后续独立引入专门的 session GC 能力时，session 才允许被删除
- task 状态真相只能由 task 元数据决定，不允许根据 tmux session 是否存在反推 task 状态

这样保证：

- group 维度决定 session 归属
- 实际运行态仍保留可观测的 `tmux_session`
- task 结束后用户仍可进入原 session 做复盘和 debug

### 4. tmux window 命名升级为“可读且稳定”

当前 window 只按 `task_id[:48]` 命名，信息密度不足。

工业级建议：

- 优先使用 `name`
- 若无 `name`，回落到 `task_id`
- 当名称冲突时，由 tmux 自动追加索引，或显式拼接短 task_id

推荐策略：

```text
window_label = sanitize(name) if name else task_id
final_window_name = f"{window_label}__{task_id[:8]}"
```

这样可以同时满足：

- 用户肉眼可读
- 多次同名 run 不冲突
- 排障时能快速定位 task_id

### 5. 观察面必须显式展示 `group`

至少以下命令应补出 group 维度：

- `qexp status`
- `qexp inspect <task_id>`
- `qexp top`
- `qexp list` / `machines` / `batch` 中与 task 展示相关的表格

最低要求：

- task 明细中可见 `group`
- task 运行中时可见真实 `tmux_session`
- task 已终态但曾成功进入 tmux 时，仍应保留最近一次实际 `tmux_session`

### 6. 批量入口必须支持 group 归组

若后续主要使用 `batch-submit` 来表达一组相关 task，则不能只给单 task `submit` 加字段。

工业级契约要求：

- batch manifest 可以声明默认 `group`
- task 级声明可以覆盖 batch 默认值
- 同一 batch 允许包含多个 group，但默认不推荐

正式优先级：

1. `tasks[].group` 优先级最高
2. 若 `tasks[].group` 缺失，则继承 `batch.group`
3. 若二者都缺失，则该 task 的 `group = null`

原因：

- batch 和 group 不是同一个概念
- batch 是“一起提交的一组 task”
- group 是“工具归组键”

二者可以相同，但不能强行绑定成一个字段。

进一步约束：

- `batch` 解决的是“这次一起交了哪些 task”
- `group` 解决的是“这些 task 长期属于哪个工作上下文”
- 一个长期实验上下文中，允许出现多次 batch-submit，因此也允许存在多个 batch

## 行为语义

### 用户提交

输入：

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
```

处理：

1. CLI 解析 `--group`
2. API 创建 task，并将 `group="contract_n_4and6"` 写入 task payload
3. task 进入 pending 队列
4. 调度器读取该 task
5. 计算 `session_name = "contract_n_4and6"`
6. 若 session 不存在，则创建
7. 在该 session 下创建一个新 window
8. 在该 window 中启动 runner
9. 将 `tmux_session="contract_n_4and6"` 和 `tmux_window_id` 写入运行态

输出：

- 用户可在 `tmux ls` 中看到 session：`contract_n_4and6`
- 可在该 session 下看到本 task 对应的 window

### 用户再次提交到同一 group

输入：

```bash
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

处理：

- 不新建第二个 session
- 复用已有 session：`contract_n_4and6`
- 仅新增一个 window

输出：

- 同一 group 的多个 task run 被稳定聚合到同一 session

### 用户重试已有 task

默认输入：

```bash
qexp retry task_xxx
```

处理：

1. 读取原 task 真相
2. 创建一个新的 task run
3. 默认继承原 task 的 `group`
4. 重新按继承后的 `group` 计算目标 session
5. 在对应 session 中创建新的 tmux window

输出：

- retry 产生的新 task 与原 task 保持相同 group 归属
- 用户可以在同一 group session 中连续观察多次 attempt

显式覆盖输入：

```bash
qexp retry task_xxx --group regrouped_debug
```

处理：

- 若显式传入 `--group`，则新 task 使用覆盖后的 group
- 原 task 真相不被改写
- 新 task 进入新 group 对应的 tmux session

输出：

- retry 可以在“保持原归组”与“迁移到新归组”之间显式选择

### 用户未声明 group

输入：

```bash
qexp submit --name baseline -- python train.py
```

处理：

- task `group = null`
- 调度时回落到默认 session：`experiments`

输出：

- 使用默认未归组路径：`experiments`

## CLI Examples

下面列出几种推荐的命令范式，作为本提案的直接用户入口示例。

### 1. 单任务提交，不声明 group

这是默认路径。

```bash
qexp submit --name baseline -- python train.py --config configs/base.yaml
```

预期行为：

- task 会正常进入队列
- 若未声明 `group`，调度时回落到默认 tmux session：`experiments`
- 该 task 在 `experiments` session 下占用一个独立 window

### 2. 单任务提交，显式声明 group

当用户已经知道这个 task 属于某个明确归组时，可以直接声明：

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
```

预期行为：

- task 元数据中记录 `group=contract_n_4and6`
- 调度时目标 tmux session 为 `contract_n_4and6`
- 若 session 不存在，则自动创建
- 该 task 在该 session 下占用一个独立 window

### 3. 同一 group 下连续提交多个任务

这是本提案最核心的使用范式。

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group contract_n_4and6 --name n6 -- python train.py --n 6
```

预期行为：

- 两个 task 共享同一个 tmux session：`contract_n_4and6`
- 每个 task 各自创建独立 window
- 用户 attach 到该 session 后，可以直接在一个会话内观察和切换本次相关 run

### 4. 不同 group 下分别提交任务

当两组 task 不属于同一个工作上下文时，应显式分到不同 group：

```bash
qexp submit --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --group dropout_ablation --name dropout_01 -- python train.py --dropout 0.1
```

预期行为：

- 第一条进入 session：`contract_n_4and6`
- 第二条进入 session：`dropout_ablation`
- 两组 task 在 tmux 中互相隔离，不混在同一个 session 里

### 5. 同一 group 下混合显式 task_id

当用户希望手动指定 task_id 时，仍然不影响 group 归组语义：

```bash
qexp submit --task-id qm9_n4 --group contract_n_4and6 --name n4 -- python train.py --n 4
qexp submit --task-id qm9_n6 --group contract_n_4and6 --name n6 -- python train.py --n 6
```

预期行为：

- `task_id` 继续只承担唯一标识职责
- `group` 继续决定 tmux session
- `name` 继续承担展示职责

三者语义不得混用

### 6. 批量入口形态

`batch-submit` 的 manifest 形态应类似：

```yaml
batch:
  name: contract-compare
  group: contract_n_4and6

tasks:
  - name: n4
    group: contract_n_4and6
    command: ["python", "train.py", "--n", "4"]
  - name: n6
    group: regrouped_debug
    command: ["python", "train.py", "--n", "6"]
```

预期行为：

- 第一条 task 使用显式 `group=contract_n_4and6`
- 第二条 task 使用显式 `group=regrouped_debug`
- 若 task 未声明 `group`，则继承 `batch.group=contract_n_4and6`

补充说明：

- 这里的 `batch.name` 仍然表示“这次批量提交”的名字
- `group` 表示这些 task 所属的长期归组
- 因此，后续继续补交同一实验计划的新任务时，应优先复用同一个 `group`，而不是强行复用同一个 `batch`

## 全链路推演

输入：

- 用户声明一个 `group`
- 用户在该 group 下提交多个 task run

处理：

1. CLI 接收 `--group`
2. API 校验并持久化 `group`
3. pending task 在元数据中保留归组信息
4. 调度器按 `group` 计算目标 session
5. tmux 层确保该 session 存在
6. 每个 task run 创建独立 window
7. runner 启动后继续现有日志、取消、退出码、状态流转链路
8. observer 从 task 真相中读取 `group` 与 `tmux_session`

状态流转：

- `group` 在 task 生命周期内保持稳定
- `tmux_session` 从“默认固定值”变为“按 task 归组动态决定”
- 一个 group 下的多个 task run 可共享 session
- task 进入终态后，原 session 继续保留，不因最后一个 task 结束而被自动销毁
- session 是否仍存在只影响用户调试便利性，不影响 task 的正式终态判定

输出：

- CLI 可见 `group`
- tmux 结构可直接映射业务实验计划
- inspect / status / top 的输出更接近用户真实工作流

影响：

- 新任务可获得更清晰的归组与运行视图
- task 结束后仍保留可用于 debug 的 tmux 上下文
- 不改变 task 作为调度主键的基本架构

## 代码改动范围

最低必改项：

- `src/qqtools/plugins/qexp/v2/models.py`
- `src/qqtools/plugins/qexp/v2/api.py`
- `src/qqtools/plugins/qexp/v2/cli.py`
- `src/qqtools/plugins/qexp/v2/scheduler.py`
- `src/qqtools/plugins/qexp/tmux.py`
- `src/qqtools/plugins/qexp/v2/observer.py`

## 测试要求

至少补以下测试：

### 1. 模型测试

- `group=None` 可正常解析
- 合法 `group` 可通过校验
- 非法 `group` 被拒绝
- 缺少 `group` 字段的 task payload 可正常读取

### 2. 调度测试

- 同一 `group` 的两个 task 会选择同一个 `tmux_session`
- 两个不同 `group` 的 task 会进入不同 session
- 未声明 `group` 的 task 会回落到 `experiments`

### 3. CLI/API 测试

- `submit --group ...` 会把 group 写入 task
- `retry <task_id>` 默认继承 group
- `retry <task_id> --group <new_group>` 会覆盖继承值
- `batch.group` 会作为 task 默认值下发
- `tasks[].group` 会覆盖 `batch.group`
- inspect / status 输出包含 group

### 4. 观察测试

- `top` / `status` / `inspect` 能展示 group
- task 运行中时展示真实 `tmux_session`
- task 终态后仍可展示最后一次实际 `tmux_session`
- task 终态不会触发 session 自动删除
- session 缺失不会导致已终态 task 被误判为异常状态

## 验收标准

- 用户可以通过 `qexp submit --group <name> -- ...` 声明归组
- 同一 `group` 下的多个 task run 会进入同一个 tmux session
- 同一 `group` 下的 task run 会在该 session 中创建不同 window
- 不同 `group` 默认进入不同 session
- 未声明 `group` 的用法继续回落到 `experiments`
- `name` 仍是展示字段，不被偷用为 session 主键
- `status` / `inspect` / `top` 中可以看到 group
- `retry <task_id>` 默认不会丢失 group 归组信息
- `retry <task_id> --group <new_group>` 可以显式迁移到新 group
- task 终态后对应 tmux session 默认继续保留，便于用户 debug

## 非目标

- 不把 `group` 升级成独立的数据库级实体
- 不在首版引入 `qexp experiment create` 一类预创建命令
- 不把 batch 强制等同于 group
- 不改变 agent / daemon 使用 `qqtools_internal` session 的内部约定

## 需要特别确认的点

- `docs/manual/qexp.md` 需要同步补齐 `name`、`group` 与 `task_id` 的术语边界
- `docs/manual/qexp.md` 需要同步补齐 `batch.group` 与 `tasks[].group` 的优先级规则

## 后续文档影响

至少需要同步更新：

- `docs/manual/qexp.md`

需要补齐的正式契约包括：

- `group` 的术语定义与字段语义
- `group -> tmux session` 的运行时映射
- 未声明 `group` 时的回落规则
- `retry` 与 `batch-submit` 的 group 继承与覆盖规则

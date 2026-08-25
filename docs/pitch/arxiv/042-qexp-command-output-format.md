---
doc_type: pitch
status: archived
updated_at: 2026-08-25
archived_at: 2026-08-25
---

# qexp 命令输出格式

状态：已归档

更新时间：2026-08-25
归档时间：2026-08-25

关联任务：为 qexp 的结构化命令结果提供面向人类的默认输出，同时保留机器可消费的 JSON 输出。

## 背景与目标

当前 qexp 的多数结构化命令直接向 stdout 输出 JSON。JSON 适合脚本、自动化和程序解析，但人在终端中检查任务、组、机器和 Agent 状态时，需要从嵌套字段中手动提取关键信息，阅读效率较低。

本需求的目标是一次性升级 qexp 的观察型 CLI 契约：

- qexp 默认输出人类可读的纯文本；
- `--format=json` 输出完整、可解析的 JSON；
- 使用统一的 `--format` 参数选择输出格式；
- 不改变调度、状态流转、退出码和共享 JSON 持久化记录。

## 范围与非目标

本期范围：

- 为所有产生有限结构化结果的 qexp 命令提供 `--format` 参数；
- 支持 `human` 与 `json` 两种格式，默认 `human`；
- 将当前 `task offer`、`task share`、`task keep-local` 的 `text` 格式值移除，以 `human` 统一替代；
- 使用 `--format=json` 时输出完整、可解析的 JSON 结果；
- 将仓库内依赖默认 JSON stdout 的调用与测试迁移为显式 `--format=json`。

本期非目标：

- 不删除或缩减 JSON 结果字段；
- 不改变共享目录中的 JSON 持久化记录；
- 不引入 YAML、表格导出、颜色主题或 TUI；
- 不重新格式化 `qexp logs` 与 `qexp task logs` 的日志正文；
- 不把单值成功输出（task ID、路径或无输出成功）强制改写为多行展示；
- 不实现或恢复与输出格式无关的观察选项，例如 `top --all`、`top --group`。

## 接口升级与迁移

本期是一次性 CLI stdout 契约升级，不提供 `text` 别名、默认 JSON 开关、环境变量回退或其他后向兼容层。

- 未指定 `--format` 的结构化命令，输出从 JSON 变为人类可读文本。这是本期唯一有意的破坏性 stdout 变更。
- `--format=text` 必须被 argparse 拒绝；用户应改为 `--format=human`。
- 所有需要机器解析 stdout 的调用必须显式传入 `--format=json`。
- `qexp batch-submit` 的预提交恢复通知是例外：它在 Task staging 前将 operation ID 和 idempotency key 输出至 stderr，以保证中断后仍可恢复。JSON 模式的 stdout 只输出提交完成后的单个 JSON 文档。
- 自动化调用应优先显式提供 `--idempotency-key`，而不是从预提交 stderr 通知中提取生成的 key。

该升级不保留旧 stdout 默认行为，因为“人类观察默认可读”与“默认 JSON 可被脚本解析”不能由同一无格式参数调用同时满足。显式格式选择是清晰、可验证且长期稳定的边界。

## 使用场景

### 场景 A：终端人工观察

用户执行下列命令时，应直接看到突出状态和关键字段的可读结果：

```bash
qexp task list
qexp task show task_xxx
qexp group show stage-c1
qexp top
qexp machines
qexp agent status
```

例如，任务列表应按任务展示 task ID、名称、状态、GPU 数、所属组、机器和失败原因（存在时），而不是输出 JSON 数组。

### 场景 B：脚本与自动化

脚本需要读取完整结构化数据时，显式请求 JSON：

```bash
qexp task list --format=json
qexp task show task_xxx --format=json
qexp group show stage-c1 --format=json
qexp top --format=json
```

除 `batch-submit` 的 stderr 预提交恢复通知外，stdout 必须保持为单个 JSON 文档，便于 `json.loads`、`jq` 和其他机器消费者解析。

## 功能性需求

### 1. 参数契约

- 每个纳入格式化范围的叶子命令必须接受位于命令路径末尾的 `--format` 参数。
- 参数可选值固定为 `human` 和 `json`。
- 默认值必须为 `human`。
- `--format=text`、`--format=xml` 等不支持值必须在业务执行前返回参数错误；不得静默回退。
- `qexp task list --format=json`、`qexp group show <name> --format=json` 等嵌套写法必须有效。
- `submit` 不定义 qexp 的 `--format`，并使用 `argparse.REMAINDER` 保留训练命令 argv；尤其 `--` 后出现的 `--format=json` 必须原样透传给训练命令，不由 qexp 解释、拒绝或格式化。

### 2. JSON 输出契约

- `--format=json` 必须返回该命令当前完整结果数据，不得因新增人类格式而丢失字段或语义。
- stdout 中除该 JSON 文档外不得混入说明文本、标题或 ANSI 控制序列。
- 错误信息继续写入 stderr，退出码保持现有行为。
- 既有 JSON 结果的数据模型必须保持兼容；空白、缩进和键排序不属于兼容承诺。
- `batch-submit` 的命令工作流必须在首次提交与幂等重试两条路径均返回同一完整提交结果对象；该对象至少包含 `operation_id`、持久化的 `idempotency_key`、`target_group`、`task_ids` 和提交状态。
- `batch-submit` 必须在成功时向 stdout 输出上述提交结果对象的 JSON 文档；预提交恢复通知不写入 stdout。CLI formatter 仅渲染该对象，不从 Task 或 Submission Operation 文件反查、拼装领域结果。
- `agent run` 仅输出一个启动状态结果，然后进入前台运行；启动状态依其 `--format` 选择渲染，agent 运行期间不得向 stdout 输出额外的结构化结果。

### 3. 默认人类输出契约

所有 `human` 输出均为纯文本，不依赖 TTY 检测；重定向到文件时仍保持可读。

- 列表结果以固定表头和一行一个对象展示；无对象时固定输出 `No results.`。
- 详情结果以固定顺序的 `Label: value` 行展示，并按摘要、状态和控制信息分组；分组之间使用一空行。
- 操作结果以 `Action: <value>` 和 `Status: <value>` 开头，随后按固定顺序展示对象标识、待处理对象及原因。
- 缺失值展示为 `-`；空集合展示为 `none`；失败原因、待确认信息或阻塞原因存在时必须单独显示。
- 人类输出只展示观察和操作所需的摘要字段，不展开原始持久化 JSON、完整 attempt 历史或配置内部结构。

各结果族的最低字段集合如下：

| 结果族 | 命令 | 人类输出的最低字段，按所列顺序 |
| :--- | :--- | :--- |
| Task 列表 | `task list` | Task ID、Name、State、GPUs、Group、Home machine、Queue scope、Attempt、Claimed machine、Reason |
| Task 详情 | `task show` | Task ID、Name、Command、State、GPUs、Group、Placement、Queue scope、Control、Attempts、Reason |
| Group 列表与详情 | `group list`、`group show` | Group、Admission、Dispatch、Workers、Task summary、Queue summary、Control operation、Reason |
| Group 控制与 worker 操作 | `group create`、`group seal|reopen|pause|resume|cancel|retry-failed`、`group machines add|drain|remove` | Action、Status、Group、Worker machine、Task IDs、Pending machines、Reason |
| Machine 与 top 观察 | `machines`、`top` | Machine、Availability、GPU visible、GPU reserved、GPU unreserved、Agent state、Task summary、Reason |
| Task 控制与可用性 | `task cancel|offer|share|keep-local` | Action、Status、Task ID、Queue scope、Eligible machines、Pending acknowledgement、Reason |
| Agent | `agent start|run|restart|status|stop` | Action、Agent mode、Machine、Agent state、PID、Previous PID、Reason |
| 诊断与清理 | `doctor verify|repair`、`clean` | Action、Status、Summary、Candidates、Removed、Pending machines、Failures、Reason |
| 配置与租约策略 | `config notifications show|set|provider set`、`lease-policy show|set` | Action、Status、Enabled、Provider、Credential source、Lease fields、Reason |
| Batch 提交 | `batch-submit` | Action、Status、Operation ID、Idempotency key、Group、Task count、Task IDs、Reason |
| CLI 上下文 | `use --show` | Shared root、Machine、Runtime root |

字段不存在时按本节缺失值规则展示，不为不同命令临时改变标签或顺序。表格列宽、换行位置与长字符串截断宽度不属于稳定契约；截断时必须保留可辨识的对象 ID，并以 `…` 标识。

### 4. 命令覆盖边界

下列有限结构化结果命令必须纳入 `--format`：

- `batch-submit`；
- `task cancel`、`task offer`、`task share`、`task keep-local`、`task list`、`task show`；
- `group create`、`group list`、`group show`、`group seal`、`group reopen`、`group pause`、`group resume`、`group cancel`、`group retry-failed`、`group machines add|drain|remove`；
- `agent start`、`agent run` 的启动状态、`agent restart`、`agent status`、`agent stop`；
- `top`、`machines`、`doctor`、`clean`、`use --show`；
- `config notifications show|set|provider set`、`lease-policy show|set`。

下列命令不纳入格式化范围，并保持现有单值或日志行为：

- `qexp logs` 与 `qexp task logs`；
- `submit` 与 `task retry`，继续输出单个 task ID；
- `init` 与 `migrate`，继续输出路径；
- `use` 的无输出成功路径与 `use --clear`。

### 5. 输出边界与执行安全

- 格式化层必须位于 CLI 输出边界，不得让人类展示逻辑污染 `observer.py`、命令工作流或 runtime JSON 记录。
- 格式选择只能在业务操作完成后影响渲染；非法格式必须由参数解析器在业务操作前拒绝。
- 格式化不得改变命令执行路径、调度状态流转、持久化内容、激活行为或退出码。
- `batch-submit` 的预提交恢复通知必须在 Task staging 前写入 stderr 并立即 flush；该通知必须包含 operation ID 和 idempotency key。
- 人类格式应使用标准库或项目现有依赖；本需求不以引入第三方渲染库为前提。

## 文档同步

实现时必须扫描并同步所有仍维护的 qexp CLI 示例与正文契约；`docs/pitch/arxiv/` 仅保存历史上下文，不在本期改写范围。

- `docs/spec/qexp_product_spec.md`：将所有 `--json` 与 `--format json` 写法统一改为 `--format=json`；移除当前 parser 不支持的 `top --all` 与 `top --group` 示例，并将 `top` 的观察范围描述更正为当前项目级行为。该项只纠正文档与当前行为的偏差，不实现新的 top 过滤接口。
- `README.md`：将现有 `qexp task offer TASK_ID --format json` 示例升级为 `--format=json`，并说明所有机器读取场景必须显式传入该参数。

## 验收标准

- [ ] `qexp task list` 默认输出人类可读结果，且不包含 JSON 数组或对象外壳。
- [ ] `qexp task list --format=json` 输出可由 JSON 解析器解析，并保留当前列表项字段。
- [ ] `qexp task show <id>` 默认输出可读详情；`qexp task show <id> --format=json` 保留完整任务及 Attempt 数据。
- [ ] 所有“命令覆盖边界”中列出的有限结构化结果命令均接受 `--format=human|json`，并遵守对应输出契约。
- [ ] `task offer`、`task share` 与 `task keep-local` 接受 `--format=human|json`；`--format=text` 被拒绝。
- [ ] `qexp batch-submit --format=json` 的 stdout 为单个可解析 JSON 文档；Task staging 前的 operation ID 与 idempotency key 仅写入 stderr。
- [ ] `agent run --format=json` 仅输出一个可解析的启动状态文档，随后不向 stdout 输出其他结构化结果。
- [ ] `qexp logs`、`qexp task logs`、单 task ID 输出和路径输出保持既有文本行为。
- [ ] `--format=xml` 等不支持值返回参数错误，且不执行对应业务操作。
- [ ] 仓库内依赖 JSON stdout 的测试和自动化调用显式改用 `--format=json` 后继续通过。

## 实施方案

### Phase 1：固化 CLI 格式契约

- [ ] 在 `src/qqtools/plugins/qexp/cli.py` 为本 pitch 的覆盖命令逐一挂载 `--format`，仅允许 `human` 与 `json`，默认 `human`。
- [ ] 删除 `task offer`、`task share`、`task keep-local` 的 `text` 格式值；不增加兼容别名。
- [ ] 保持 `submit` 的 `argparse.REMAINDER` 路径不变；验证 `--` 后的 `--format=json` 原样进入训练命令 argv。
- [ ] 将 `batch-submit` 命令工作流升级为在首次提交和已提交的幂等重试中均返回同一完整提交结果对象。
- [ ] 将 `batch-submit` 的预提交恢复通知移至 stderr，并在 stdout 仅保留该完整结果对象的单一最终结果。

交付物：无兼容层的统一 CLI 参数与 stdout/stderr 契约。

### Phase 2：实现输出边界

- [ ] 新增仅负责渲染的 qexp CLI formatter 模块。
- [ ] 按“默认人类输出契约”的结果族、标签和顺序实现 human renderer。
- [ ] 保持 JSON 分支直接输出命令工作流返回的完整结果数据；batch formatter 不读取或拼装 Submission Operation 持久化记录。
- [ ] 将所有 CLI 输出调用统一收敛到 formatter，不修改 `observer.py`、命令工作流或 runtime 数据模型。

交付物：默认人类输出与明确 JSON 模式。

### Phase 3：迁移、验证与文档同步

- [ ] 将 unit、E2E 和仓库内自动化中依赖默认 JSON 的调用显式改为 `--format=json`；更新 `jrun()` 等 JSON 辅助入口。
- [ ] 为 human、JSON、非法格式、日志边界、batch stderr 时序及幂等重试结果、agent run 启动结果、`text` 拒绝和 submit 的 `--format=json` argv 透传补充 CLI 测试。
- [ ] 扫描并同步所有仍维护的 qexp CLI 文档契约；不改写 `docs/pitch/arxiv/`。

交付物：回归测试、调用方迁移与长期命令契约文档。

| ID | 测试场景 | 输入/操作 | 预期结果 |
| :--- | :--- | :--- | :--- |
| TC-01 | 默认人工观察 | `qexp task list` | stdout 为带固定表头的任务摘要，不是 JSON 数组。 |
| TC-02 | 机器读取列表 | `qexp task list --format=json` | stdout 是可解析 JSON，列表字段与当前结构兼容。 |
| TC-03 | 机器读取详情 | `qexp task show <id> --format=json` | stdout 包含完整 Task 与 Attempt 数据。 |
| TC-04 | 嵌套命令参数 | `qexp group show <name> --format=json` | 参数可被解析，stdout 为 JSON。 |
| TC-05 | 一次性升级 | `qexp task share <id> --format=text` | 参数错误、非零退出，且不执行业务操作。 |
| TC-06 | 非法格式 | `qexp top --format=xml` | 参数错误、非零退出，且不执行业务操作。 |
| TC-07 | Batch JSON 边界 | `qexp batch-submit --format=json ...` | stdout 为单一 committed JSON；预提交 operation ID/key 出现在 stderr。 |
| TC-08 | Batch 幂等重试 | 以已提交的 idempotency key 重试 `qexp batch-submit --format=json ...` | 返回与首次提交同构的完整结果，包含持久化 operation ID、key、target group、task IDs 和 committed 状态。 |
| TC-09 | Submit 参数透传 | `qexp submit -- python train.py --format=json` | `--format=json` 原样传给训练命令，qexp 不解析为自身格式参数。 |
| TC-10 | Agent run 边界 | `qexp agent run --format=json` | stdout 仅有启动状态 JSON，前台运行逻辑不变。 |
| TC-11 | 日志边界 | `qexp logs <id>` | 日志正文原样输出，不受格式化影响。 |
| TC-12 | E2E 调用迁移 | 已安装 entry point 下的 JSON 辅助调用 | 每个机器解析调用均显式传 `--format=json`。 |

## 验收清单

- [ ] 逻辑验证：所有结构化命令的格式参数、默认 human 输出和 JSON 路径均通过测试。
- [ ] 接口升级：旧 `text` 值与默认 JSON 依赖已从仓库调用方移除，且不保留运行时兼容层。
- [ ] 风险控制：batch 的预提交恢复信息在 staging 前写入 stderr；`submit` 参数透传、`agent run` 和 `doctor --strict` 的既有执行/退出码语义已验证。
- [ ] 文档同步：产品规格与 README 已更新为 `--format=json`，且删除当前未实现的 top 选项示例。

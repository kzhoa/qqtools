---
doc_type: pitch
status: drafting
updated_at: 2026-09-01
archived_at:
---

# Main-first Unit、Integration 与 E2E 测试治理提案

关联文档：

- [ADR-0001: Govern Compatibility Through the Local Release Preflight](../adr/0001-local-preflight-compatibility-governance.md)
- [qexp 测试架构治理提案](qexp-test-architecture-governance.md)
- [ADR-QEXP-0005: Single-Host qexp Test Architecture](../adr/qexp/0005-single-host-test-architecture.md)
- [测试契约矩阵](../../tests/CONTRACT_MATRIX.md)
- [当前 CI workflow](../../.github/workflows/ci.yml)
- [当前 tox 入口](../../tox.ini)

## 背景

qqtools 的主要开发路径是在本地 `main` 上完成修改和 commit，运行 preflight，通过后直接 push。PR 可以存在，但不是默认入口。

仓库已经使用 `tests/unit/`、`tests/integration/` 和 `tests/e2e/` 三层目录。当前问题不是缺少测试层级，而是执行职责发生了漂移：

- CI 重复运行本应由本地 preflight 证明的 Unit 和 Integration；
- qexp 的 `host_exclusive` 当前位于 Integration，但它实际验证生产默认宿主资源；
- 普通 `tox run -e e2e` 使用 editable 源码环境，不能证明安装后的 wheel；
- CI 还存在定时测试入口，与本提案需要的日常治理无关。

本提案只恢复一条简单边界：

> 本地用 Unit 和 Integration 证明代码正确；CI 用少量 E2E 证明安装后的产品可用。

## 当前实现基线

本节以 2026-09-01 的 `main` HEAD `cd0b67b` 为迁移起点。以下是代码和配置事实，不是目标状态。

### 已经完成、必须保留的能力

| 现有实现 | 当前路径 | 迁移决定 |
| --- | --- | --- |
| 状态机、reference model、seed、trace、fault/crash point | `tests/qexp_architecture.py` | 保留行为，移动到 `tests/helpers/qexp/` |
| 测试资源命名空间与资源账本 | `tests/qexp_test_support.py` | 保留并强化清理后置检查，移动到 `tests/helpers/qexp/` |
| 单机独立 participant 控制面 | `SingleHostMachineLab` | 保留，仍服务于 Integration |
| Unit 决策证据 | `tests/unit/qexp/test_architecture_primitives.py` | 保留 |
| 扩大 seed 与 crash point 参数化 | `tests/unit/qexp/test_architecture_stress.py` | 保留为普通 Unit，不再拥有定时 lane |
| 真实进程、CAS、fencing 和 cancel/launch 竞态 | `tests/integration/qexp/test_test_architecture.py` | 按行为拆分文件，仍属于 Integration |
| 非 editable wheel E2E | `tox run -e release-e2e`、`tests/e2e/qexp/` | 复用其安装机制建立普通 main E2E |
| 发布完整源码回归 | `scripts/release_preflight.py` 调用 `qexp-full` | 保留，但不再收集生产宿主独占测试 |

这些能力证明单机上的真实进程、文件系统和竞态行为，不证明真实跨主机文件系统语义。

### 当前用例集合

当前环境的 pytest 收集结果如下：

| 集合 | 收集结果 | 当前执行位置 |
| --- | ---: | --- |
| qexp Unit | 183 | 通用 CI Unit 矩阵、`qexp-fast`、`qexp-full` |
| `qexp_fast` Integration | 6 / 351 | `qexp-fast` CI |
| `machine_lab` Integration | 7 / 351 | main push CI |
| `host_exclusive` Integration | 1 / 351 | main push CI |
| 非 `host_exclusive` qexp Integration | 350 / 351 | 完整 qexp 回归 |
| installed-wheel qexp E2E | 7 | tag publish workflow |

一次本地实测中，qexp Unit 为 2.17 秒，6 个 `qexp_fast` Integration 为 5.00 秒，7 个 `machine_lab` Integration 为 16.55 秒。以上仅是单次 pytest 运行，不包含稳定 p95，也不能直接作为最终预算。

### 当前配置漂移

1. `.github/workflows/ci.yml` 在 Python 3.11 至 3.14 的 Unit Job 中重复收集 qexp Unit，`qexp-fast` 又再次收集全部 qexp Unit。
2. `test_architecture_stress.py` 已被 Unit、`qexp-fast` 和 `qexp-full` 收集，定时 `qexp-stress` 再次运行同一批 node ID。
3. `qexp-host-exclusive` 直接从 checkout 导入 `MachineRuntime`，没有从安装 wheel 的公共入口验证产品，因此还不是 E2E。
4. 普通 `e2e` 使用 `usedevelop = true`；只有 `release-e2e` 已具备非 editable wheel 和 `site-packages` 导入检查。
5. `qexp_resource_scope` 当前创建隔离目录和账本，但 fixture 返回后没有统一检查账本中的进程是否已经退出。
6. `tests/readme.md`、`tests/CONTRACT_MATRIX.md`、`tox.ini`、`pytest.ini` 和 ADR-QEXP-0005 仍使用 PR、merge、nightly 和三 profile 语义。

## 已确认约束

1. 开发和自动测试只使用一台 Linux 主机，不建设真实双机环境。
2. 多 machine 行为由单机上的隔离进程、目录和运行时资源模拟。
3. push 前的本地 preflight 是常规源码测试的主要门禁。
4. main CI 在 push 后运行，只能检测并标红，不能阻止 commit 已进入 main。
5. 不修改已经进入开发的 qexp 测试架构 Pitch。

## 目标

1. 全仓库只使用 Unit、Integration、E2E 三个测试层级。
2. 本地 preflight 与 main CI 不重复执行同一个测试用例。
3. qexp 的进程、文件系统和竞态测试保持可隔离、可复现、可清理。
4. CI E2E 必须验证非 editable 安装产物和公共入口。
5. 测试职责可以由目录、配置和自动检查长期守住。

## 非目标

- 不建立 PR-first、merge queue 或远端预合并门禁。
- 不建设真实双机、远端 SSH 或专用分布式测试设施。
- 不增加定时、nightly 或 stress CI 通道。
- 不把完整状态机、竞态或故障组合复制到 E2E。
- 不以覆盖率百分比或持续增加 timeout 代替行为证据。

## 核心架构

```text
tests/
├── unit/          # 纯逻辑和局部行为
├── integration/   # 隔离环境中的组件协作
└── e2e/           # 安装产物的公共用户流程
```

### Unit

Unit 验证不需要真实跨组件资源协作的逻辑，例如状态迁移、调度决策、错误分类和数据模型。

Unit 必须快速、确定，不通过启动真实 qexp agent 或制造真实文件竞争来证明业务规则。

### Integration

Integration 验证源码组件之间的协作，可以使用真实文件系统、锁、tmux 和独立进程，但必须使用每个测试独有的隔离资源。

qexp Integration 包含两类能力：

- 普通隔离测试：验证文件、锁、恢复和组件协作；隔离是默认要求，不再作为独立测试层级。
- `machine_lab`：在单机启动独立参与进程，模拟多 machine 共享项目状态以及竞态。

`machine_lab` 是 Integration 的能力标记，不是第四个测试层级。它允许选择较昂贵的多进程集合，但用例仍位于 `tests/integration/qexp/`。

`qexp_fast_io` 可以继续作为 fixture 行为标记，用于不需要证明物理 `fsync` 延迟的测试。它不能改变断言、隐藏竞态或决定测试层级。

Integration 禁止使用生产默认宿主 authority、默认 tmux socket 或其他可能与开发机真实 qexp agent 冲突的资源。

### E2E

E2E 从干净 checkout 构建非 editable wheel，安装后仅通过公共 CLI 或公开 Python 入口验证少量关键用户流程。

E2E 只证明交付边界：

- wheel 能够安装并包含所需文件；
- entry point 和运行依赖正确；
- 安装后的关键用户流程能够完成；
- 生产默认宿主资源契约在干净 runner 上成立。

`host_exclusive` 不再是测试层级。它只表示某个 E2E 用例需要独占 CI runner 并串行运行。qexp 的宿主全局 authority 验证必须通过安装后的真实入口执行，因此属于 E2E。

锁算法、竞争结果和异常恢复仍由隔离的 Unit 或 Integration 验证，不因宿主 E2E 再次穷举。

## 放置规则

新增测试按以下顺序判断：

1. 是否必须验证安装后的公共产品入口：是则放入 `tests/e2e/`。
2. 否则，是否需要多个组件、真实文件系统或独立进程：是则放入 `tests/integration/`。
3. 其余局部确定性行为放入 `tests/unit/`。

qexp 的目标结构是：

```text
tests/
├── unit/qexp/
├── integration/qexp/
│   ├── test_resource_isolation.py
│   ├── test_store_crash_boundaries.py
│   ├── test_machine_lab.py
│   └── ...
├── e2e/qexp/
└── helpers/qexp/
    ├── architecture.py
    └── resources.py
```

不新增 `tests/host_exclusive/`、`tests/machine_lab/` 或 `tests/hermetic/` 顶层目录。能力与运行约束使用 marker 或 fixture 表达，目录只表达测试层级。

现有 `host_exclusive` 文件不能机械搬迁：隔离环境可证明的锁算法和竞争行为留在 Integration；只有安装产物使用生产默认宿主资源的契约进入 E2E。

## 执行位置

```text
开发阶段       直接相关的 Unit / Integration
push preflight 全部 Unit + 预算内 Integration
main CI        少量安装态 E2E
release        完整源码回归 + 精确发布产物 E2E
```

### 本地 push preflight

普通开发入口统一为：

```bash
tox run -e preflight
```

它运行：

- 全仓库 Unit，一次且只运行一次；
- 当前通用 Integration 和迁移期 `tests/functional/` 常规集合；
- qexp 的资源隔离、durable write crash boundary 和全部 7 个 `machine_lab` 代表用例；
- 测试归属、资源隔离和残留进程检查。

qexp 其余完整 Integration 仍由相关开发改动直接选择，并在 release preflight 中完整运行。preflight 代表集由 `tox.ini` 中的具体测试文件清单管理，不再使用 `qexp_fast` 作为模块专属 CI lane。新增代表文件必须同时满足稳定性、不可由 Unit 替代和实测预算三项条件。

完整的昂贵 Integration 组合不建立定时任务。

### Main CI

普通 main CI 只验证安装产物。它从干净 checkout 构建 wheel，以非 editable 方式安装，并拒绝从仓库 `src/` 导入。

qexp E2E 保持少量并默认串行。带有 `host_exclusive` 标记的用例必须独占 runner；这是 CI 内部调度约束，不对外形成新的测试类别或顶层 Job 体系。

为保留当前 Python 3.11 至 3.14 支持声明的跨版本证据，非 canonical Python 只执行 wheel 安装、import 和 CLI `--help` smoke；canonical Python 3.13 执行完整 E2E。smoke 与完整 E2E 都验证安装产物，不重新运行源码 Unit。

偶发 PR 若继续触发 CI，只复用相同 E2E 定义。CI 不重新运行 Unit 或 Integration，也不设置 schedule 触发器。

### Release

发布验证运行完整 Unit、完整 Integration，并对精确发布 wheel 运行 E2E。发布重复常规源码证据是明确升级，不属于普通 preflight 与普通 main CI 的重复。

## 四条治理原则

### 1. 按验证边界分层

目录只回答“被测边界有多大”：局部逻辑、组件协作或安装后的产品。资源能力和执行位置不创造新层级。

### 2. 最低充分层验证

行为放在能够可靠证明它的最低层。Unit 能证明的状态规则不在 Integration 重复；Integration 能证明的竞态组合不在 E2E 重复。

### 3. 隔离是 Integration 的硬契约

每个 qexp Integration 必须拥有独立临时根、HOME/XDG、tmux socket、进程账本和清理检查。测试结束后的资源残留必须使测试失败，不能由后续测试吸收。

### 4. 常规路径受预算约束

preflight 保留稳定的代表性 Integration；E2E 只保留关键产品流程。超预算时先删除重复证据、减少无意义真实 I/O，或把完整组合保留给明确升级和发布验证，不默认增加 timeout。

## 长期守护

Pitch 只推动实施。方案获批后，仓库级 ADR 记录 main-first 的本地/CI 分工；后继 qexp ADR 保留单机多进程架构，并替换 ADR-QEXP-0005 的 `merge gate` 和三 profile 执行位置。

执行规则由以下受版本控制资产共同守住：

- `tests/unit/`、`tests/integration/`、`tests/e2e/`：测试层级；
- `pytest.ini`：`machine_lab`、`host_exclusive` 和 `qexp_fast_io` 等能力或运行约束；
- `tox.ini`：preflight、E2E 和 release 的收集边界；
- `.github/workflows/ci.yml`：main/PR/tag 触发和 E2E 环境；
- `tests/CONTRACT_MATRIX.md`：关键行为由哪个最低层测试证明。

自动检查至少验证：

1. preflight 只收集 `tests/unit/` 和 `tests/integration/`，main CI 只收集 `tests/e2e/`。
2. `host_exclusive` 只允许出现在 E2E，`machine_lab` 只允许出现在 Integration。
3. Integration 不使用生产默认宿主资源，且测试后没有进程或目录泄漏。
4. E2E 使用非 editable wheel，实际导入路径不指向仓库 `src/`。
5. 普通 CI 不存在 schedule 触发，也不重新收集 Unit 或 Integration。

完成迁移后删除 `hermetic` 和 `qexp_fast` marker：前者已成为所有 Integration 的默认硬约束，后者已由 `tox.ini` 的 preflight 文件清单取代。保留 `machine_lab`、`host_exclusive` 和确有 fixture 语义的 `qexp_fast_io`。

## 失败语义

1. preflight 失败：禁止 push；修复后重新运行完整 preflight。
2. main E2E 失败：main 已标红；保留 wheel、安装日志和运行证据，以后续修复 commit 恢复。
3. `host_exclusive` E2E 发现 runner 已被外部 agent 占用：明确失败，不自动重试制造绿色。
4. release 验证失败：停止发布；普通 preflight 或 main CI 通过不能替代发布证据。

跳过本地 preflight 后直接 push 无法由事后 CI 阻止。这是可信开发者 main-first 流程的显式边界。

## 实施方案

迁移必须按以下顺序实施。旧 lane 只能在对应的新证据入口已经可执行后删除，不允许先删测试再补替代用例。

### Phase 0：冻结现状和一次性测量

- [ ] 保存当前各 tox 入口的 pytest node ID，明确 `qexp-fast` 对 qexp Unit 的重复集合以及 `qexp-stress` 对 Unit 的重复集合。
- [ ] 在目标开发机手动连续运行 20 次候选 qexp preflight 集合：183 个 Unit、6 个资源/持久化 Integration 和 7 个 `machine_lab` 用例。
- [ ] 若已有 `qexp-fast-baseline` artifact，则读取最后一个完整的 20 次结果后删除定时 Job；若没有，则只对本地 preflight 做一次性 20 次测量，不为 E2E 新建临时定时任务。
- [ ] 对已有 20 次集合记录 p50、p95、最大值、失败 node ID 和残留资源；artifact E2E 不足 20 次时先使用保守硬 timeout，并由切换后的前 20 次正常 push 补齐基线。

Phase 0 不修改测试归属。交付物是一次性的基线记录和精确重复 node ID 清单。

### Phase 1：整理测试基础设施并强化隔离

#### 1.0 已完成：选择高效且可退化的测试临时根

- [x] pytest 启动时对 `/tmp` 执行实际的创建、写入和删除探测；探测成功时使用
  `/tmp`，失败时退化到仓库 `tmp/`。
- [x] 在选中的根下为每次 pytest 会话创建唯一目录，避免并行或异常退出后的会话互相
  复用 node ID 路径。
- [x] `QQTOOLS_PRESERVE_TEST_ARTIFACTS=1` 时保留会话证据；默认在用例和会话结束时清理。
- [x] 为系统根优先、系统根不可用时退化，以及两个根都不可用时明确失败增加 Unit 证据。

选择只在 pytest 启动时执行一次，不在每个用例中重复探测，也不把 `/tmp` 可用性作为
环境假设。2026-09-01 的一次实现前后测量中，原 `qexp-integration` 在仓库所在 9p 文件
系统耗时约 506 秒；使用上述选择逻辑后，完整收集执行到末尾耗时 51.34 秒，其中 350 个
用例通过，唯一失败是 `host_exclusive` 用例先启动两个进程再假定第一个必先持锁的测试
编排竞态。该用例改为确认第一个持锁后再启动竞争者，随后单独验证通过。最终完整绿色
耗时的 p50/p95 仍按 Phase 0 的连续基线补充，不能由这一次测量代替。

#### 1.1 移动共享测试代码

| 当前路径 | 目标路径 | 要求 |
| --- | --- | --- |
| `tests/qexp_architecture.py` | `tests/helpers/qexp/architecture.py` | 只移动测试模型和 `SingleHostMachineLab`，不进入生产包 |
| `tests/qexp_test_support.py` | `tests/helpers/qexp/resources.py` | 保留 `TestResourceScope`，不增加生产配置开关 |

- [x] 新增 `tests/helpers/qexp/__init__.py`，更新 Unit、Integration 和 fixture 的导入路径。
- [ ] 对移动后的 Unit 与 Integration 做 collect-only，再运行直接相关测试，保证目录迁移不改变行为。

#### 1.2 拆分现有架构 Integration

将 `tests/integration/qexp/test_test_architecture.py` 按行为拆成：

| 目标文件 | 从当前文件迁入的证据 |
| --- | --- |
| `test_resource_isolation.py` | 资源 scope、账本、独立 authority root，以及改造后的测试内 authority 互斥 |
| `test_store_crash_boundaries.py` | `file_fsync`、`replace`、`directory_fsync` 三个真实崩溃边界 |
| `test_machine_lab.py` | 当前全部 7 个 `machine_lab` 用例 |

文件按验证行为拆分，不按 CI Job 命名。`machine_lab` marker 保留；`hermetic` 和 `qexp_fast` 暂不在本阶段删除，直到新 preflight 已接管选择责任。

#### 1.3 让清理失败成为测试失败

- [ ] 将 `qexp_resource_scope` 改为 yield fixture，在 teardown 检查账本中登记的 PID/PGID 已退出。
- [ ] `SingleHostMachineLab.close()` 必须等待所有 participant；terminate 超时后只 kill 账本登记且由测试创建的进程组。
- [ ] participant 未退出、存在未消费 cleanup diagnostic 或测试拥有的 tmux socket 仍活跃时，使当前用例失败并保留诊断路径。
- [ ] 禁止通过进程名、用户级全量扫描或默认 tmp 根清理资源；清理范围只能来自本测试的 resource ledger。

Phase 1 完成后，除待迁移的现有 `host_exclusive` 用例外，所有 qexp Integration 均可在开发机存在真实 qexp agent 时安全运行。

### Phase 2：拆分生产 authority 证据

当前 `test_host_exclusive_authority_allows_only_one_runtime_per_os_user` 同时承担锁算法和生产默认路径证据，必须拆成两个不同层级的用例。

#### 2.1 隔离 Integration

- [ ] 在 `test_resource_isolation.py` 中让两个真实 participant 共享同一个测试专属 `TMPDIR`，但使用不同 runtime root。
- [ ] 断言第一个 participant 持有 authority、第二个被拒绝；第一个退出后第三个能够取得 authority。
- [ ] 该用例不得读取操作系统默认 tmp，不使用 `host_exclusive`，也不因开发机已有 agent 而 skip。

该用例证明锁互斥和释放算法。

#### 2.2 安装态 E2E

- [ ] 新增 `tests/e2e/qexp/test_host_authority.py`，标记 `host_exclusive`。
- [ ] 使用安装 wheel 的 `qexp` CLI，针对两个独立 project/runtime root 启动 agent；两个子进程保持同一 OS user，并移除 `TMPDIR`、`TMP`、`TEMP` 的测试覆盖，使生产 resolver 使用 clean runner 的默认 authority namespace。
- [ ] 断言第二个 agent 通过公共 CLI 返回明确的占用失败；停止第一个 agent 后，第二个能够启动。
- [ ] `finally` 中只停止本用例创建的 agent，并保留失败时的 CLI 输出、PID/PGID 和 authority 诊断。
- [ ] E2E 不得导入 `MachineRuntime` 来代替公共产品入口。

同一提交中删除 Integration 中原有的 `host_exclusive` 用例和 skip/fail 探针，确保一个行为只有一份最终证据。

### Phase 3：建立目标 tox 入口

#### 3.1 `preflight`

在 `tox.ini` 新增 canonical Python 3.13 的 `preflight`。它按顺序执行：

1. 全仓库 `tests/unit/`，不再额外调用 `qexp-unit`。
2. 当前通用 `integration` 集合和迁移期 `tests/functional/` 集合。
3. `test_resource_isolation.py` 与 `test_store_crash_boundaries.py`。
4. `test_machine_lab.py` 的全部 7 个用例。
5. 测试 lane、导入路径和资源清理检查。

qexp 的其余 Integration 不进入普通 preflight；开发时按改动行为直接运行，release preflight 通过 `qexp-full` 完整运行。新增 qexp preflight 代表文件必须修改 `tox.ini`，不能仅靠默认 pytest 收集静默扩大门禁。

#### 3.2 `artifact-e2e`

- [ ] 以现有 `release-e2e` 为模板新增普通 main 的 `artifact-e2e`：`package = wheel`、`usedevelop = false`、清空 `PYTHONPATH`。
- [ ] 将 `tests/e2e/release_pytest.ini` 泛化为 installed-artifact pytest 配置，供 main 和 tag workflow 共同使用。
- [ ] 把 `ensure_site_packages_import()` 提升为 installed E2E session 级前置条件，使所有 E2E 在从 checkout `src/` 导入时立即失败。
- [ ] canonical Python 3.13 运行 `tests/e2e/` 的完整公共流程；其他声明支持的 Python 版本只安装 wheel 并执行 import/CLI smoke。
- [ ] `release-e2e` 继续支持 `--installpkg <exact-wheel>`，tag workflow 仍验证精确构建物。

#### 3.3 保留与删除的 tox 入口

| 入口 | 最终处理 | 原因 |
| --- | --- | --- |
| `qexp-unit` | 保留 | 本地模块快捷入口 |
| `qexp-integration` | 保留 | 直接升级到完整 qexp Integration |
| `qexp-machine-lab` | 保留 | Integration 能力快捷入口，不进入 CI |
| `qexp-full` | 保留 | release preflight 的完整源码证据 |
| `qexp-fast` | 删除 | 被唯一 `preflight` 取代，且当前重复 qexp Unit |
| `qexp-host-exclusive` | 删除 | authority 产品契约已进入 installed E2E |
| `qexp-stress` | 删除 | 文件保留为 Unit，永久定时 lane 被取消 |
| `e2e` | 由 `artifact-e2e` 取代 | 当前 editable 环境不能证明交付物 |

`test_architecture_stress.py` 的 64 seeds 和全部稳定 crash point 保留在 Unit。当前 qexp Unit 单次实测仅 2.17 秒，因此先移除其 `slow` marker 并随 Unit 运行；若后续实测使整个 preflight 超预算，再基于证据重新评审，不预先恢复定时 lane。

### Phase 4：原子切换 CI

修改 `.github/workflows/ci.yml` 时，在一个提交内完成以下切换：

- [ ] 删除 `schedule` 触发器、`qexp-fast-baseline` 和 `qexp-stress`。
- [ ] 删除普通 CI 的 `unit`、`integration`、`qexp-fast` 和 `qexp-machine-lab` 源码 Job。
- [ ] 新增 installed wheel smoke：Python 3.11、3.12、3.14 只验证安装、import 和 CLI 启动。
- [ ] Python 3.13 运行一次完整 `artifact-e2e`，其中 qexp 用例串行执行，`host_exclusive` 使用干净 runner。
- [ ] push main 与偶发 PR 复用同一套 artifact 定义，不创建 PR 专属测试规则。

切换前必须先证明 `tox run -e preflight` 在开发机通过，并证明 `tox run -e artifact-e2e` 在干净 runner 通过。main CI 切换后，普通 CI 的 pytest node ID 与 preflight 的 pytest node ID 交集必须为空。

### Phase 5：删除过渡语义并固化规则

- [ ] 从 `pytest.ini` 删除 `hermetic` 和 `qexp_fast`；保留 `machine_lab`、仅限 E2E 的 `host_exclusive` 和 fixture 仍使用的 `qexp_fast_io`。
- [ ] 从 `tests/integration/qexp/conftest.py` 删除 host probe、local skip 和自动添加 `hermetic` 的 hook；隔离环境改为所有 qexp Integration 的无条件 fixture。
- [ ] 新增 `scripts/checks/check_test_lanes.py`，并让 preflight 与 artifact CI 都执行该检查。
- [ ] 将存量 `tests/functional/` 用例迁入正确的 Integration 或 E2E 目录；迁移完成前 `preflight` 显式运行它，完成后删除迁移期 `full` tox 入口。
- [ ] 更新 `tests/readme.md`、`tests/CONTRACT_MATRIX.md`、测试目录规范和本地 verification workflow，删除 PR/merge/nightly 表述。
- [ ] 创建仓库级 ADR，并创建后继 qexp ADR supersede ADR-QEXP-0005；原 qexp Pitch 保持不动。

lane checker 必须失败于以下情况：Integration 出现 `host_exclusive`、E2E 出现 `machine_lab`、普通 CI 收集源码测试、preflight 收集 E2E、installed E2E 从 `src/` 导入、workflow 重新出现 schedule，或同一 node ID 同时进入普通 preflight 与普通 CI。

Phase 5 交付物是可审计、可执行且不依赖 Pitch 记忆的长期规则。

## 验证用例

| ID | 场景 | 操作 | 预期结果 |
| --- | --- | --- | --- |
| TG-01 | 正常 push | 本地运行 preflight 后 push main | 本地完成 Unit/Integration；CI 只运行安装态 E2E |
| TG-02 | Integration 越界 | Integration 使用生产默认宿主 authority | 自动检查失败并指出违规用例 |
| TG-03 | marker 越层 | 在 Integration 标记 `host_exclusive` | 收集或治理检查失败 |
| TG-04 | 错误打包 | wheel 遗漏 qexp 运行文件 | 源码测试可以通过，但 CI E2E 失败 |
| TG-05 | editable 泄漏 | E2E 从仓库 `src/` 导入 | E2E 失败并报告实际导入路径 |
| TG-06 | 多进程竞态 | 运行 `machine_lab` 代表场景 | 独立进程共享项目状态，断言唯一结果并完成清理 |
| TG-07 | 宿主互斥 | 两个安装态 qexp agent 竞争默认 authority | E2E 观察到唯一持有者及明确失败诊断 |
| TG-08 | 非 qexp 改动 | 执行同一 preflight/CI 流程 | 复用三层规则，不增加 qexp 专属仓库级流程 |
| TG-09 | 清理失败 | participant 忽略正常退出和 terminate | 只 kill 本测试账本登记的进程组，当前测试失败并保留诊断 |
| TG-10 | CI 拓扑回退 | workflow 新增 schedule 或源码 pytest Job | lane checker 失败 |

## 验收标准

1. 测试目录只使用 Unit、Integration 和 E2E 三个正式层级。
2. 普通 preflight 只执行 Unit 和 Integration；普通 main CI 只执行安装态 E2E。
3. qexp Integration 全部使用隔离资源，`machine_lab` 仍属于 Integration。
4. `host_exclusive` 只作为 E2E 内部运行约束，不成为独立测试类别。
5. E2E 从非 editable wheel 运行公共入口，且不重复内部状态和竞态组合。
6. 普通 CI 不再运行 Unit、Integration 或定时任务。
7. qexp 不拥有永久独立的仓库级源码 CI Job。
8. preflight 与 CI 的时长预算来自目标环境实测并由配置约束。
9. `qexp-fast`、`qexp-host-exclusive` 和 `qexp-stress` tox 入口已经删除；保留的能力均有新入口。
10. qexp 测试共享基础设施位于 `tests/helpers/qexp/`，且 participant 泄漏能够使所属测试失败。

## 假设与未验证

- **未验证**：使用公共 CLI 的 installed-wheel host authority E2E 尚未实现；当前证据仍是直接导入 `MachineRuntime` 的 Integration。
- **未验证**：7 个 `machine_lab` 用例单次实测为 16.55 秒，但其 p95 和不稳定率尚需一次性连续基线确认。
- **未验证**：普通 preflight 和安装态 E2E 的合理硬时长上限尚无目标环境实测数据。
- **未验证**：现有 qpipeline E2E 从 editable 环境迁移到 installed wheel 后是否需要额外 package data 或依赖调整，尚需在 Phase 3 验证。
- **假设**：生产默认宿主 authority 只有在安装态公共入口中验证才具有独立产品证据价值；如果隔离 Integration 已能完整证明该契约，应删除对应 E2E，而不是重复保留。

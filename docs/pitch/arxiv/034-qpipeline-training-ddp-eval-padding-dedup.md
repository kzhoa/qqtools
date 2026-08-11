---
doc_type: pitch
status: archived
updated_at: 2026-08-11
archived_at: 2026-08-11
---

# qpipeline 训练期 DDP 评估 Padding Dedup

## 背景

`qpipeline` 已在独立 `evaluate_once()` 和 `infer_only()` 路径支持 DDP sampler padding
去重，但训练期间由 `RunningAgent` 触发的周期性 validation/test 尚未接入该能力，也没有在进入
评估循环前验证各 rank 的 batch 数是否一致。

真实故障场景：

- validation set `val_id` 包含 394,727 个结构；
- 使用 8 个 rank；
- `eval_batch_size=2`；
- 一个全局 chunk 包含 `8 × 2 = 16` 个样本；
- `394727 % 16 = 7`。

现有 `OC22BalancedBatchSampler` 对尾部 7 个样本执行 LPT 负载均衡，将它们分配给 7 个
rank，剩余 1 个 rank 没有尾部样本。最终：

- rank 0–6 各执行 24,671 个 eval batch；
- rank 7 执行 24,670 个 eval batch。

训练期评估循环会在每个 eval batch 内执行 DDP metric collective。rank 7 先退出循环后，其余
rank 仍在最后一个 batch 内进入 collective，导致 collective 次序错位并最终超时。

该问题包含两部分责任：

- sampler 未保证 DDP eval 各 rank 等步，违反分布式评估输入契约；
- qpipeline 未在评估前验证契约，而是允许运行进入不可恢复的 collective hang，属于框架健壮性
  缺陷。

相关既有设计：

- [qpipeline DDP Eval/Infer Dedup](./arxiv/017-qpipeline-ddp-eval-output-dedup.md)
- [qpipeline 单/多验证集与测试集 Loader 契约](./arxiv/022-qpipeline-single-multi-eval-loader-contract.md)

## 目标

本需求必须达成：

- 将已有 DDP eval padding dedup 能力覆盖到训练期间的周期性 validation/test；
- 对动态持有的单个或多个 eval loader，在每次 evaluation boundary 建立本次 pass 的局部快照；
- 在训练会话首次实际触发 evaluation 时、进入每个 eval loader 的 batch loop 前验证所有 rank 的执行步数一致；
- 对首次 evaluation 中不合法的 DDP eval plan fail-fast，禁止以 collective timeout 的方式暴露问题；
- 使用单一、扁平的 runner 配置控制所有 qpipeline eval/infer 路径；
- 保留 graph 输出维度校验，同时移除重复的 `is_graph` YAML 配置。

## 非目标

本期不负责：

- 由 runner 自动为不等步 sampler 合成 padding batch；
- 修改训练集 sampler 行为；
- 支持 `IterableDataset` 的全局逻辑样本 dedup；
- 将 intentional repeated sampling 转换为 unique-sample evaluation；
- 为每个动态 loader name 提供独立的 dedup 配置；
- 改变用户 task 的 metric、cache 或模型 forward 业务语义。

## 配置契约

### Runner 配置

新增唯一开关：

```yaml
runner:
  ddp_eval_dedup: true
```

规则：

- 类型为 `bool`；
- 默认值为 `true`；
- DDP 模式下，`true` 表示从 metric、cache 和 output 收集中排除 sampler padding 产生的
  重复逻辑样本；
- DDP 模式下，`false` 表示保留 sampler 产生的全部 occurrence；
- 非 DDP 模式下该字段为 no-op；
- 无论该字段取值为何，训练会话首次实际 evaluation 的 DDP eval 各 rank 等步检查始终启用，不提供关闭安全检查的配置。

### Breaking change

删除以下旧配置，不提供兼容读取、迁移别名或 deprecation 周期：

```yaml
task:
  eval:
    ddp_dedup:
      enabled: true
      is_graph: false
      node_aligned_output_keys: []
```

同时删除对应的公开配置类型与 schema：

- `EvalConfig`；
- `EvalDDPDedupConfig`；
- `TaskConfig.eval`；
- qConfigGen 中的 `task.eval` 生成逻辑。

旧配置进入新版 schema 时必须被拒绝，不能被静默忽略。

## Graph 输出契约

### Graph 模式来源

`is_graph` 不再由 qConfig 重复声明。框架按以下优先级读取当前 loader 的 graph 模式：

1. `loader.is_graph`；
2. `loader.dataset.is_graph`；
3. 默认 `False`。

`qDictDataloader(..., is_graph=...)` 必须持久化公开的 `loader.is_graph` 属性。loader 级显式值
优先于 dataset 值，以支持 per-loader override。

框架不得通过 batch 内容或输出 tensor shape 猜测 graph 模式。

### Node-aligned 输出

保留 graph 输出维度校验，并将 node-aligned 输出声明改为 task 下的扁平字段：

```yaml
task:
  node_aligned_output_keys:
    - forces
```

规则：

- 默认值为空列表；
- 仅影响需要收集输出的独立 eval/infer 路径，不影响 metric dedup；
- graph loader 中，未声明的输出默认视为 sample-aligned，其 leading dimension 必须等于去重后的
  真实逻辑样本数；
- `node_aligned_output_keys` 中的字段允许 leading dimension 为节点数，不执行 sample-count
  断言；
- 该校验不负责验证节点总数或结构到节点的归属关系。

## Sampler 与框架责任边界

### Sampler 必须保证

对每个独立 eval loader，sampler 必须保证：

- 所有 rank 产生相同数量的非空 batch；
- 所有 rank 的采样计划合并后完整覆盖目标 eval set；
- `ddp_eval_dedup=true` 时，同一个逻辑 idx 的额外 occurrence 只能来自 DDP padding；
- sampler/batch_sampler 有限、可枚举，且当前 pass 的 batch plan 可以被框架安全观测；
- sampler idx 在各 rank 上具有相同的逻辑含义，并可用于对应 dataset 的 `__getitem__`；
- `len(loader)` 与实际执行计划一致；启用 dedup 时，以 materialized plan 为最终检查依据。

原生 `torch.utils.data.DistributedSampler(drop_last=False)` 通过重复 idx 补齐各 rank 的样本数，
因此满足等步要求。自定义 balanced sampler 必须在自身计划中完成 padding，因为只有 sampler 掌握
样本代价、LPT 分配及可接受的重复策略。

### qpipeline 必须保证

qpipeline 必须：

- 在训练会话首次实际 evaluation 中、进入 eval batch loop 前同步并检查 rank-local plan；
- 保持 padding 所需的 forward step 数，避免破坏 DDP 模型或 runner collective 时序；
- mixed batch 中只将真实样本送入 metric/cache/output 处理；
- all-duplicate batch 仍执行一次 forward，但不更新 metric/cache/output；
- 在 loader 自然完成后校验收集到的真实逻辑 idx 与同步计划一致；
- 对不支持的 loader 类型或不可观测 sampler 抛出包含 stage、loader name 和原因的异常。

runner 不得为少步 rank 自动复制任意样本。框架无法可靠决定 dummy 样本的计算代价、索引可见性、
collate 语义或 intentional repeat 语义，自动补齐会隐藏 sampler 契约错误。

## 动态与多 Eval Loader 生命周期

`task.val_loader` 和 `task.test_loader` 可以在 evaluation boundary 之间动态变化，并可以分别是
单个 `DataLoader` 或命名 `dict[str, DataLoader]`。

每次 evaluation pass 必须执行：

1. 重新读取并解析当前 `task.val_loader` 和 `task.test_loader`；
2. 建立本次 pass 的 `(stage, loader_name, loader)` 局部快照；
3. 在所有 rank 间同步 loader manifest，验证 stage 和 loader name 集合一致；
4. 使用稳定顺序执行 loader，避免 dict 插入顺序差异改变 collective 次序；
5. 为每个 `(stage, loader_name)` 独立构建 dedup runtime 和采样计划；
6. 仅在训练会话首次含 eval loader 的 pass 中，验证该 loader 在各 rank 上的 batch 数一致；
7. standard model 与 EMA model 复用同一份 pass-local prepared loader snapshot；
8. evaluation pass 结束后丢弃快照，下一次 boundary 重新解析。

dedup seen-set 不得跨 loader、stage 或 evaluation pass 共享。两个不同 eval set 中相同的 idx 值
不代表同一个逻辑样本。

## 训练期评估行为

训练期 `_evaluate_loader()` 必须与独立 `evaluate_runner()` 共享以下行为：

- 解包 dedup control；
- all-duplicate batch 不更新 `AvgBank` 或 `TensorBank`；
- 最终 metric 使用 dedup-aware gather，使没有真实样本贡献的 rank 以零 sum、零 count 参与；
- epoch metric 的 cache gather 能处理某些 rank 本地 cache 为空的情况；
- progress metric collective 的次数在所有 rank 上一致；
- metric key 在 rank 间不完全一致时，不得通过遍历本地 key 集合执行不对称 collective。

如果 `batch_metric()` 返回 batch mean，task 应通过 `(value, real_sample_count)` 提供权重。未提供
count 时，`AvgBank` 只能按 batch 等权聚合，这一既有语义不由本需求改变。

## 异常行为

### Loader topology 不一致

如果不同 rank 解析出的 stage/loader name manifest 不一致，所有 rank 必须在进入任何 loader-specific
collective 前失败。异常至少包含各 rank manifest。

### Batch 数不一致

如果同一 loader 的各 rank batch 数不一致，异常至少包含：

- stage；
- loader name；
- world size；
- 每个 rank 的 batch 数；
- sampler 必须自行 padding 或使用 `drop_last=True` 的修复提示。

示例：

```text
DDP evaluation loader batch-count mismatch: stage=val, loader=val_id,
counts={0: 24671, 1: 24671, 2: 24671, 3: 24671,
        4: 24671, 5: 24671, 6: 24671, 7: 24670}.
All ranks must execute the same number of eval batches.
Pad the sampler plan or use drop_last=True.
```

### 不支持的 loader

启用 dedup 时，以下输入必须 fail-fast：

- `IterableDataset`；
- sampler 和 batch_sampler 均不可安全观测；
- batch 边界无法重建；
- sampler idx 不能作为逻辑样本标识使用。

## 非功能性要求

- 不永久修改 task 持有的原始 DataLoader；
- 不在训练初始化阶段缓存动态 eval loader；
- 不新增第三方依赖；
- 错误必须在进入可能失配的 batch-level collective 前暴露；
- 配置、qConfigGen schema、README 和实现必须保持一致；
- 独立 eval/infer 与训练期 eval 不得维护两套配置解析逻辑。

## 验收标准

- `runner.ddp_eval_dedup` 能统一控制训练期 eval、`evaluate_once()` 和 `infer_only()`；
- 新版 schema 接受 `runner.ddp_eval_dedup` 并拒绝 `task.eval`；
- 原生 `DistributedSampler(drop_last=False)` 的 padding occurrence 不进入最终 metric/output；
- `ddp_eval_dedup=false` 时重复 occurrence 保留，但训练会话首次实际 evaluation 的各 rank 等步检查仍然执行；
- OC22 示例中的 `{0..6: 24671, 7: 24670}` 在第一个 eval batch forward 前明确报错；
- 动态单 loader、多 loader、val/test 混合场景均按本次 pass 快照执行；
- standard/EMA 使用同一份 prepared loader snapshot；
- graph loader 不需要额外 YAML `is_graph`，且错误的 sample-aligned 输出首维仍会被拒绝；
- 非 graph、单 rank路径保持原有 metric/output 行为。

## 实施方案

### Phase 1：Breaking config

- [ ] 在 `RunnerConfig`、内部 `RunConfig` 和 runner schema 中增加 `ddp_eval_dedup: bool = True`。
- [ ] 在 `TaskConfig` 中增加 `node_aligned_output_keys: List[str]`。
- [ ] 删除 `EvalConfig`、`EvalDDPDedupConfig`、`TaskConfig.eval` 及全部旧 schema/generator 输出。
- [ ] 让旧 `task.eval` 配置被 schema 明确拒绝。
- [ ] 统一训练、独立 eval 和 infer 的新配置读取路径。

### Phase 2：Loader graph metadata

- [ ] 让 `qDictDataloader` 持久化 `is_graph` 属性。
- [ ] 实现 loader 优先、dataset 次之的 graph mode 解析。
- [ ] 将 graph 输出校验改读 `task.node_aligned_output_keys`。

### Phase 3：动态 loader 准备与安全检查

- [ ] 每个 evaluation boundary 生成 val/test pass-local snapshot。
- [ ] 在所有 rank 间同步并验证 loader manifest。
- [ ] 对每个 loader 构建独立 dedup runtime，并在首次实际 evaluation 验证 materialized batch plan 等步。
- [ ] 未启用 dedup 时仍在首次实际 evaluation 对每个 loader 执行 batch-count 检查。
- [ ] standard/EMA 复用同一 prepared snapshot。

### Phase 4：训练期 dedup 集成

- [ ] 在 `RunningAgent._evaluate_loader()` 中接入 batch control。
- [ ] all-duplicate batch 保留 forward step，但跳过 metric/cache 更新。
- [ ] 使用 dedup-aware metric/cache gather，并保证 progress collective 对称。
- [ ] 保持独立 eval/infer 的既有 dedup 行为。

### Phase 5：文档与验证

- [ ] 更新 qpipeline README、CHANGELOG 和配置示例。
- [ ] 增加 breaking config、动态多 loader、rank 不等步、训练期 metric/cache dedup 回归测试。
- [ ] 运行 qpipeline、qConfigGen、DDP deduper 和 qDictDataloader 相关测试。

## 验证用例

| ID | 场景 | 输入或操作 | 预期结果 |
| :--- | :--- | :--- | :--- |
| TC-01 | 新配置默认值 | 未显式设置 `runner.ddp_eval_dedup` | DDP eval 默认启用 padding dedup |
| TC-02 | Breaking schema | 配置包含 `task.eval.ddp_dedup` | 配置校验失败，不做兼容转换 |
| TC-03 | 原生 sampler padding | dataset 长度不能被 world size 整除 | 各 rank 等步，重复 idx 不进入 metric/output |
| TC-04 | 不等步 sampler | batch counts 为 `24671 × 7 + 24670 × 1` | 首个 eval forward 前抛出含逐 rank counts 的异常 |
| TC-05 | Dedup 关闭 | `runner.ddp_eval_dedup=false` 且 sampler 等步 | 重复 occurrence 保留，评估正常完成 |
| TC-06 | Dedup 关闭但不等步 | 开关为 false，rank batch 数不一致 | 仍然 fail-fast |
| TC-07 | 动态 loader | 两次 boundary 返回不同 loader snapshot | 每次重新解析，各 pass 内保持稳定 |
| TC-08 | 多 eval set | val/test 各包含多个命名 loader | 首次 pass 中每个 `(stage, name)` 独立检查；每个 pass 独立 dedup |
| TC-09 | EMA | 同一 boundary 评估 standard 和 EMA | 两个模型复用同一 prepared snapshot |
| TC-10 | Graph sample output 错位 | graph loader 返回首维为 node count 的未声明输出 | 输出维度校验失败 |
| TC-11 | Graph node output | `forces` 被声明为 node-aligned | 不执行 sample-count 断言，输出正常收集 |
| TC-12 | IterableDataset | DDP dedup 启用且 eval loader 为 iterable | 进入 batch loop 前报不支持错误 |

## 假设与未验证

- 假设：使用 intentional repeated sampling 的 eval sampler 可以通过
  `runner.ddp_eval_dedup=false` 保留既有语义，但仍能满足各 rank 等步约束。
- 未验证：仓库外部用户是否已使用 `task.eval.ddp_dedup.is_graph` 或
  `node_aligned_output_keys`；本需求已明确选择 breaking change，不提供兼容迁移。
- 未验证：所有仓库外自定义 graph DataLoader 是否暴露 `is_graph`。不暴露该属性且 dataset 也无
  `is_graph` 时，框架按非 graph loader 处理，不执行 graph 输出维度校验。

## 验收清单

- [ ] 配置检查：新字段生成、解析和 schema 校验一致。
- [ ] 逻辑验证：训练、独立 eval、infer 三条路径使用同一 dedup policy。
- [ ] DDP 安全：动态多 loader topology 和逐 loader batch counts 在 forward 前验证。
- [ ] 数据正确性：padding occurrence 不污染 metric/cache/output。
- [ ] Graph 输出：自动 graph mode 与 node-aligned key 声明覆盖现有校验能力。
- [ ] Breaking change：旧 `task.eval` 被完整删除且不兼容。
- [ ] 文档同步：README、CHANGELOG 和配置示例反映最终实现。

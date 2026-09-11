---
doc_type: pitch
status: archived
updated_at: 2026-09-12
archived_at: 2026-09-12
---

# qPipeline 分布式评估与推理的平衡采样自动补齐

## 背景

`BalancedDistributedSampler` 能按样本计算成本平衡各 rank，但在确定性评估中，
当样本数不能被 `world_size × batch_size` 整除时会直接失败。qPipeline 的
`DDPOutputDeduper` 虽然能够消除 DDP padding 对 metric 和输出的影响，却只能在
loader 创建成功后运行，因而无法处理 sampler 构造阶段的失败。

这会导致一个实际的用户问题：任务在 2 卡上训练时自行创建了 `val_loader`，设置
`pad=False`；训练完成后，用户希望在 3 卡和新的验证集上直接调用 `infer()`，却因
rank step 数不一致而无法开始推理。用户不应为了改变卡数或数据集而重写 loader。

## 目标

- 用户继续直接调用现有的 `evaluate()` / `infer()` API，无需填写 `purpose`、
  `deduplicate` 或手工创建第二套 loader。
- `BalancedDistributedSampler` 增加显式 `pad` 参数，行为与 PyTorch 原生
  `DistributedSampler` 的补齐语义一致。
- sampler 只负责生成平衡且可执行的 index 序列，不持有 dedup 逻辑。
- qPipeline 在执行评估/推理前自动检查 rank step 是否一致；必要时为用户提供的
  loader 创建临时的 padded execution loader。
- padding occurrence 只在物理执行层存在，并由现有 `DDPOutputDeduper` 在结果收集
  层按稳定样本身份去重。
- 原始用户 loader 和 sampler 不被修改，训练行为保持不变。

## 用户可见契约

```python
BalancedDistributedSampler(dataset, ..., pad=False)
```

- `pad=True`：补齐各 rank 的样本数量，使 DDP step 数一致。
- `pad=False`：不补齐，不丢弃真实样本。
- `pad=True` 与 `drop_last=True` 同时设置时拒绝构造。

`pad` 的默认值保持向后兼容。用户未显式指定时，qPipeline 可以在评估/推理执行层
临时采用 padding；用户显式写入 `pad=False` 不应阻止 qPipeline 为 DDP 对齐创建临时
执行视图，也不改变最终 metric/output 的逻辑样本集合。

## 设计

1. `BalancedDistributedSampler` 支持 `pad=True/False`，并暴露足够的采样元数据或
   原始 index，使上层能够识别 padding occurrence。sampler 不依赖 qPipeline，也不
   执行去重。
2. qPipeline 在 `evaluate()` / `infer()` 开始前检查用户提供 loader 的每 rank step
   数。若已经一致，直接使用原 loader；若不一致，则复用 sampler 已经生成的 rank
   分配，只在较短 rank 的尾部执行序列中确定性重复已有 index，创建临时的 padded
   execution loader。该过程不重新读取完整 cost 文件、不重新排序，也不改变既有的
   cost balancing 结果。
3. 临时执行 loader 完成 forward 后，`DDPOutputDeduper` 按稳定 sample identity
   去重，再计算 metric 或写出结果。
4. 若自定义 sampler 无法重建、无法提供稳定样本身份，pipeline 在 forward 前给出
   明确错误；不得让 collective 在运行中失步，也不得静默丢样本或重复计数。

## 非目标

- 不把 dedup 职责下沉到 sampler。
- 不要求用户为 train、val、test 或 infer 填写新的 purpose 参数。
- 不默认启用 uneven-input/`Join` 机制。
- 不修改用户原始 loader 的配置、顺序或训练语义。
- 不为了 DDP 尾部补齐重新读取完整 cost 数据或执行全局重平衡。

## Padding 与负载平衡

自动补齐的目标是统一 DDP step 数，而不是重新优化整个数据集的 cost 分配。原有
`BalancedDistributedSampler` 已经完成的 rank balancing 应被完整复用；通常只有最后
一个物理 batch 会因为重复 occurrence 产生有限的 cost 差异。

补齐策略必须满足：

- 仅补齐到所有 rank 的最大 batch 数；
- 只在尾部追加确定性重复 index；
- 补齐 occurrence 带有稳定 sample identity 和 padding 标记；
- 不丢弃任何逻辑样本；
- 不将 padding occurrence 写入最终 output 或计入 metric。

如果自定义 sampler 产生的 rank 长度差异异常大，导致需要复制大量样本，pipeline 应在
forward 前报告诊断信息或拒绝自动适配；常规的非整除尾 batch 不应触发完整 cost 重算。

## 验收标准

- 非整除数据集在 2 卡切换到 3 卡后可直接 `infer()`。
- 所有 rank 执行相同数量的 forward step。
- 自动补齐不读取完整 cost 文件、不重新排序，且保留原有 rank 的主要 cost balancing。
- padding 前后，MAE 等 metric 与单卡全量结果一致。
- 输出中每个逻辑 sample identity 最多出现一次。
- `pad=False` 的用户 loader 在训练路径上保持原有行为。
- 自定义 sampler 能力不足时，在执行前报告可操作的诊断信息。

**假设/未验证：** 当前 `DDPOutputDeduper` 是否已经能从 loader/batch 获取稳定的原始
样本 identity，需要实现前核对；若不能，应优先补齐 identity 传递，而不是在 sampler
中加入去重逻辑。


# general position

`.pipeline` 是 `.torch` 的 消费者。


# cmd_args

标准args通过 `python xx.py --config /path/to/config.yml`的方式，指定config文件。
在py文件中通过这个函数加载config文件中的内容。
`args = prepare_cmd_args()`

若是希望通过 cmd line 自定义更多其它参数，可以通过 `patch` 回调向基础 parser 注册额外参数。
此外，未被显式 parser 消费的 dotted CLI 参数会作为配置树覆盖项处理，例如
`--task.dataloader.eval_batch_size 32`。

覆盖规则：

- 显式 parser 参数和 `patch` 注册参数优先被 `argparse` 正常消费
- 未消费的 `--a.b.c value` / `--a.b.c=value` 会在 YAML 加载后写入最终 `args`
- dotted override 优先级最高，高于 YAML 和基础显式参数合并结果
- flag-only 形式 `--a.b.c` 只允许用于布尔安全目标
- 负值或以 `-` 开头的值必须使用等号形式，例如 `--task.threshold=-0.5`

example:
```python
from qqtools.pipeline import prepare_cmd_args

def patch(parser: argparse.ArgumentParser):
    parser.add_argument("--file", type=str)
    return parser

args = prepare_cmd_args(patch=patch)
file = args.file
```

example:
```bash
python train.py \
  --config configs/train.yaml \
  --task.dataloader.eval_batch_size 32 \
  --task.val_split val_ood \
  --runner.fast_dev_run
```


# trainPipeline



## pipeline init
有两种init方式。

第一种直接传入实例化的task和model。

```python
model = prepare_model(args)
task = prepare_task(args)
pipe = QPipeline(args, mode="train", task=task, model=model)
```



在每个训练脚本里面继承，
子类只需要实现prepare_task和prepare_model方法。

```python
class MyPipeline(QPipeline):
    @staticmethod
    def prepare_task(args):
        pass

    @staticmethod
    def prepare_model(args):
        pass

# or

class MyPipeline(QPipeline):
    prepare_model = staticmethod(prepare_model)
    prepare_task = staticmethod(prepare_task)


pipe = MyPipeline(args, mode="train")
```

这两个方法只接受一个标准args入参 （后面会解释为什么是标准args）
其中model就是我们理解的model。
需要在这里完成model.to(device)操作。
task见下一节。


## pipeline cache

```python
def regist_extra_ckp_caches(self, caches: dict):
```
接受传入一个字典，保存本次任务的全局信息。
这个字典中的内容会存入每个checkpoint文件中。
使用：
`pipe.regist_extra_ckp_caches({'a':1})`




# pipeline 可以自定义覆盖的地方

### 自定义optimizer
 一个常见需求是。
 对不同的参数用不同的lr或者weightdecay。


# No Weight Decay 自动发现

qpipeline 支持通过约定方法自动发现哪些参数应该免除 weight decay。
当 `optim.optimizer_params.weight_decay > 0` 时，框架会递归遍历模型的 module tree，
根据约定方法将参数自动拆分为两个 optimizer param group：正常衰减组 + 免除衰减组。

## 约定方法

在 `nn.Module` 子类上实现以下方法之一（返回值 `List[str]`）：

### `no_decay() -> List[str]`

返回当前模块中不需要 weight decay 的**局部参数名**列表。
框架会**继续递归**遍历子模块。

```python
class MyLayerNorm(nn.LayerNorm):
    def no_decay(self) -> List[str]:
        return ["weight", "bias"]

class MyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 64, 128))
        self.norm = MyLayerNorm(128)
        self.linear = nn.Linear(128, 128)

    def no_decay(self) -> List[str]:
        return ["pos_embed"]
```

上例中，`pos_embed` 由 `MyBlock.no_decay()` 声明免除，
`norm.weight` 和 `norm.bias` 由 `MyLayerNorm.no_decay()` 声明免除。
框架在收集 `MyBlock` 的声明后，仍会递归进入 `self.norm`、`self.linear` 检查。

### `no_decay_deep() -> List[str]`

返回不需要 weight decay 的参数名列表（支持带点的子路径）。
框架遇到此方法后**停止递归**——该模块对其整个子树的 no-decay 声明负全责。

```python
class PretrainedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))
        self.blocks = nn.ModuleList([...])

    def no_decay_deep(self) -> List[str]:
        return [
            "cls_token",
            "pos_embed",
            "blocks.0.norm1.weight",
            "blocks.0.norm1.bias",
            # ... 对整个子树的完整声明
        ]
```

## 行为规则

- 若同一模块同时实现 `no_decay_deep` 和 `no_decay`，仅 `no_decay_deep` 生效
- 当 `weight_decay=0` 时，整个发现机制不执行（no-op）
- 冻结参数（`requires_grad=False`）不会进入任何 optimizer group
- 返回的参数名不存在于实际模型参数中时，触发 warning 并跳过
- 最终产生两个 param group，共享 lr、betas 等所有超参，仅 `weight_decay` 不同：
  - Group 0: `weight_decay = 配置值`
  - Group 1: `weight_decay = 0.0`
- 若所有参数都不声明 no_decay，则维持单 param group（与未实现约定方法等价）




# task

task的设计概念，能同时看到dataset实现和model实现的胶水层。
但同时又看不到runner细节，只知道根据约定实现runner需要的某些接口，
是夹在中间的这么一个单位。


task需要继承qTaskBase，

实现 `__init__` 方法，在初始化过程中，准备3个dataloader赋予自身。
    - self.train_loader
    - self.val_loader
    - self.test_loader  (optional, can be None)

并实现 
- batch_forward
- batch_metric
- batch_loss
- post_metric_to_err方法。

```python
from qqtools.pipeline import qTaskBase
class MyTask(qTaskBase):

    def __init__(self, args):
        super().__init__()
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def batch_forward(self, model, batch_data) -> Dict[str, Tensor]:
        """
        Returns:
            out (dict): dict-like
        """
        pass

    def batch_metric(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        """
        Returns:
            dict:  `{ 'metric_name': (metric_value, sample_count) }`
        """
        pass

    def batch_loss(self, out, batch_data) -> Dict[str, Tuple[Tensor, int]]:
        """
        Returns:
            dict:  `{ 'loss': (loss_value, sample_count) }`
        """
        pass

    def post_metric_to_err(self, result) -> float:
        """
        In cases where multiple metrics are available, the error metric must be prioritized
        to identify the optimal validation performance.

        Args:
            result: Dict[metric_name, metric_avg]

        Returns:
            float: The performance error value.
        """
        pass
```


task 为了适应不同的模型，
允许添加optional function
- pre_batch_forward(batch_data)  处理batch数据格式
- post_batch_forward(output, batch_data)  处理模型输出



这些操作可以统一成pipeline的middleware
- pipeline.regist_middleware()
- `def middileware(pipe):` 这里可以对task做任意修改。  


## Task Runner的生命周期



```bash
┌───────────────────────────────────────────────────────┐
│          pre_batch_forward(batch_data) -> batch_data  │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌───────────────────────────────────────────────────────┐
│         batch_forward(batch_data) -> batch_out        │
└───────────────────────┬───────────────────────────────┘
                        │
                        ↓
┌────────────────────────────────────────────────────────┐
│ post_batch_forward(batch_out, batch_data) -> batch_out │
└───────────────────────┬────────────────────────────────┘
                        │
           ┌────────────┴────────────────────────────────┐
           │                                             │
           v                                             v
┌────────────────────────────────────┐ ┌──────────────────────────────────┐
│batch_metric(batch_out, batch_data) │ │batch_loss(batch_out, batch_data) │
│-> batch_metrics                    │ │-> loss (if training)             │
└──────────┬─────────────────────────┘ └───────────────┬──────────────────┘
           │                                           │
           │                                           │
           └──────────────────┬────────────────────────┘
                              │  
                              v
┌────────────────────────────────────────────────────────────────┐
│   gather_all_batch_metrics  -> (epoch_metrics, epoch_loss_avg) │
└─────────────────────────────┬──────────────────────────────────┘        
                              │       
                              v
┌──────────────────────────────────────────────────────────────┐
│       post_metric_to_err(epoch_metrics)-> err: float         │
└──────────────────────────────────────────────────────────────┘
```

- 约定batch_data需要支持 `.to(device)` 方法，runner会自动调用
- post_metric_to_err 输出一个float，用来作为validation metric，挑选best model，以及控制early stop逻辑。


## task的可选方法
如果task 实现了`.to(device)` 方法,pipeline会在prepare_task后调用

# Task lifecycle hooks

除了必须实现的方法外，`qTaskBase` 还支持少量显式声明的 task 生命周期 hook。
这些 hook:

- 必须声明在 `qTaskBase`
- 必须列在 `OPTIONAL_METHODS`
- 必须使用固定 typed context 签名：
  - `on_epoch_start(self, context: BaseEventContext) -> None`
  - `on_epoch_end(self, context: BaseEventContext) -> None`
  - `on_train_batch_end(self, context: ProgressEventContext) -> None`
  - `on_validation_end(self, context: ValidationEndEventContext) -> None`
  - `on_early_stop(self, context: BaseEventContext) -> None`

目前支持的 task 生命周期 hook:

- `on_epoch_start`
- `on_epoch_end`
- `on_train_batch_end`
- `on_validation_end`
- `on_early_stop`

这些 hook 会由 runner 自动桥接到内部 listener 事件系统。
未列入 `qTaskBase` / `OPTIONAL_METHODS` 的生命周期方法，不属于官方支持面。

context 读取约定：

- 宏观运行信息统一走 `context.runner.*`
- `context.runner.run_state` 默认直接暴露当前 `RunningState`
- `batch_idx` / `total_batches` 只在 batch/progress 相关事件上可用
- `signal` 是默认唯一允许 listener 回写主流程的通道
- `signal` 不是普遍可用字段；仅 `on_validation_end` / `on_early_stop` / `on_checkpoint_request` 等控制型事件保证存在
- `run_state` 与其它 payload 一样都不做机制上的只读防护；如果用户确实要修改，本框架保留这种自由度，但默认语义仍建议将其视为事件输入数据
- `on_epoch_start` 的 public context 不承诺 `total_batches`

# qstd Args

根据约定的一组args。

接受`qTrainSchema.json`校验。

保留关键字：
- `$BASE` 负责继承其它配置文件。
- log_dir 所有日志、ckp、metric的存储路径。
- ckp_file 指定模型ckp文件。 



# RunAgent 与 logger类

logger类本质上是实现了一系列监听接口，注册在 run agent 的生命周期钩子里。

其中需要区分：

- 公共生命周期 hook: 面向 task / 外部 listener 的稳定接口
- 内部 listener hook: 面向 runner 内建组件的内部事件，不属于 task 官方支持面

- on_batch_end
- on_train_batch_end
- on_epoch_start_internal
- on_epoch_end

此外还有：

- on_eval_start
- on_eval_end
- on_validation_end
- on_checkpoint_request
- on_early_stop





# ddp remarks

```python
# torch/nn/parallel/distributed.py
# self._pre_forward(*inputs, **kwargs)
# moved_inputs, moved_kwargs = _to_kwargs(

```


# EMA support

TODO
qt.recover  add  
try_ema = True, 



# 单双精度支持

自动读取 `args.model.dtype`
> assert args.model.dtype in ['float32', 'float64']

- 自动 model.to(dtype)
- 约定 batch_data 支持 `to_dtype(target)` 方法
- 如果 batch_data不支持 `to_dtype`，可以在 `task.pre_batch_forward(batch_data)` 中处理dtype。





# 多阶段 Optim

Demand:
>stage 1 用一组超参 比如控制lr
> stage 2 换一组超参，weight 甚至换optim。
需要提供修改optim的接口。

>还有一个问题，如果要修改的stage2的超参是task wise的
怎么跟task约定接口？
task的实现者需要知道什么？

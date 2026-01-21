
# general position

`.pipeline` 是 `.torch` 的 消费者。


# cmd_args

标准args通过 `python xx.py --config /path/to/config.yml`的方式，指定config文件。
在py文件中通过这个函数加载config文件中的内容。
`args = prepare_cmd_args()`

若是希望通过cmd line自定义更多其它参数可以使用`updateParser`
example:
```python
from qqtools.pipeline import prepare_cmd_args

def updateParser(parser: argparse.ArgumentParser):
    parser.add_argument("--file", type=str)
    return parser
args = prepare_cmd_args(updateParser=updateParser)
file = args.file
```


# trainPipeline



## pipeline init
有两种init方式。

第一种直接传入实例化的task和model。

```python
model = prepare_model(args)
task = prepare_task(args)
pipe = QPipeline(args, test=False, task=task, model=model)
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


pipe = MyPipeline(args, test=False)
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

# Runner可选的生命周期

除了4个必须实现的方法外，
还可以选择性实现几个生命周期方法，
上层的runner会检测 `if hasattr(task, "somename")`， 如果实现了就在适当的时候调用它们。


- onBatchStart

- extra_batch_metrics()  这里可以看到model，如果想监控model里面的东西
 

(具体逻辑在EpochAgent里
和task的接口约定都在EpochAgent里。
)

# qstd Args

根据约定的一组args。

接受`qTrainSchema.json`校验。

保留关键字：
- `$BASE` 负责继承其它配置文件。
- log_dir 所有日志、ckp、metric的存储路径。
- ckp_file 指定模型ckp文件。 



# RunAgent 与 logger类

logger类本质上是实现了一系列监听接口，注册在 run agent 的生命周期钩子里。

- onBatchEnd
- onTrainBatchEnd
- onEpochStart
- onEpochEnd

> 实践角度来看目前没有遇到需要onBatchStart的情况，也许未来可以支持一下。





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


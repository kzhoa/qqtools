<div style="
  position: relative;
  width: 100%;
  padding-top: 66.66%; 
  margin-bottom: 20px;
  background: #f0f0f0 url('static/banner_960.jpg') center/contain no-repeat;
  background-size: cover;
">
  <img src="static/banner_960.jpg" 
       alt="" 
       style="
         position: absolute;
         top: 0;
         left: 0;
         width: 100%;
         height: 100%;
         opacity: 0;
       ">
</div>

# ✨qqtools✨
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qqtools?period=total&units=ABBREVIATION&left_color=GREY&right_color=BRIGHTGREEN&left_text=PyPI+Downloads)](https://pepy.tech/projects/qqtools) ![PyPI - Monthly Downloads](https://img.shields.io/pypi/dm/qqtools?color=3cb371&label=Monthly) ![Python version](https://img.shields.io/badge/python->=3.11-blue)

A lightweight library, crafted and battle-tested daily by *qq*, to make PyTorch life a little easier.

I’ve gathered the repetitive parts of my day-to-day work and refined them into this slim utility library.
It serves as my personal toolkit for handling data, training, and experiments, designed to keep projects moving fast with cleaner code and smoother workflows (and hopefully yours too!).

>Built for me, shared for you.

## What it includes

At its core, `qqtools` is a collection of small utilities I use around PyTorch projects:

- data containers such as `qDict` and `qData`
- dataset and dataloader helpers such as `qDictDataset` and `qDictDataloader`
- small neural network helpers such as `qMLP`
- a lightweight training framework, `qpipeline`
- a command-line experiment queue for Linux, `qexp`
- config and serialization helpers for YAML, JSON, pickle, and LMDB


For example, `qDict` is mainly there for cleaner attribute access in batch-like code:

```python
# Instead of dirty dict brackets:
# batch["input_ids"], batch["attention_mask"]

# Use clean attribute access:
batch = qt.qDict({"input_ids": input_ids, "attention_mask": attention_mask})
out = model(batch.input_ids)
```

At the core, it is still a practical toolbox for the repetitive parts around experiments.

## Install

Core install:

```bash
pip install qqtools
```

Full install:

```bash
pip install qqtools[full]
```

If you only want the experiment queue extras:

```bash
pip install qqtools[exp]
```

Requirements:

- Python `>=3.11`
- `torch>=2.0`
- `PyYAML>=6.0`

Notes:

- some parts still work with `torch==1.x`
- `torch>=2.4` is recommended

## A tiny example

```python
import qqtools as qt
qt.import_common(globals())

x = np.random.rand(100, 5)
y = np.random.rand(100, 1)

data_list = [qt.qData({"x": x[i], "y": y[i]}) for i in range(len(x))]
dataset = qt.qDictDataset(data_list=data_list)
dataloader = qt.qDictDataloader(dataset=dataset, batch_size=32, shuffle=True)

model = qt.nn.qMLP([5, 5, 1], activation="relu")
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)

device = torch.device("cuda")
model.to(device)

for epoch in range(100):
    for batch in dataloader:
        batch.to(device)
        out = model(batch.x)
        loss = loss_fn(out, batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"{epoch} {loss.item():4.6f}")
```

## qexp

`qexp` is the most standalone tool in this repository.
It is a lightweight experiment queue for Linux GPU hosts built around a shared project root.

Install:

```bash
pip install qqtools[exp]
```

Quick start:

```bash
qexp init --shared-root /mnt/share/myproject/.qexp --machine gpu-a
qexp submit --name demo -- python train.py --epochs 10
qexp agent start --background
qexp list
qexp top
qexp logs <task_id> --follow
```

After `init`, `qexp` saves the current `shared_root` and `machine` as CLI context, so you usually do not need to repeat them on every command.

Python API:

```python
from qqtools.plugins import qexp

task = qexp.submit(
    qexp.load_root_config("/mnt/share/myproject/.qexp", "gpu-a"),
    command=["python", "train.py", "--epochs", "10"],
    name="demo",
)
print(task.task_id)
```

Notes:

- packaged CLI surface: `qexp`
- packaged Python API surface: `from qqtools.plugins import qexp`
- actual execution is supported on local Linux GPU hosts with `tmux` installed
- non-Linux development is mainly for parsing, rendering, and tests

## Plugin modules

Under `src/qqtools/plugins/`, there are also:

- `qchem` - tools for reading and processing quantum chemistry outputs
- `qpipeline` - a training pipeline framework built on top of the core torch utilities
- `qhyperconnect` - an implementation of Hyper-Connection for PyTorch

## Test

```bash
tox
```

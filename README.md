

<!-- <div>
  <img src="static/banner_1960.jpg" width="400" height="300" alt="">
</div> -->
<!-- <pre style="margin: 0; padding: 0; height: 552px; visibility: hidden;"></pre> -->

<!-- ![qqtools-banner](static/banner_960.jpg) -->

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

# qqtools
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qqtools?period=total&units=ABBREVIATION&left_color=GREY&right_color=BRIGHTGREEN&left_text=PyPI+Downloads)](https://pepy.tech/projects/qqtools)  ![Python version](https://img.shields.io/badge/python->=3.11-blue)  


A small tool package for qq

# Requirements

- torch>=2.0 for full functionality 
  - Some components maintain backward compatibility with torch==1.x
  - Recommended: torch>=2.4
- pyyaml>=6.0
  - We recommend using YAML format for all configuration files.

This provides a unified approach to drive and manage all workflow operations.

For simplicity install 
```bash
pip install qqtools
```

For full support, run:
```bash
pip install qqtools[full]
```


# Data Format Support

Non-torch formats:
```bash
qDict : Enhanced of basic Dict.
qScalaDict : Dict[str, num]. A dict that maps str to scala;
qListData : List[dict]. A list of dicts.
```

Torch-related data formats
```bash
qData
qBatchList
```



# Simple Training Loop

For jupyter users

```python
import qqtools as qt
qt.import_common(globals())

x = np.random.rand(100, 5)
y = np.random.rand(100)

# dataset wrap
xs = [ x[i] for i in range(len(x))]
ys = [ y[i] for i in range(len(y))]
data_list = [ qt.qData({'x': x[i], 'y':y[i]})  for i in range(len(x))] 
dataset = qt.qDictDataset(data_list=data_list)
dataloader = qt.qDictDataloader()

# model
model = qt.nn.qMLP([5,5,1], activation="relu")
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)

# device
device = torch.device("cuda")
model.to(device)

# loop
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


# Individual Modules

The following modules are consumers of the core functionality provided by this package. Each is designed to be independent, allowing for sole import.

under `plugins/`

- qchem
- qpipeline



# Test

```bash
tox
```

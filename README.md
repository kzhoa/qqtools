# qqtools

A small tool package for qq

# Requirements

- torch>=2.0 for full functionality 
  - Some components maintain backward compatibility with torch==1.x
  - Recommended: torch>=2.4
- pyyaml>=6.0
  - We recommend using YAML format for all configuration files.
This provides a unified approach to drive and manage all workflow operations.

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

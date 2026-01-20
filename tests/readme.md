# Tests Construction Progress​


## 1 Common Functionals 

- √ qdict 
-  qtimer
-  qdata
-  qlogreader

### 1.1 `config` related
- dump_yaml, load_yaml
- load_pickle, save_pickle
- find_root, update_sys
- batch_assert_type 



## 2 Torch / DeepLearning Related

### auxiliary
- recover, save_ckp
- parse_device
- random_split_train_valid, random_split_train_valid_test, get_data_splits
- qContextProvider  # TBC


### operator
- qscatter: scatter, softmax


## 3 General Utilities

literally

- from .utils.qtypecheck import ensure_scala, ensure_numpy, str2number, is_number, is_inf
- from .utils.check import check_values_allowed, is_alias_exists


## 4 independent modules

- qchem
- qpipeline

### 4.1 qchem



### 4.2 qpipeline




# User Story


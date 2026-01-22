v1.0.0
    first version
v1.0.2
    add cuda synchronization for qtimer
v1.0.3 
    add assert 
v1.0.5
    refactor torch ops
v1.0.6
    enhance timer
v1.0.7
    add freeze random seed
v1.0.8
    ddp safe & time format
v1.0.9
    donot sort keys by default when dump yaml 
v1.0.10
    add find root
v1.0.11
    add qtyping
v1.0.12
    add yaml inherit loader
    fix find_root function
v1.0.13
    add qDict from_args
    style typing alias
    fix qDict subclass copy bug with dataloader while pin_memory is True
v1.0.14
    add find_root optional return type
    change inherit loader to $base
v1.0.15
    add early stop support to ckp save&load
    add pickle i/o
    add lazyimport for class or function & transfer import common to lazy style 
v1.0.0
    feat import common support custom global
    feat save ckp allow verbose control
v1.1.0
    refactor module organization
v1.1.1
    feat global import nn.funtional as F
    feat add nn submdule
    feat add nn.DoNothing
    feat add qcontextprovider
v1.1.2
    feat add functional donothing
    feat recover support lr scheduler
    feat add get_data_splits tool function
    feat add freeze_module
    fix qcontextprovider apply to general object instance 
    refactor find root return Path by default 
v1.1.3
    fix qdataset fetch logics
v1.1.4
    add qmlp
    fix qdict deepcopy memo 
v1.1.5
    add qdict fetch
    add unfreeze module
v1.1.6
    add ensure numpy
    fix unfreeze import relationship
v1.1.7 
    add more import common packages
    fix qdict _copy_ compatible with copy.copy()
v1.1.9
    add numeric check and convert from str input
    add alias for scatter mean&add
    add staticmethod qData.get_data_splits
    fix get nonlinearity discriminate str and callable activation
v1.1.10
    feat qlistdata 
    feat avgbank
    feat more qimports
    feat set seed argument to naive split implementations
    feat naive initialization for qdictdataset
    feat naive& graph collate fn for qdictdataloader
    refactor get_data_split moved to qsplit module
v1.1.12
    add numpy type annotation in qtyping
    refactor change qData __init__ into kwargs style
    refactor change qDictDataset interject name to raw_files_exist 
V1.1.16
    fix qtyping np.bool_ 
    fix qData initialize for backward compatibility
    fix qData init signature inconsist with parent which causes copy.copy() problem
v1.1.17
    feat qdatalist plot_counts, support iter
    feat add smart_combine for qDictDataset's collate fn
    refactor data.qdatalist filename
    refactor add staticmethod collate_dict_samples & graph in qDictDataset for convenience
    fix qimport for matplotlib.pyplot
v1.1.19
    feat rank zero only
    fix qdict init for kv tuple list
v1.1.20
    feat add BatchList class
v1.1.21
    feat add fn binary_metrics for a set of useful indicators 
    fix qDataList check bool before int
v1.1.22
    feat to device support for qData, automatically adapt single/double precision
    doc qDataset init 
v1.1.25
    chore switch to pyproject.toml build system
v1.1.26
    feat add attr-related functions is_override
    feat under test: add plugin qpipeline
v1.1.27
    feat add plugin qchem
    feat add hyperconnect
    fix pipeline task opt impl check

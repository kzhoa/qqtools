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
    fix qdict deepcopy memo 

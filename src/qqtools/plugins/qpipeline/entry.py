import types

from .cmd_args import prepare_cmd_args
from .entry_utils import get_param_stats
from .qpipeline import qPipeline


def create_pipeline_class(prepare_model_func, prepare_task_func, base_pipeline_class=qPipeline):
    class_name = "DynamicPipeline"
    class_attributes = {
        "prepare_model": staticmethod(prepare_model_func),
        "prepare_task": staticmethod(prepare_task_func),
    }
    # DynamicPipelineClass = types.new_class(class_name, (base_pipeline_class,), kwds=class_attributes)
    DynamicPipelineClass = type(class_name, (base_pipeline_class,), class_attributes)
    return DynamicPipelineClass


def general_train(prepare_model, prepare_task):
    args = prepare_cmd_args()
    assert args.test == False

    model = prepare_model(args)
    print("model", get_param_stats(model))

    MyPipeline: qPipeline = create_pipeline_class(prepare_model, prepare_task)

    pipe = MyPipeline(args, train=True, model=model)
    model.to(args.device)

    print("ddp:", args.distributed, "rank:", args.rank)

    if hasattr(pipe.task, "pipe_middle_ware"):
        pipe.task.pipe_middle_ware(pipe)
    return pipe

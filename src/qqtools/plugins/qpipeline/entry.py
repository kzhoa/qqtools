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


def create_pipeline(prepare_model, prepare_task):
    args = prepare_cmd_args()
    mode = "test" if args.test else "train"

    MyPipeline: qPipeline = create_pipeline_class(prepare_model, prepare_task)
    pipe = MyPipeline(args, mode=mode)

    pipe.logger.info(f"ddp: {args.distributed}, rank: {args.rank}")

    if pipe.model is not None:
        target_model = pipe.model.module if hasattr(pipe.model, "module") else pipe.model
        pipe.logger.info(f"model {get_param_stats(target_model)}")
    return pipe

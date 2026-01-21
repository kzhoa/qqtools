from ..qpipeline import qPipeline

__all__ = ["middleware_extra_ckp_caches"]


def middleware_extra_ckp_caches(pipe: qPipeline):
    task = pipe.task
    extra_ckp_caches = {"norm_factor": task.norm_factor}
    if hasattr(task, "target"):
        extra_ckp_caches["target"] = task.target
    pipe.regist_extra_ckp_caches(extra_ckp_caches)

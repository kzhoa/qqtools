"""Structured evaluation results and canonical score targets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Union

from torch.utils.data import DataLoader

from ...types import Stage


LoaderGroup = Optional[Union[DataLoader, Dict[str, DataLoader]]]


class ScoreTarget(str, Enum):
    """Stable YAML names for framework control metrics."""

    TRAIN = "train_metric"
    VAL = "val_metric"
    TEST = "test_metric"
    EMA_VAL = "ema_val_metric"
    EMA_TEST = "ema_test_metric"


@dataclass(frozen=True)
class LoaderEvaluation:
    """Raw metrics and optional outputs produced by one evaluation loader."""

    name: Optional[str]
    metrics: Mapping[str, Any]
    outputs: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"name": self.name, "metrics": dict(self.metrics)}
        if self.outputs is not None:
            result["outputs"] = dict(self.outputs)
        return result


@dataclass(frozen=True)
class StageEvaluation:
    """All loader measurements and the task-derived score for one stage."""

    stage: Stage
    loaders: tuple[LoaderEvaluation, ...]
    score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage.value,
            "score": self.score,
            "loaders": [loader.to_dict() for loader in self.loaders],
        }


@dataclass(frozen=True)
class ModelEvaluation:
    """Stage evaluations produced by one model variant."""

    variant: str
    stages: tuple[StageEvaluation, ...]

    def stage_result(self, stage: Stage) -> Optional[StageEvaluation]:
        return next((result for result in self.stages if result.stage == stage), None)

    def to_dict(self) -> Dict[str, Any]:
        return {"variant": self.variant, "stages": [stage.to_dict() for stage in self.stages]}


@dataclass(frozen=True)
class TrainingResult:
    """Aggregated metrics and score for the training interval before evaluation."""

    metrics: Mapping[str, Any]
    score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {"metrics": dict(self.metrics), "score": self.score}


@dataclass(frozen=True)
class EvaluationResult:
    """Complete structured result for one evaluation boundary or standalone evaluation."""

    models: tuple[ModelEvaluation, ...]
    training: Optional[TrainingResult] = None

    @staticmethod
    def validate_target(target: str) -> ScoreTarget:
        try:
            return ScoreTarget(target)
        except ValueError as error:
            allowed = ", ".join(item.value for item in ScoreTarget)
            raise ValueError(f"Unsupported score target {target!r}. Allowed targets: {allowed}.") from error

    def target_value(self, target: Union[str, ScoreTarget]) -> Optional[float]:
        target = target if isinstance(target, ScoreTarget) else self.validate_target(target)
        if target is ScoreTarget.TRAIN:
            return self.training.score if self.training is not None else None

        variant = "ema" if target in {ScoreTarget.EMA_VAL, ScoreTarget.EMA_TEST} else "standard"
        stage = Stage.VAL if target in {ScoreTarget.VAL, ScoreTarget.EMA_VAL} else Stage.TEST
        model_result = next((result for result in self.models if result.variant == variant), None)
        stage_result = model_result.stage_result(stage) if model_result is not None else None
        return stage_result.score if stage_result is not None else None

    def to_dict(self) -> Dict[str, Any]:
        result = {"models": [model.to_dict() for model in self.models]}
        if self.training is not None:
            result["training"] = self.training.to_dict()
        return result


def resolve_loader_group(loader: LoaderGroup, *, group_name: str) -> list[tuple[Optional[str], DataLoader]]:
    """Validate one task-owned loader property and return its pass-local snapshot."""
    if loader is None:
        return []
    if isinstance(loader, DataLoader):
        return [(None, loader)]
    if not isinstance(loader, dict):
        raise TypeError(
            f"{group_name} loader must be None, DataLoader, or dict[str, DataLoader]; "
            f"got {type(loader).__name__}."
        )
    if not loader:
        raise ValueError(f"{group_name} loader mapping must be non-empty; use None to disable the stage.")

    resolved: list[tuple[Optional[str], DataLoader]] = []
    for name, data_loader in loader.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{group_name} loader name must be a non-empty string; got {name!r}.")
        if not isinstance(data_loader, DataLoader):
            raise TypeError(
                f"{group_name} loader {name!r} must be a DataLoader; got {type(data_loader).__name__}."
            )
        resolved.append((name, data_loader))
    return resolved

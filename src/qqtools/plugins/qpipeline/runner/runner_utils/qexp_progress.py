"""Optional rank-zero qexp observer, independent of every terminal renderer."""

from __future__ import annotations

import os
from typing import Any

from qqtools.qexp.progress import Reporter


class QexpProgressObserver:
    """Translate committed runner facts without maintaining a training counter."""

    def __init__(self, config: Any, reporter: Reporter) -> None:
        self._config = config
        self._reporter = reporter

    def on_train(self, fact: Any) -> None:
        self._reporter.update(
            stage="train", current=fact.global_step, total=self._config.max_steps,
            unit="step", message=f"epoch {fact.epoch}",
        )

    def on_evaluation_started(self, fact: Any) -> None:
        # EvaluationStartedFact has no val/test discriminator. Do not pretend
        # that test-only evaluation is validation before the first tick arrives.
        self._reporter.update(stage="evaluation", message=f"global_step {fact.global_step}")

    def on_progress_tick(self, fact: Any) -> None:
        stage = getattr(fact.stage, "value", fact.stage)
        if stage not in {"val", "test"}:
            return
        total = fact.total_batches if fact.total_batches > 0 else None
        current = fact.batch_index + 1
        if total is not None and current > total:
            current, total = None, None
        self._reporter.update(
            stage="validation" if stage == "val" else "test",
            current=current, total=total, unit="batch",
            message=f"global_step {fact.global_step}",
        )

    def close(self) -> None:
        self._reporter.close()


def bind_qexp_progress(observers: Any, config: Any) -> QexpProgressObserver | None:
    """Attach peer observers before bindings freeze; absent qexp means no work."""
    path = os.environ.get("QEXP_PROGRESS_PATH")
    if not path or config.rank != 0:
        return None
    try:
        observer = QexpProgressObserver(config, Reporter(path))
        for name, callback in (
            ("epoch_started", observer.on_train),
            ("train_boundary", observer.on_train),
            ("evaluation_started", observer.on_evaluation_started),
            ("progress_tick", observer.on_progress_tick),
            ("evaluation_committed", observer.on_train),
        ):
            observers.bind(name, callback, policy="best_effort")
        return observer
    except Exception:
        return None

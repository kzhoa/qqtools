import json

import pytest

from qqtools.plugins.qpipeline.runner.runner_utils.metrics_jsonl import MetricsJsonlLogger


def test_metrics_jsonl_logger_writes_one_parseable_record_per_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    logger = MetricsJsonlLogger(path)
    logger.write({"event": "train_batch", "global_step": 1, "metrics": {"loss": 0.2}})
    logger.write({"event": "evaluation", "global_step": 2, "evaluation": {"models": []}})
    logger.close()

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"event": "train_batch", "global_step": 1, "metrics": {"loss": 0.2}},
        {"event": "evaluation", "global_step": 2, "evaluation": {"models": []}},
    ]


def test_metrics_jsonl_logger_rejects_writes_after_finalization(tmp_path):
    logger = MetricsJsonlLogger(tmp_path / "metrics.jsonl")
    logger.abort()
    with pytest.raises(RuntimeError, match="closed"):
        logger.write({"event": "evaluation"})

from __future__ import annotations

import pytest

from qqtools.plugins.qexp.agent.project_io_controller import (
    _PREVALIDATION_RECONCILIATION_OPERATION_KINDS,
    _PREVALIDATION_RESOLVER_NAMES,
    _VALIDATION_INDEPENDENT_RECONCILIATION_OPERATION_KINDS,
    _VALIDATION_OPERATION_KINDS,
    _validate_prevalidation_operation_partition,
)
from qqtools.plugins.qexp.agent.project_io_protocol import PROJECT_IO_OPERATIONS


def test_prevalidation_policy_partitions_every_project_io_operation() -> None:
    _validate_prevalidation_operation_partition(
        _VALIDATION_OPERATION_KINDS,
        _PREVALIDATION_RECONCILIATION_OPERATION_KINDS,
        _VALIDATION_INDEPENDENT_RECONCILIATION_OPERATION_KINDS,
    )

    assert (
        _VALIDATION_OPERATION_KINDS
        | _PREVALIDATION_RECONCILIATION_OPERATION_KINDS
        | _VALIDATION_INDEPENDENT_RECONCILIATION_OPERATION_KINDS
        == PROJECT_IO_OPERATIONS
    )
    assert frozenset(_PREVALIDATION_RESOLVER_NAMES) == _PREVALIDATION_RECONCILIATION_OPERATION_KINDS


def test_prevalidation_policy_rejects_an_unclassified_operation() -> None:
    with pytest.raises(RuntimeError, match="partition"):
        _validate_prevalidation_operation_partition(
            _VALIDATION_OPERATION_KINDS,
            _PREVALIDATION_RECONCILIATION_OPERATION_KINDS - {"machine_snapshot_publish"},
            _VALIDATION_INDEPENDENT_RECONCILIATION_OPERATION_KINDS,
        )


def test_prevalidation_policy_rejects_an_operation_in_multiple_classes() -> None:
    with pytest.raises(RuntimeError, match="partition"):
        _validate_prevalidation_operation_partition(
            _VALIDATION_OPERATION_KINDS | {"machine_snapshot_publish"},
            _PREVALIDATION_RECONCILIATION_OPERATION_KINDS,
            _VALIDATION_INDEPENDENT_RECONCILIATION_OPERATION_KINDS,
        )

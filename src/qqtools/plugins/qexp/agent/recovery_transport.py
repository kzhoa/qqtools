"""Closed non-authorizing transport for retained legacy evidence reads."""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ..runtime.responsibility_backfill import (
    CapturedEvidenceLocator,
    _canonical_capture_root,
    _evidence_identity,
    _relative_record,
)
from ..runtime.responsibility_capture import SOURCE_CAPTURE_FORMAT, SourceHold
from ..runtime.responsibility_import import RECORD_KEYS, recovery_locator
from ..runtime.responsibility_source_scan import source_scan_cursor, source_scan_result

_CAPTURE_ID = re.compile(r"[0-9a-f]{32}\Z")
_PARAMETERS = {"machine_name", "capture_id", "backfill_id", "backfill_revision", "lane", "relative"}
_LOCATOR_FIELDS = {"source_root", "lane", "relative", "identity", "task_id", "attempt_number"}


def recovery_completion_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"machine_name", "completion_digest"}:
        raise ValueError("recovery completion parameters are invalid")
    digest = value["completion_digest"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("recovery completion digest is invalid")
    return dict(value)


def recovery_generation_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"machine_name", "completion_digest", "phase"}:
        raise ValueError("capture generation parameters are invalid")
    if not isinstance(value["phase"], str) or value["phase"] not in {"retain", "normalize"}:
        raise ValueError("capture generation phase is invalid")
    return {
        **recovery_completion_parameters({key: item for key, item in value.items() if key != "phase"}),
        "phase": value["phase"],
    }


def recovery_completion_evidence(value: Mapping[str, Any], *, completed_state: str) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"state"}
        or not isinstance(value["state"], str)
        or value["state"] not in {completed_state, "waiting", "stale"}
    ):
        raise ValueError("recovery completion evidence is invalid")
    return dict(value)


def legacy_scan_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != (_PARAMETERS - {"relative"}) | {"cursor"}:
        raise ValueError("legacy source discovery parameters are invalid")
    relative = "attempt/decision.json" if value["lane"] == "termination_decisions" else "attempt.json"
    parameters = legacy_capture_parameters(
        {**{key: item for key, item in value.items() if key != "cursor"}, "relative": relative}
    )
    parameters.pop("relative")
    return {**parameters, "cursor": source_scan_cursor(value["cursor"])}


def legacy_scan_evidence(value: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"state", "scan"}:
        raise ValueError("legacy source discovery evidence fields are invalid")
    if value["state"] == "stale" and value["scan"] is None:
        return {"state": "stale", "scan": None}
    scan = value["scan"]
    if value["state"] != "observed":
        raise ValueError("legacy source discovery evidence is invalid")
    return {"state": "observed", "scan": source_scan_result(scan, parameters["lane"])}


def recovery_admission_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"state", "registration_prepared", "admission_fenced"}:
        raise ValueError("recovery admission evidence fields are invalid")
    if type(value["registration_prepared"]) is not bool or type(value["admission_fenced"]) is not bool:
        raise ValueError("recovery admission evidence flags must be booleans")
    state = value["state"]
    prepared, fenced = value["registration_prepared"], value["admission_fenced"]
    if not (
        (state == "ready" and prepared and fenced)
        or (state == "waiting" and not fenced)
        or (state in ("stale", "superseded") and not prepared and not fenced)
    ):
        raise ValueError("recovery admission evidence does not match its state")
    return dict(value)


def source_hold_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"machine_name", "capture_id"}:
        raise ValueError("source retention parameters are invalid")
    if not isinstance(value["capture_id"], str) or not _CAPTURE_ID.fullmatch(value["capture_id"]):
        raise ValueError("source retention capture identity is invalid")
    return dict(value)


def decode_source_hold(value: Mapping[str, Any], capture_id: str) -> SourceHold:
    if not isinstance(value, Mapping) or set(value) != {
        "format",
        "runtime_root",
        "target_root",
        "instance",
        "capture_id",
        "phase",
    }:
        raise ValueError("source retention result fields are invalid")
    roots = []
    for key in ("runtime_root", "target_root"):
        raw = value[key]
        if not isinstance(raw, str) or len(raw.encode()) > 4096:
            raise ValueError("source retention runtime root is invalid")
        root = _canonical_capture_root(Path(raw))
        if str(root) != raw or root == Path("/"):
            raise ValueError("source retention runtime root is not canonical")
        roots.append(root)
    if (
        value["format"] != SOURCE_CAPTURE_FORMAT
        or value["phase"] != "pending"
        or value["capture_id"] != capture_id
        or not isinstance(capture_id, str)
        or not _CAPTURE_ID.fullmatch(capture_id)
        or roots[0] == roots[1]
        or not isinstance(value["instance"], str)
        or not _CAPTURE_ID.fullmatch(value["instance"])
    ):
        raise ValueError("source retention result identity is invalid")
    return SourceHold(roots[0], roots[1], value["instance"], capture_id)


def source_hold_evidence(value: Mapping[str, Any], capture_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"state", "hold"}:
        raise ValueError("source retention evidence fields are invalid")
    if value["state"] == "stale" and value["hold"] is None:
        return {"state": "stale", "hold": None}
    if value["state"] != "retained":
        raise ValueError("source retention evidence state is invalid")
    hold = decode_source_hold(value["hold"], capture_id)
    return {"state": "retained", "hold": hold.to_dict()}


def legacy_capture_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PARAMETERS:
        raise ValueError("legacy source read parameters are invalid")
    parameters = dict(value)
    for key in ("capture_id", "backfill_id"):
        if not isinstance(parameters[key], str) or not _CAPTURE_ID.fullmatch(parameters[key]):
            raise ValueError(f"legacy source read {key} is invalid")
    if type(parameters["backfill_revision"]) is not int or parameters["backfill_revision"] < 1:
        raise ValueError("legacy source read revision must be positive")
    if not isinstance(parameters["lane"], str) or parameters["lane"] not in RECORD_KEYS:
        raise ValueError("legacy source read lane is invalid")
    if not isinstance(parameters["relative"], str) or len(parameters["relative"].encode()) > 512:
        raise ValueError("legacy source read relative path is invalid")
    _relative_record(parameters["lane"], parameters["relative"])
    return parameters


def encode_capture_locator(captured: CapturedEvidenceLocator) -> dict[str, Any]:
    return {**asdict(captured), "source_root": str(captured.source_root)}


def decode_capture_locator(value: Mapping[str, Any], parameters: Mapping[str, Any]) -> CapturedEvidenceLocator:
    if not isinstance(value, Mapping) or set(value) != _LOCATOR_FIELDS:
        raise ValueError("legacy captured locator fields are invalid")
    if value["lane"] != parameters["lane"] or value["relative"] != parameters["relative"]:
        raise ValueError("legacy captured locator differs from the requested record")
    if not isinstance(value["source_root"], str) or len(value["source_root"].encode()) > 4096:
        raise ValueError("legacy source root is invalid")
    source_root = _canonical_capture_root(Path(value["source_root"]))
    if str(source_root) != value["source_root"] or source_root == Path("/"):
        raise ValueError("legacy source root is not a canonical runtime root")
    relative = _relative_record(value["lane"], value["relative"])
    identity = _evidence_identity(value["lane"], relative)
    payload = {"task_id": value["task_id"], "attempt_number": value["attempt_number"]}
    if value["identity"] != identity or recovery_locator(identity, payload) != payload:
        raise ValueError("legacy captured locator identity is invalid")
    return CapturedEvidenceLocator(source_root, value["lane"], value["relative"], identity, **payload)


def legacy_capture_evidence(value: Mapping[str, Any], parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"state", "locator"}:
        raise ValueError("legacy source read evidence fields are invalid")
    if not isinstance(value["state"], str) or value["state"] not in {"observed", "stale"}:
        raise ValueError("legacy source read evidence state is invalid")
    if value["state"] == "stale":
        if value["locator"] is not None:
            raise ValueError("stale legacy source read cannot return a locator")
        return dict(value)
    captured = decode_capture_locator(value["locator"], parameters)
    return {"state": "observed", "locator": encode_capture_locator(captured)}

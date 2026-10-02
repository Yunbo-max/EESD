"""Strict experiment-contract loading for EESD."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ContractError(ValueError):
    """Raised before compute when an experiment identity is incomplete."""


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ContractError(f"{field} must be a 40-character lowercase git revision")
    return value


def load_experiment(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ContractError("experiment config must be a mapping")
    schema = payload.get("schema")
    if schema == "eesd-recorded-lexical-v1" and payload.get("method_status") == "prospective_shapley":
        raise ContractError("recorded lexical evidence cannot be labeled prospective Shapley")
    if schema != "eesd-final-experiment-v1":
        raise ContractError(f"unsupported experiment schema: {schema!r}")
    for name, model in payload.get("models", {}).items():
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str):
            raise ContractError(f"model {name} lacks model_id")
        _sha(model.get("revision"), f"models.{name}.revision")
    for name, dataset in payload.get("benchmarks", {}).items():
        if not isinstance(dataset, dict) or not isinstance(dataset.get("source"), str):
            raise ContractError(f"benchmark {name} lacks source")
        _sha(dataset.get("revision"), f"benchmarks.{name}.revision")
    cells = payload.get("gate_cells")
    if not isinstance(cells, list) or not cells:
        raise ContractError("gate_cells must be nonempty")
    for cell in cells:
        if cell.get("model") not in payload["models"] or cell.get("dataset") not in payload["benchmarks"]:
            raise ContractError("gate cell references unknown model or benchmark")
    return payload


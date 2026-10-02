"""Deterministic, no-compute experiment planning."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import ContractError


STAGES = {"signal", "mechanism", "downstream", "expansion"}


def plan_stage(config: dict[str, Any], stage: str, *, seed: int) -> dict[str, Any]:
    if stage not in STAGES:
        raise ContractError(f"unknown stage: {stage}")
    allowed = config["protocol"]["seeds"]
    if seed not in allowed:
        raise ContractError(f"seed {seed} is outside the frozen seed list")
    plan = {
        "schema": "eesd-stage-plan-v1",
        "method_status": "prospective_shapley",
        "stage": stage,
        "seed": seed,
        "cells": deepcopy(config["gate_cells"]),
        "models": deepcopy(config["models"]),
        "benchmarks": deepcopy(config["benchmarks"]),
        "method": deepcopy(config["method"]),
    }
    if stage == "signal":
        plan.update(trajectories_per_cell=64, coalitions_per_trajectory=16)
    elif stage == "mechanism":
        plan.update(config["mechanism"])
    elif stage == "downstream":
        arms = list(config["downstream"]["training_arms"])
        budget = int(config["downstream"]["response_token_budget"])
        plan.update(config["downstream"])
        plan["arm_budgets"] = {arm: 0 if arm == "no_update" else budget for arm in arms}
    else:
        plan["cells"] = deepcopy(config["expansion"]["cells"])
        plan.update(config["expansion"])
    return plan


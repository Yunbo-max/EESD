#!/usr/bin/env python3
"""Validate and materialize one frozen EESD experiment stage.

Dry-run is the default safe planning interface. Actual model execution requires an
input manifest that binds correction banks, public data, hidden evaluators, model
snapshots, and output roots. Stage-specific worker scripts are recorded in the plan
so an agent can execute and resume cells independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eesd.contracts import ContractError, load_experiment
from eesd.planner import plan_stage


WORKERS = {
    "signal": ["score_eesd_shapley_relevance.py"],
    "mechanism": ["score_eesd_shapley_relevance.py", "report_eesd_axiomatic_validation.py"],
    "downstream": ["score_eesd_shapley_relevance.py", "run_eesd_weighted_sft.py", "run_eesd_downstream.py"],
    "expansion": ["run_eesd_downstream.py", "run_eesd_evalplus_transfer.py"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(WORKERS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = plan_stage(load_experiment(args.config), args.stage, seed=args.seed)
    plan["worker_scripts"] = WORKERS[args.stage]
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return
    if args.inputs is None:
        parser.error("--inputs is required unless --dry-run is used")
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")
    inputs = json.loads(args.inputs.read_text())
    required = {"schema", "model_snapshots", "dataset_roots", "evaluator_roots"}
    missing = required - set(inputs)
    if missing:
        raise ContractError(f"input manifest missing: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=False)
    plan["input_manifest"] = str(args.inputs.resolve())
    plan["execution_status"] = "planned_not_started"
    (args.output / "plan.json").write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "planned_not_started", "plan": str(args.output / "plan.json")}))


if __name__ == "__main__":
    main()

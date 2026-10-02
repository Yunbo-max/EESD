from pathlib import Path

import pytest

from eesd.contracts import ContractError, load_experiment
from eesd.planner import plan_stage


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "experiments" / "eesd_final.yaml"


def test_final_contract_has_three_gate_cells_and_exact_revisions():
    config = load_experiment(CONFIG)
    assert len(config["gate_cells"]) == 3
    assert {cell["role"] for cell in config["gate_cells"]} == {
        "positive_preservation", "moderate_signal", "harm_prevention"
    }
    for cell in config["gate_cells"]:
        model = config["models"][cell["model"]]
        dataset = config["benchmarks"][cell["dataset"]]
        assert len(model["revision"]) == 40
        assert len(dataset["revision"]) == 40


def test_downstream_plan_is_baseline_complete_and_budget_matched():
    plan = plan_stage(load_experiment(CONFIG), "downstream", seed=1701)
    expected = {
        "no_update", "equal_weight", "final_correctness", "scalar_confidence",
        "fixed_mass_dirichlet", "lexical_eed", "shapley_eed_no_kl", "eesd_full",
    }
    assert set(plan["training_arms"]) == expected
    assert plan["matched_response_token_budget"] is True
    assert plan["primary_sources_per_cell"] == 500
    assert len(plan["cells"]) == 3


def test_no_update_has_zero_training_budget():
    plan = plan_stage(load_experiment(CONFIG), "downstream", seed=1701)
    assert plan["arm_budgets"]["no_update"] == 0
    nonzero = {v for k, v in plan["arm_budgets"].items() if k != "no_update"}
    assert len(nonzero) == 1
    assert next(iter(nonzero)) > 0


def test_recorded_and_prospective_schemas_cannot_be_mixed(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema: eesd-recorded-lexical-v1\nmethod_status: prospective_shapley\n")
    with pytest.raises(ContractError, match="recorded lexical"):
        load_experiment(bad)


def test_unknown_stage_is_rejected():
    with pytest.raises(ContractError, match="stage"):
        plan_stage(load_experiment(CONFIG), "paper_claim", seed=1701)

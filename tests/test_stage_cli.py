import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).parents[1]


def test_stage_cli_dry_run_emits_complete_downstream_plan():
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_stage.py"),
         "--config", str(ROOT / "configs" / "experiments" / "eesd_final.yaml"),
         "--stage", "downstream", "--seed", "1701", "--dry-run"],
        check=True, capture_output=True, text=True, env=env,
    )
    payload = json.loads(result.stdout)
    assert payload["stage"] == "downstream"
    assert len(payload["cells"]) == 3
    assert payload["arm_budgets"]["no_update"] == 0
    assert payload["matched_response_token_budget"] is True


def test_stage_cli_refuses_execution_without_input_manifest(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_stage.py"),
         "--config", str(ROOT / "configs" / "experiments" / "eesd_final.yaml"),
         "--stage", "signal", "--seed", "1701", "--output", str(tmp_path / "out")],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0
    assert "--inputs" in result.stderr

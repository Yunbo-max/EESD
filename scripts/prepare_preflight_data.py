#!/usr/bin/env python3
"""Generate minimal preflight fixtures for scoring + 1-step LoRA training.

Produces two inputs that are NOT part of any scientific claim:
  - preflight/corrections-shapley.jsonl  (input for score_eesd_shapley_relevance.py)
  - preflight/corrections-score.jsonl    (input for score_eesd_corrections.py)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "preflight"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"

SYSTEM = "You repair Python programs using public execution feedback."
PREFIX = (
    "Repair the following stdin/stdout Python program. Use only the public "
    "execution evidence. Return only complete Python source code."
)

ORIGINAL = "def f(x):\n    return x + 1\n"
CORRECTION = "def f(x):\n    return x * 2\n"

EXECUTIONS = [
    {
        "id": 0,
        "input": "1",
        "expected": "2",
        "actual": "2",
        "stderr": "",
        "outcome": "WRONG_OUTPUT",
    },
    {
        "id": 1,
        "input": "2",
        "expected": "4",
        "actual": "3",
        "stderr": "",
        "outcome": "WRONG_OUTPUT",
    },
    {
        "id": 2,
        "input": "3",
        "expected": "6",
        "actual": "4",
        "stderr": "",
        "outcome": "WRONG_OUTPUT",
    },
    {
        "id": 3,
        "input": "4",
        "expected": "8",
        "actual": "5",
        "stderr": "",
        "outcome": "WRONG_OUTPUT",
    },
]


def user_prompt(idx: int) -> str:
    payload = {
        "task": f"double the input #{idx}",
        "program": ORIGINAL,
        "public_executions": EXECUTIONS,
    }
    return PREFIX + "\n" + json.dumps(payload, sort_keys=True, ensure_ascii=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- Shapley scoring input (eesd-correction-trajectory-v1) ---
    shapley_rows = []
    for i in range(2):
        shapley_rows.append(
            {
                "schema": "eesd-correction-trajectory-v1",
                "task_id": f"preflight/task/{i}",
                "model_id": MODEL_ID,
                "model_revision": REVISION,
                "adapter_sha256": None,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_prompt(i)},
                ],
                "original": ORIGINAL,
                "correction": CORRECTION,
                "before_outcomes": [
                    "WRONG_OUTPUT",
                    "RUNTIME_EXCEPTION",
                    "WRONG_OUTPUT",
                    "COMPILE_ERROR",
                ],
                "after_outcomes": ["PASS", "PASS", "PASS", "PASS"],
            }
        )
    (OUT / "corrections-shapley.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in shapley_rows) + "\n"
    )

    # --- score_eesd_corrections input (scored trajectory) ---
    score_rows = []
    for i in range(3):
        split = "development" if i == 2 else "train"
        score_rows.append(
            {
                "trajectory_id": f"preflight/traj/{i}",
                "source_component_id": f"preflight/source/{i}",
                "split": split,
                "prompt": user_prompt(i),
                "correction": CORRECTION,
                "before_outcomes": [
                    "WRONG_OUTPUT",
                    "RUNTIME_EXCEPTION",
                    "WRONG_OUTPUT",
                    "COMPILE_ERROR",
                ],
                "after_outcomes": ["PASS", "PASS", "PASS", "PASS"],
                "relevance": [0.25, 0.25, 0.25, 0.25],
            }
        )
    (OUT / "corrections-score.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in score_rows) + "\n"
    )

    print(f"wrote {OUT / 'corrections-shapley.jsonl'} ({len(shapley_rows)} rows)")
    print(f"wrote {OUT / 'corrections-score.jsonl'} ({len(score_rows)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Blind, create-once CPU-only EESD Gate 0. No model execution or training."""
import argparse
import json
from pathlib import Path

from eesd.update_gate0 import gate_update


def run(input_path, output_path):
    inp, out = Path(input_path), Path(output_path)
    if inp.resolve() == out.resolve() or out.exists():
        raise FileExistsError("output must be new and distinct from input")
    predictions = []
    seen = set()
    required = {"source_id", "before", "after", "score_kernel"}
    for line in inp.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if set(row) != required or not isinstance(row["source_id"], str):
            raise ValueError("only source_id/before/after/score_kernel allowed; hidden labels forbidden")
        source_id = row["source_id"]
        if not source_id or source_id in seen:
            raise ValueError("source_id must be nonempty and unique")
        seen.add(source_id)
        predictions.append({"source_id": source_id, **gate_update(row["before"], row["after"], row["score_kernel"])})
    if not predictions:
        raise ValueError("input bank is empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in predictions))
    return {"sources": len(predictions), "admitted": sum(p["accepted"] for p in predictions),
            "status": "CPU diagnostic only; no hidden evaluation or RL training"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), sort_keys=True))

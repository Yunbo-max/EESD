"""Diagnose cold-start: print Qwen's generated candidates and their outcomes.

Answers two questions: (1) are candidates syntactically valid Python, and
(2) at what sample count does pass@k become nonzero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import load_sources
from execution import execute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-sources", type=int, default=3)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    model_cfg = yaml.safe_load(args.model_config.read_text())
    model_id = model_cfg["model_id"]
    revision = model_cfg["revision"]

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
    )
    model.eval()

    sources = load_sources(
        args.problems, args.invocations, n_sources=args.n_sources, seed=args.seed
    )
    lines: list[str] = []
    for source in sources:
        prompt = source["prompt"] + "\n"
        prompt_ids = tokenizer(
            prompt, add_special_tokens=True, return_tensors="pt"
        ).input_ids.to(model.device)
        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=args.num_generations,
                pad_token_id=tokenizer.pad_token_id,
            )
        passes = 0
        for seq in generated:
            text = tokenizer.decode(
                seq[prompt_ids.shape[1] :], skip_special_tokens=True
            )
            candidate = prompt + text
            row = []
            for test in source["public_tests"]:
                result = execute(candidate, test)
                row.append(result.outcome)
            n_pass = row.count("PASS")
            if n_pass > 0:
                passes += 1
            lines.append(
                json.dumps(
                    {
                        "problem_id": source["problem_id"],
                        "outcomes": row,
                        "n_pass": n_pass,
                        "candidate": candidate,
                    }
                )
            )
        lines.append(
            json.dumps(
                {
                    "problem_id": source["problem_id"],
                    "summary": {
                        "generations": args.num_generations,
                        "any_pass": passes,
                    },
                }
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnose.jsonl").write_text("\n".join(lines) + "\n")
    print("wrote", args.output / "diagnose.jsonl")


if __name__ == "__main__":
    main()

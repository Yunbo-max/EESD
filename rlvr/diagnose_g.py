"""Probe pass@G vs G for problem 861 on the bare base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import load_sources
from execution import execute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=861)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(0)
    cfg = yaml.safe_load(args.model_config.read_text())
    model_id, revision = cfg["model_id"], cfg["revision"]
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

    sources = load_sources(args.problems, args.invocations)
    source = next(s for s in sources if s["problem_id"] == args.pid)
    prompt = source["prompt"] + "\n"
    tests = source["public_tests"]
    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(model.device)

    lines = []
    for g in (8, 16, 32, 64):
        torch.manual_seed(0)
        with torch.no_grad():
            generated = model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=g,
                pad_token_id=tokenizer.pad_token_id,
            )
        pass1 = 0
        nonzero = 0
        for seq in generated:
            text = tokenizer.decode(
                seq[prompt_ids.shape[1] :], skip_special_tokens=True
            )
            candidate = prompt + text
            row = [execute(candidate, t).outcome for t in tests]
            pc = row.count("PASS")
            if pc == 4:
                pass1 += 1
            if pc > 0:
                nonzero += 1
        print(json.dumps({"G": g, "pass@1": pass1, "nonzero": nonzero}))
        lines.append(json.dumps({"G": g, "pass@1": pass1, "nonzero": nonzero}))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pass_vs_g.jsonl").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

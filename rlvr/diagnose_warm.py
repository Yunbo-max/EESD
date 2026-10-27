"""Check what the warm-started model actually generates (greedy + sampled)."""

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
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=861)
    args = parser.parse_args()

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
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
    base = prepare_model_for_kbit_training(
        AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    sources = load_sources(args.problems, args.invocations)
    source = next(s for s in sources if s["problem_id"] == args.pid)
    prompt = source["prompt"] + "\n"
    tests = source["public_tests"]
    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(model.device)

    lines = []
    with torch.no_grad():
        greedy = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    greedy_text = tokenizer.decode(
        greedy[0, prompt_ids.shape[1] :], skip_special_tokens=True
    )
    greedy_candidate = prompt + greedy_text
    greedy_pass = [execute(greedy_candidate, t).outcome for t in tests]
    print(json.dumps({"mode": "greedy", "pass": greedy_pass}))
    print("GREEDY CODE:\n" + greedy_candidate)
    lines.append(
        json.dumps(
            {"mode": "greedy", "pass": greedy_pass, "candidate": greedy_candidate}
        )
    )

    for temp in (0.6, 1.0):
        torch.manual_seed(0)
        with torch.no_grad():
            sampled = model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=256,
                do_sample=True,
                temperature=temp,
                top_p=0.95,
                num_return_sequences=8,
                pad_token_id=tokenizer.pad_token_id,
            )
        pass_counts = []
        for seq in sampled:
            text = tokenizer.decode(
                seq[prompt_ids.shape[1] :], skip_special_tokens=True
            )
            cand = prompt + text
            row = [execute(cand, t).outcome for t in tests]
            pass_counts.append(row.count("PASS"))
        print(json.dumps({"mode": f"sample_t{temp}", "pass_counts": pass_counts}))
        lines.append(
            json.dumps({"mode": f"sample_t{temp}", "pass_counts": pass_counts})
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "warm_check.jsonl").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

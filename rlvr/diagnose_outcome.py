"""Diagnose outcome distribution of temp-0.8 sampling with r64 adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import load_sources
from execution import execute, _extract_code


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
    base.gradient_checkpointing_disable()
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    sources = load_sources(args.problems, args.invocations)
    source = next(s for s in sources if s["problem_id"] == args.pid)
    prompt = source["prompt"] + "\n"
    tests = source["public_tests"]
    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(model.device)

    torch.manual_seed(0)
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=16,
            pad_token_id=tokenizer.pad_token_id,
            tokenizer=tokenizer,
            stop_strings=["\n\n#", "\nif __name__"],
        )

    from collections import Counter

    dist = Counter()
    samples = []
    dump = []
    for seq in generated:
        text = tokenizer.decode(seq[prompt_ids.shape[1] :], skip_special_tokens=True)
        cand = prompt + text
        first = execute(cand, tests[0])
        row = [first.outcome] + [execute(cand, t).outcome for t in tests[1:]]
        for o in row:
            dist[o] += 1
        if len(samples) < 4:
            samples.append((cand, row, _extract_code(cand), first.feedback))
        dump.append({"candidate": cand, "outcomes": row, "feedback": first.feedback})
    print(json.dumps({"outcome_dist": dict(dist)}))
    for i, (cand, row, ext, feedback) in enumerate(samples):
        print(f"=== sample {i} outcomes={row} feedback={feedback!r} ===")
        print("extracted:", repr(ext))
        print("cand tail:", repr(cand[-200:]))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "outcome.jsonl").write_text(json.dumps({"dist": dict(dist)}) + "\n")
    (args.output / "dump.jsonl").write_text(
        "\n".join(json.dumps(d) for d in dump) + "\n"
    )


if __name__ == "__main__":
    main()

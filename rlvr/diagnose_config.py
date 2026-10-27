"""Ablate which model transformation kills sampling quality.

Tests 4 configs on one problem (861) with 8 samples each:
  1. base            (bare NF4 model)
  2. base + prepare  (prepare_model_for_kbit_training only)
  3. base + lora     (get_peft_model only)
  4. base + prepare + lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import load_sources
from execution import execute


def sample(model, tokenizer, prompt: str, n: int) -> list[str]:
    import torch

    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(model.device)
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=n,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        prompt + tokenizer.decode(seq[prompt_ids.shape[1] :], skip_special_tokens=True)
        for seq in generated
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pid", type=int, default=861)
    parser.add_argument("--num-generations", type=int, default=8)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )

    lora = lambda: LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sources = load_sources(args.problems, args.invocations)
    source = next(s for s in sources if s["problem_id"] == args.pid)
    prompt = source["prompt"] + "\n"
    tests = source["public_tests"]

    configs = {}
    configs["1_base"] = load_base()
    configs["2_prepare"] = prepare_model_for_kbit_training(load_base())
    configs["3_lora"] = get_peft_model(load_base(), lora())
    configs["4_prepare_lora"] = get_peft_model(
        prepare_model_for_kbit_training(load_base()), lora()
    )

    lines = []
    for name, model in configs.items():
        candidates = sample(model, tokenizer, prompt, args.num_generations)
        pass_counts = []
        for candidate in candidates:
            row = [execute(candidate, t).outcome for t in tests]
            pass_counts.append(row.count("PASS"))
        n_nonzero = sum(1 for p in pass_counts if p > 0)
        lines.append(
            json.dumps(
                {"config": name, "pass_counts": pass_counts, "nonzero": n_nonzero}
            )
        )
        print(
            json.dumps(
                {"config": name, "nonzero": n_nonzero, "pass_counts": pass_counts}
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config_ablation.jsonl").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

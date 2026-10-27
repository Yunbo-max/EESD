"""Time generate() for different num_return_sequences to find the slowdown."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from data import load_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )

    base = prepare_model_for_kbit_training(load_base())
    base.gradient_checkpointing_disable()
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    sources = load_sources(args.problems, args.invocations)
    source = next(s for s in sources if s["problem_id"] == 788)
    prompt = source["prompt"] + "\n"
    prompt_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids.to(model.device)

    lines = []
    for g in (1, 8, 16, 64):
        torch.manual_seed(0)
        t0 = time.time()
        with torch.no_grad():
            model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=160,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=g,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.time() - t0
        print(f"G={g}: {dt:.2f}s", flush=True)
        lines.append(json.dumps({"G": g, "seconds": dt}))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "speed.jsonl").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

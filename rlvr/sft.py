"""SFT warm-start: supervised fine-tune on gold implementations before RLVR.

Trains a LoRA adapter with CE loss on each source's gold function body, so the
policy starts with task semantics instead of cold-starting on execution reward.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from data import load_sources, make_chat_messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-sources", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
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
    model = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
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
        ),
    )
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    sources = load_sources(
        args.problems, args.invocations, n_sources=args.n_sources, seed=args.seed
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        for source in sources:
            raw_prompt = source["prompt"]
            gold_body = "\n".join(source["gold"].split("\n")[1:])
            messages = make_chat_messages(raw_prompt)
            messages.append({"role": "assistant", "content": gold_body})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(
                make_chat_messages(raw_prompt),
                add_generation_prompt=True,
                tokenize=False,
            )
            encoded = tokenizer(text, add_special_tokens=True, return_tensors="pt")
            input_ids = encoded.input_ids.to(model.device)
            attention_mask = encoded.attention_mask.to(model.device)
            prompt_len = len(
                tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
            )
            labels = input_ids.clone()
            labels[:, :prompt_len] = -100
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        print(json.dumps({"epoch": epoch, "mean_loss": total_loss / len(sources)}))

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "adapter")
    report = {
        "model_id": model_id,
        "revision": revision,
        "n_sources": args.n_sources,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "problem_ids": [s["problem_id"] for s in sources],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

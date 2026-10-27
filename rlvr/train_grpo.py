"""Minimal GRPO trainer with an evidence-calibrated execution reward.

Sampling G candidates per source, executing them against the public tests with
the local sandbox, scoring them with the evidence-calibrated reward, then a
group-relative policy-gradient step with a forward-KL anchor to the frozen base.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from data import load_sources, make_chat_messages
from execution import execute
from reward import (
    evidence_calibrated_reward,
    evidence_soft_reward,
    output_similarity,
    plain_reward,
    soft_reward,
)


def pass_at_k(pass_flags, k: int) -> float:
    n = len(pass_flags)
    c = sum(pass_flags)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def diversity(candidates) -> float:
    return len(set(candidates)) / len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--invocations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reward",
        choices=("plain", "evidence", "soft", "evidence-soft"),
        default="evidence",
    )
    parser.add_argument("--previous-adapter", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n-sources", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    if args.reward == "plain":
        reward_fn = plain_reward
        use_soft = False
    elif args.reward == "evidence":
        reward_fn = evidence_calibrated_reward
        use_soft = False
    elif args.reward == "soft":
        reward_fn = soft_reward
        use_soft = True
    else:
        reward_fn = evidence_soft_reward
        use_soft = True

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

    def load_base():
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=True,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
        )

    def build_model(enable_ckpt: bool):
        base = prepare_model_for_kbit_training(load_base())
        if enable_ckpt:
            base.gradient_checkpointing_enable()
        else:
            base.gradient_checkpointing_disable()
        if args.previous_adapter is not None:
            from peft import PeftModel

            return PeftModel.from_pretrained(base, args.previous_adapter)
        return get_peft_model(
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

    model = build_model(True)
    sample_model = build_model(False)
    sample_model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora" in name:
            parameter.requires_grad_(True)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    reference = load_base()
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    sources = load_sources(
        args.problems, args.invocations, n_sources=args.n_sources, seed=args.seed
    )
    log_lines: list[str] = []

    for step in range(args.steps):
        for source in sources:
            raw_prompt = source["prompt"]
            chat_prompt = tokenizer.apply_chat_template(
                make_chat_messages(raw_prompt),
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_ids = tokenizer(
                chat_prompt, add_special_tokens=True, return_tensors="pt"
            ).input_ids.to(model.device)

            torch.manual_seed(args.seed + step * 1000 + source["problem_id"])
            stop_strings = ["\n\n#", "\nif __name__", "\n```", "<|im_end|>"]
            with torch.no_grad():
                generated = sample_model.generate(
                    prompt_ids,
                    attention_mask=torch.ones_like(prompt_ids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=0.95,
                    num_return_sequences=args.num_generations,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    tokenizer=tokenizer,
                    stop_strings=stop_strings,
                )
            full_sequences = generated.sequences
            response_texts = [
                tokenizer.decode(seq[prompt_ids.shape[1] :], skip_special_tokens=True)
                for seq in full_sequences
            ]
            candidates = [raw_prompt + "\n" + text for text in response_texts]

            tasks = [
                (candidate, test)
                for candidate in candidates
                for test in source["public_tests"]
            ]
            with ThreadPoolExecutor(max_workers=32) as pool:
                results = list(pool.map(lambda ct: execute(ct[0], ct[1]), tasks))
            if use_soft:
                scores = np.array(
                    [
                        (
                            1.0
                            if r.outcome == "PASS"
                            else (
                                output_similarity(r.actual, r.expected)
                                if r.outcome == "WRONG_OUTPUT"
                                else 0.0
                            )
                        )
                        for r in results
                    ],
                    dtype=float,
                ).reshape(len(candidates), len(source["public_tests"]))
                outcome_matrix = scores > 0.5
                rewards = reward_fn(scores)
            else:
                outcome_matrix = np.array(
                    [r.outcome == "PASS" for r in results], dtype=bool
                ).reshape(len(candidates), len(source["public_tests"]))
                rewards = reward_fn(outcome_matrix)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=model.device)

            pass_flags = outcome_matrix.all(axis=1).tolist()
            pass_k = {
                "pass@8": pass_at_k(pass_flags, 8),
                "pass@16": pass_at_k(pass_flags, 16),
                "pass@32": pass_at_k(pass_flags, 32),
                "pass@64": pass_at_k(pass_flags, 64),
            }
            div = diversity(candidates)

            model.train()
            response_ids = full_sequences[:, prompt_ids.shape[1] :]
            attention_mask = (response_ids != tokenizer.pad_token_id).long()
            batch = full_sequences.shape[0]
            prompt_mask = torch.ones(
                batch, prompt_ids.shape[1], dtype=torch.long, device=model.device
            )
            full_attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-4
            advantages = (rewards - mean_reward) / std_reward
            lengths = attention_mask.sum(dim=1).float()

            micro = 16
            ref_chunks = []
            with torch.no_grad():
                for i in range(0, batch, micro):
                    ref_outputs = reference(
                        input_ids=full_sequences[i : i + micro],
                        attention_mask=full_attention_mask[i : i + micro],
                    )
                    ref_logits = ref_outputs.logits[:, prompt_ids.shape[1] - 1 : -1]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_pt = ref_log_probs.gather(
                        2, response_ids[i : i + micro].unsqueeze(-1)
                    ).squeeze(-1)
                    ref_chunks.append(ref_pt * attention_mask[i : i + micro].float())
            ref_per_token = torch.cat(ref_chunks, dim=0)

            optimizer.zero_grad()
            total_loss = 0.0
            total_policy = 0.0
            total_kl = 0.0
            for i in range(0, batch, micro):
                mb_logits_out = model(
                    input_ids=full_sequences[i : i + micro],
                    attention_mask=full_attention_mask[i : i + micro],
                )
                mb_logits = mb_logits_out.logits[:, prompt_ids.shape[1] - 1 : -1]
                mb_log_probs = F.log_softmax(mb_logits, dim=-1)
                mb_log_prob = mb_log_probs.gather(
                    2, response_ids[i : i + micro].unsqueeze(-1)
                ).squeeze(-1)
                mb_mask = attention_mask[i : i + micro].float()
                mb_log_prob = mb_log_prob * mb_mask
                mb_seq_log_prob = mb_log_prob.sum(dim=1)

                mb_ref = ref_per_token[i : i + micro]
                log_ratio = mb_ref - mb_log_prob
                mb_kl = ((log_ratio.exp() - log_ratio - 1.0) * mb_mask).sum(dim=1)

                mb_adv = advantages[i : i + micro]
                mb_len = lengths[i : i + micro]
                mb_policy = -(mb_adv * mb_seq_log_prob / mb_len).mean()
                mb_kl_loss = (mb_kl / mb_len).mean()
                mb_loss = (mb_policy + args.beta * mb_kl_loss) / math.ceil(
                    batch / micro
                )
                mb_loss.backward()
                total_loss += float(mb_loss.detach())
                total_policy += float(mb_policy.detach())
                total_kl += float(mb_kl_loss.detach())
            optimizer.step()
            policy_loss = total_policy / math.ceil(batch / micro)
            kl_loss = total_kl / math.ceil(batch / micro)
            loss = total_loss

            log_lines.append(
                json.dumps(
                    {
                        "step": step,
                        "problem_id": source["problem_id"],
                        "rewards": rewards.tolist(),
                        "pass_counts": outcome_matrix.sum(axis=1).tolist(),
                        "pass_k": pass_k,
                        "diversity": div,
                        "policy_loss": policy_loss,
                        "kl_loss": kl_loss,
                        "loss": loss,
                    }
                )
            )
            print(
                f"step {step} pid {source['problem_id']} pass@1={pass_k['pass@64']:.3f} div={div:.3f} loss={loss:.4f}",
                flush=True,
            )

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output / "adapter")
    report = {
        "model_id": model_id,
        "revision": revision,
        "reward": args.reward,
        "n_sources": args.n_sources,
        "num_generations": args.num_generations,
        "steps": args.steps,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "problem_ids": [source["problem_id"] for source in sources],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "log.jsonl").write_text("\n".join(log_lines) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

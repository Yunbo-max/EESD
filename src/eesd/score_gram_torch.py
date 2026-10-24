"""Exact finite-rollout score Gram in a model's trainable parameter subspace.

Requires optional torch. A score is the sum of teacher-forced next-token log
probabilities of the COMPLETE sampled response, conditioned on the original
prompt token IDs. The caller must supply original token IDs from the generation
ledger; re-tokenizing displayed code is not generally equivalent.
"""
from __future__ import annotations

import numpy as np


def _tokens(seq, *, name):
    if not isinstance(seq, (list, tuple)) or not seq or any(type(t) is not int or t < 0 for t in seq):
        raise ValueError(f'{name} must be a nonempty list of nonnegative integer token IDs')
    return seq


def _trainable_params(model):
    pairs = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not pairs:
        raise ValueError('no trainable parameters; attach a frozen-checkpoint LoRA adapter')
    return pairs


def teacher_forced_sequence_logprob(model, prompt_ids, response_ids):
    """Sum log pi(response tokens | prompt), including any stored EOS token.

    An autoregressive causal LM shifts logits at position t-1 to predict token t.
    The first response token is at absolute position len(prompt_ids).
    """
    import torch
    prompt = _tokens(prompt_ids, name='prompt_ids')
    response = _tokens(response_ids, name='response_ids')
    input_device = model.get_input_embeddings().weight.device
    if input_device.type == 'meta':
        raise ValueError('model input embedding is on meta; materialize checkpoint')
    ids = torch.tensor([prompt + response], dtype=torch.long, device=input_device)
    logits = model(input_ids=ids, use_cache=False).logits
    if logits.shape[0] != 1 or logits.shape[1] != ids.shape[1]:
        raise ValueError('unexpected causal language model logits shape')
    # Work in FP32 for stable logsumexp/gather even if the model uses bf16/nf4.
    selected_logits = logits[0, len(prompt)-1:-1].float()
    target_ids = ids[0, len(prompt):]
    token_scores = selected_logits.gather(1, target_ids[:,None]).squeeze(1) - selected_logits.logsumexp(dim=-1)
    return token_scores.sum()


def score_gram_for_rollouts(model, prompt_ids, candidate_response_ids, *, max_trainable_parameters=12000000):
    """Return symmetric PSD numerical GxG Gram in the trainable adapter space.

    Computes one backward pass PER candidate. It streams gradients to CPU and
    never updates the model; expected peak CPU memory is O(G * trainable params).
    For large LoRA adapters, tune max_trainable_parameters to the hardware.
    """
    import torch
    _tokens(prompt_ids,name='prompt_ids')
    if not isinstance(candidate_response_ids,(list,tuple)) or len(candidate_response_ids)<2:
        raise ValueError('require at least two candidate responses')
    responses=[_tokens(t,name='candidate response') for t in candidate_response_ids]
    params=_trainable_params(model)
    count=sum(p.numel() for _,p in params)
    if count > max_trainable_parameters or max_trainable_parameters<=0:
        raise ValueError('trainable adapter exceeds configured memory cap')
    model.eval()
    vectors=[]
    logps=[]
    grad_params=[p for _,p in params]
    for response in responses:
        logp=teacher_forced_sequence_logprob(model,prompt_ids,response)
        grads=torch.autograd.grad(logp,grad_params,retain_graph=False,create_graph=False,allow_unused=True)
        flat=torch.cat([(torch.zeros(p.numel(),dtype=torch.float64) if grad is None else
                         grad.detach().reshape(-1).to(device='cpu',dtype=torch.float64))
                         for p,grad in zip(grad_params,grads)])
        if not torch.isfinite(flat).all() or not torch.isfinite(logp.detach()):
            raise ValueError('nonfinite score gradients or logprob')
        vectors.append(flat)
        logps.append(float(logp.detach().cpu()))
        del grads, flat, logp
    matrix=torch.stack(vectors)
    kernel=(matrix @ matrix.T).cpu().numpy().astype(np.float64,copy=False)
    kernel=(kernel+kernel.T)/2
    return {'score_kernel':kernel.tolist(),'logprobs':logps,
            'trainable_parameters':int(count),'trainable_parameter_names':[n for n,_ in params],
            'score_definition':'full_sequence_logprob_sum','score_vector_dtype':'float64_cpu',
            'model_training_mode':bool(model.training)}

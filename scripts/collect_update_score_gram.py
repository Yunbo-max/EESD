#!/usr/bin/env python3
"""Offline score-Gram collection for the PROSPECTIVE EESD update gate.

Input JSONL: source_id, prompt_ids, candidate_ids, before, after. Tokens must be
saved from the ORIGINAL generation checkpoint (including stop/EOS if sampled).
No hidden-test fields are allowed; no dataset downloads or test execution occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def read_bank(path):
    required={'source_id','prompt_ids','candidate_ids','before','after'}
    seen=set()
    rows=[]
    for number,line in enumerate(Path(path).read_text().splitlines(),1):
        if not line.strip(): continue
        row=json.loads(line)
        if set(row)!=required:
            raise ValueError(f'line {number}: expected exact fields {sorted(required)}')
        sid=row['source_id']
        if not isinstance(sid,str) or not sid or sid in seen:
            raise ValueError(f'line {number}: duplicate/invalid source_id')
        seen.add(sid)
        ids=row['candidate_ids']
        post=row['after']
        if not isinstance(ids,list) or len(ids)<2 or not isinstance(post,list) or len(post)!=len(ids):
            raise ValueError(f'line {number}: candidate_ids and after rows disagree')
        if not isinstance(row['before'],list) or not row['before'] or not all(isinstance(v,list) and len(v)==len(row['before']) for v in post):
            raise ValueError(f'line {number}: before and after test lengths disagree')
        for field in ['before']:
            if any(type(v) is not int or v not in (0,1) for v in row[field]):
                raise ValueError(f'line {number}: nonbinary before')
        if any(type(v) is not int or v not in (0,1) for rr in post for v in rr):
            raise ValueError(f'line {number}: nonbinary after')
        for token_list in [row['prompt_ids'],*ids]:
            if not isinstance(token_list,list) or not token_list or any(type(t) is not int or t<0 for t in token_list):
                raise ValueError(f'line {number}: invalid saved token IDs')
        rows.append(row)
    if not rows: raise ValueError('empty bank')
    return rows


def score_bank(args):
    input_path=Path(args.input)
    rows=read_bank(input_path)
    input_sha=hashlib.sha256(input_path.read_bytes()).hexdigest()
    if args.dry_run:
        return {'status':'validated_without_model_or_gpu','sources':len(rows),
                'max_rollouts':max(len(r['candidate_ids']) for r in rows),
                'max_sequence_tokens':max(len(r['prompt_ids'])+max(map(len,r['candidate_ids'])) for r in rows),
                'input_sha256':input_sha}

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from eesd.score_gram_torch import score_gram_for_rollouts

    output=Path(args.output)
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError('no GPU available; pass --allow-cpu only for tiny CPU debugging')
    torch.manual_seed(args.seed)
    chosen_dtype = {'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[args.dtype]
    if args.nf4 and not torch.cuda.is_available():
        raise ValueError('NF4 requires a supported GPU installation')
    model_kwargs={'local_files_only':True,'torch_dtype':chosen_dtype}
    if torch.cuda.is_available(): model_kwargs['device_map']='auto'
    if args.nf4:
        model_kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=chosen_dtype)
    base=AutoModelForCausalLM.from_pretrained(args.model_path,**model_kwargs)
    if args.adapter_path:
        model=PeftModel.from_pretrained(base,args.adapter_path,is_trainable=True,local_files_only=True)
    else:
        if args.nf4: base=prepare_model_for_kbit_training(base)
        model=get_peft_model(base,LoraConfig(r=args.lora_rank,lora_alpha=args.lora_rank,
                    lora_dropout=0.0,target_modules=['q_proj','v_proj'],bias='none'))
    model.eval()
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.eesd-gram-',suffix='.tmp',dir=output.parent)
    os.close(fd)
    try:
        with open(temp,'w') as stream:
            for row in rows:
                result=score_gram_for_rollouts(model,row['prompt_ids'],row['candidate_ids'],
                        max_trainable_parameters=args.max_trainable_parameters)
                record={k:row[k] for k in ('source_id','before','after')}
                record['score_kernel']=result['score_kernel']
                stream.write(json.dumps(record,sort_keys=True)+'\n')
                stream.flush()
        os.link(temp,output)  # Exclusive create: never clobber previous experimental bank.
    finally:
        os.unlink(temp)
    return {'status':'model_gradients_collected_not_training',
            'model_path':args.model_path,'adapter_path':args.adapter_path,
            'sources':len(rows),'output':str(output),'input_sha256':input_sha,
            'adapter_type':'loaded_existing' if args.adapter_path else 'fresh_zero_initialized_lora',
            'checkpoint_revision_must_be_locked_externally':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output')
    p.add_argument('--model-path',help='pinned local pretrained model snapshot')
    p.add_argument('--adapter-path',help='optional exact LoRA adapter checkpoint used for sampling')
    p.add_argument('--dtype',choices=['bf16','fp16','fp32'],default='bf16')
    p.add_argument('--nf4',action='store_true')
    p.add_argument('--lora-rank',type=int,default=16)
    p.add_argument('--max-trainable-parameters',type=int,default=12000000)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--allow-cpu',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if not args.dry_run and (not args.output or not args.model_path):
        p.error('--output and --model-path required for real gradient collection')
    print(json.dumps(score_bank(args),sort_keys=True,indent=2))

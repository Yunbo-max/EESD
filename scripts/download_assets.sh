#!/bin/bash
set -u
export HF_HOME=/inspire/qb-ilm2/project/multimodal-diffusion-language-model/zhangjiaquan-253108540222/global_user_mirror/hf_cache_eesd
export HF_ENDPOINT=https://hf-mirror.com
TOKEN="${HF_TOKEN:-}"
LOG=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/new/EESD/logs/qz/download_assets.log

echo "=== START $(date) ===" >> "$LOG"

download() {
  local repo="$1" type="$2" rev="$3"
  echo "=== downloading $repo ($type) @ $rev $(date) ===" >> "$LOG"
  if [ -n "$TOKEN" ]; then
    hf download "$repo" --repo-type "$type" --revision "$rev" --token "$TOKEN" >> "$LOG" 2>&1
  else
    hf download "$repo" --repo-type "$type" --revision "$rev" >> "$LOG" 2>&1
  fi
  echo "=== done $repo exit=$? $(date) ===" >> "$LOG"
}

download Qwen/Qwen2.5-Coder-7B-Instruct model c03e6d358207e414f1eca0bb1891e29f1db0e242
download deepseek-ai/deepseek-coder-6.7b-instruct model e5d64addd26a6a1db0f9b863abf6ee3141936807
download google/gemma-3-4b-it model 093f9f388b31de276ce2de164bdc2081324b9767
download giganticode/run_bug_run dataset 374251a9d65410f37e1136049cb7ff5dcca3d0ae
download Anjiang-Wei/CodeARC dataset 0e26d76e602a07a21b114fadd325f92c9da875e9

echo "=== ALL DONE $(date) ===" >> "$LOG"

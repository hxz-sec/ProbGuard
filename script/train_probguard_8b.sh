#!/usr/bin/env bash
set -euo pipefail

python probguard/train_single_guard_v8_0.py \
  --model-name ProbGuard-8B-mixed \
  --train-files data/train/train_combine_3000_qwen_16.jsonl data/train/train_combine_3000_gemma_16.jsonl data/train/train_combine_3000_llama_16.jsonl \
  --output-dir outputs/probguard_v8 \
  --log-dir logs \
  --guard-model Qwen/Qwen3-8B \
  --generation-model Qwen/Qwen3-8B \
  --generation-tokenizer Qwen/Qwen3-8B \
  --epochs 4 \
  --batch-size 16 \
  --k-min 5 \
  --k-max 15 \
  --gradient-checkpointing \
  --best-metric loss \
  --keep-only-best

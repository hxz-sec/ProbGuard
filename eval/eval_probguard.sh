#!/usr/bin/env bash
set -euo pipefail

python eval/eval_probguard_infer.py \
  --train-script scripts/train_single_guard_v8_0.py \
  --checkpoint "${1:-auto}" \
  --checkpoint-root checkpoints \
  --output-dir outputs/eval \
  --save-output-files \
  --gpu "${GPU:-auto}" \
  --batch-size 64 \
  --k-values 10

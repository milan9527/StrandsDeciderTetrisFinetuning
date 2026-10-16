#!/usr/bin/env bash
# Run on an allocated training GPU. This script never provisions or stops a host.
set -euo pipefail

: "${SFT_PYTHON:?Set SFT_PYTHON to the training environment python}"
: "${SFT_UPSTREAM:?Set SFT_UPSTREAM to the pinned upstream checkout}"
: "${SFT_CHECKPOINT:?Set SFT_CHECKPOINT to the original v19 checkpoint directory}"
: "${SFT_BASE_MODEL:?Set SFT_BASE_MODEL to the local Qwen3.5-2B-Base directory}"
SFT_DATA="${SFT_DATA:-artifacts/sft/data-v1}"
SFT_RUN="${SFT_RUN:-artifacts/sft/run-v1}"
SFT_EVAL="${SFT_EVAL:-artifacts/sft/eval-v1}"

"$SFT_PYTHON" -m finetuning.data --output "$SFT_DATA" --audit-only
"$SFT_PYTHON" -m finetuning.train \
  --upstream "$SFT_UPSTREAM" --checkpoint "$SFT_CHECKPOINT" \
  --base-model "$SFT_BASE_MODEL" --data "$SFT_DATA" --output "$SFT_RUN" \
  --epochs 1 --batch-size 2 --accumulation 8 --eval-every 200

# Both checkpoints use the same runtime, held-out questions and fresh game seeds.
"$SFT_PYTHON" -m finetuning.evaluate \
  --upstream "$SFT_UPSTREAM" --checkpoint "$SFT_RUN/source" \
  --data "$SFT_DATA" --output "$SFT_EVAL/baseline" \
  --model-name strands-decider-v19-baseline
"$SFT_PYTHON" -m finetuning.evaluate \
  --upstream "$SFT_UPSTREAM" --checkpoint "$SFT_RUN/best" \
  --data "$SFT_DATA" --output "$SFT_EVAL/adapted" --calibrate

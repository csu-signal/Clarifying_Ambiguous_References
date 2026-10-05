#!/usr/bin/env bash
# Score every checkpoint on dev (snapshots included), choose each RL method's
# checkpoint, and score the untrained policies on dev and devtest.
# Usage: scripts/evaluate.sh [device]
set -euo pipefail
DEVICE="${1:-cuda:0}"
SEEDS=(0 1 7 17 123)

for policy in natural-behavior direct-answer ask-once uncertainty-gated procot; do
    for split in dev devtest; do
        python -m promcr.evaluation.evaluate --policy "$policy" --split "$split" --device "$DEVICE"
    done
done
for split in dev devtest; do
    python -m promcr.evaluation.evaluate --checkpoint checkpoints/sft-real-synth --split "$split" --device "$DEVICE"
done
for method in grpo archer bace; do
    for seed in "${SEEDS[@]}"; do
        python -m promcr.evaluation.evaluate --checkpoint "checkpoints/$method-seed$seed" --snapshots --split dev --device "$DEVICE"
        python -m promcr.evaluation.evaluate --checkpoint "checkpoints/$method-seed$seed" --split devtest --device "$DEVICE"
    done
    python -m promcr.evaluation.select checkpoints/$method-seed{0,1,7,17,123} --reference results/sft-real-synth/final \
        | tee "results/$method-selection.md"
done
echo "Score the chosen points on devtest with: python -m promcr.evaluation.evaluate --checkpoint checkpoints/<run>/snapshots/<point> --split devtest"

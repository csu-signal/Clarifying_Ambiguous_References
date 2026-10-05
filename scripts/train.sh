#!/usr/bin/env bash
# Train every method: SFT once, then five seeds of each RL method from it.
# Usage: scripts/train.sh [sft|grpo|archer|bace|all] [device]
set -euo pipefail
WHAT="${1:-all}"
DEVICE="${2:-cuda:0}"
SEEDS=(0 1 7 17 123)

if [[ "$WHAT" == sft || "$WHAT" == all ]]; then
    python -m promcr.methods.sft.train --config configs/sft.yaml --output checkpoints/sft-real-synth --device "$DEVICE"
fi
for method in grpo archer bace; do
    if [[ "$WHAT" == "$method" || "$WHAT" == all ]]; then
        for seed in "${SEEDS[@]}"; do
            python -m "promcr.methods.$method.train" --config "configs/$method.yaml" --seed "$seed" \
                --output "checkpoints/$method-seed$seed" --device "$DEVICE"
        done
    fi
done

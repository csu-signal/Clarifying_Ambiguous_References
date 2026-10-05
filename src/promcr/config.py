"""Paths, backbone and environment constants shared by every module.

Paths are relative to the working directory, so commands are meant to run
from the repository root. `PROMCR_DATA` and `PROMCR_MODEL` override them.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("PROMCR_DATA", "data"))
SIMMC_DIR = DATA_DIR / "simmc2"  # raw SIMMC 2.1 files (scripts/download_simmc.sh)
LABELS_DIR = DATA_DIR / "labels"  # generated labels, committed

# A Hugging Face model id or a local directory.
MODEL_NAME = os.environ.get("PROMCR_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Reward and episode settings of every reported run: +1 / -1 for the answer,
# -LAMBDA per question, at most BUDGET questions.
LAMBDA = 0.1
BUDGET = 3

SPLITS = ("train", "dev", "devtest")

"""Rebuild the label files under data/labels from the raw SIMMC 2.1 data.

    python -m promcr.data.build conditions   # conditions_{split}.jsonl
    python -m promcr.data.build splitters    # splitters_{split}.jsonl (needs conditions)
    python -m promcr.data.build synthetic    # synthetic_anchored_train.jsonl (needs both)
    python -m promcr.data.build all
"""

from __future__ import annotations

import argparse
from collections import Counter

from ..config import SPLITS
from ..env.oracle import build_gold_chain
from .conditions import write_conditions
from .splitters import write_splitters
from .synthetic import depth_targets, generate_anchored_episodes, write_synthetic


def build_synthetic(split: str = "train", seed: int = 0, max_per_template: int = 20) -> None:
    targets = depth_targets(split)
    episodes, sources, rejects = generate_anchored_episodes(split, targets, seed=seed, max_per_template=max_per_template)
    depths = Counter(len(build_gold_chain(e, 3).steps) for e in episodes)
    print(f"[{split}] targets {dict(sorted(targets.items()))}")
    print(f"[{split}] generated {len(episodes)} episodes, by depth {dict(sorted(depths.items()))}; rejects {rejects}")
    print(f"  wrote {write_synthetic(split, episodes, sources)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("conditions", "splitters", "synthetic", "all"))
    parser.add_argument("--seed", type=int, default=0, help="synthetic generation seed")
    args = parser.parse_args(argv)
    if args.step in ("conditions", "all"):
        for split in SPLITS:
            print(f"wrote {write_conditions(split)}")
    if args.step in ("splitters", "all"):
        for split in SPLITS:
            print(f"wrote {write_splitters(split)}")
    if args.step in ("synthetic", "all"):
        build_synthetic("train", seed=args.seed)


if __name__ == "__main__":
    main()

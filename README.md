# ProMCR

Learning when to ask instead of answering. A user refers to an object in a
shopping scene, several objects could match, and the policy either names
one or spends a turn on a clarifying question. Language models tend to
answer too quickly, so methods are compared on when they ask and when they
stop asking, not only on accuracy.

Episodes come from SIMMC 2.1. Every method runs on Qwen2.5-7B-Instruct; the
trained ones add LoRA adapters. The repository covers:

- the data pipeline, from the raw SIMMC 2.1 files to episodes and training pools;
- the untrained policies: Natural Behavior and four prompted variants;
- the trained methods: SFT, trajectory-level GRPO, ArCHer, and BACE (a
  value-based chooser over the SFT model's own question and answer);
- the evaluation: metrics, calibration and critic diagnostics, confidence
  intervals and paired tests, checkpoint selection, and comparison tables.

## The task

An episode is a user turn that refers to an object. The policy sees the
dialogue history, the utterance, and the candidate objects with their
non-visual attributes (brand, price, customer review, available sizes, size),
their position in the scene and the scene relations. Visual attributes are
withheld. At each turn it either answers (`ANSWER: <id>`) or asks
(`ASK: <attribute> | <question>`), and a scripted user replies with the
referent's value. A correct answer earns +1 and a wrong one -1; each question
costs 0.1, and at most three questions are allowed. Each attribute can be
asked once.

Each episode's gold depth is the number of questions an oracle that knows the
referent needs: it keeps asking the attribute that best splits the remaining
candidates. Results are broken down by gold depth.

| Gold depth | Train | Dev | Devtest | Synthetic (train) |
| --- | --- | --- | --- | --- |
| 0 | 7,405 | 617 | 1,701 | -- |
| 1 | 2,613 | 247 | 550 | 4,792 |
| 2 | 840 | 74 | 184 | 6,564 |
| 3 | 162 | 18 | 33 | 7,242 |
| 4-5 | 15 | 3 | 3 | -- |
| unresolvable | 47 | 1 | 12 | -- |
| Total | 11,082 | 960 | 2,483 | 18,598 |

Synthetic episodes keep a real ambiguous turn's scene, history and utterance
and take a strict subset of its annotated candidates, with any of them as the
referent. They are used only to train SFT.

## Layout

```
configs/                 one YAML file per trained method: the reported recipe
data/labels/             generated labels (committed): conditions, splitters, anchored synthetic episodes
scripts/                 download SIMMC, rebuild the labels, train everything, evaluate everything
src/promcr/
  config.py              paths, backbone name, reward and budget
  data/                  SIMMC 2.1 -> labels -> episodes -> training pools
    simmc.py             raw dialogue, scene and metadata readers
    conditions.py        candidates, history constraints, survivors, gold referent
    splitters.py         information gain of every attribute on the survivors
    episodes.py          Episode and its rendered observation
    synthetic.py         anchored synthetic episodes
    pools.py             the `real` and `real+synth` training pools
    build.py             CLI that writes data/labels
  env/                   the MDP
    environment.py       MCREnv: rules, reward, refusals
    simulator.py         the scripted user
    oracle.py            the gold chain and gold depth
    attributes.py        askable attributes and template questions
  models/                backbone and adapter loading, sampling and scoring helpers
  policies/              output contract and parsers, prompted policies, the decision-line policy
  methods/
    common.py            config loading, seeding, LR schedule, progress logs
    chooser.py           what ArCHer and BACE share: branch prompt, critic features, decision rule
    sft/                 dataset and trainer
    grpo/                trainer
    archer/              heads, rollout, updates, trainer, evaluation policy
    bace/                belief, features, heads, rollout, warm start, update, trainer, evaluation policy
  evaluation/
    evaluate.py          CLI: score a policy on a split
    rollout.py           runs a policy through the environment, seeded per episode
    metrics.py           when to ask, when to stop, question quality
    calibration.py       decision calibration and critic diagnostics
    probe.py             the policy's referent belief and ask score at each decision
    records.py           one record per episode
    stats.py             bootstrap intervals, paired tests, Holm adjustment
    select.py            CLI: the checkpoint-selection rule
    report.py            CLI: comparison tables, paired comparisons, curves
```

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
scripts/download_simmc.sh        # SIMMC 2.1 dialogues, metadata and scene files into data/simmc2 (needs git-lfs)
```

The backbone is `Qwen/Qwen2.5-7B-Instruct` from the Hugging Face Hub. To use a
local copy, set `PROMCR_MODEL=/path/to/model`. Commands are run from the
repository root.

## Data

The label files under `data/labels` are committed, so nothing needs
rebuilding. `scripts/build_data.sh` rebuilds them from the raw files and
gives the same bytes:

```bash
python -m promcr.data.build conditions   # which turns are episodes, candidates, survivors, gold referent
python -m promcr.data.build splitters    # the best first question of each ambiguous episode
python -m promcr.data.build synthetic    # anchored synthetic episodes (seed 0)
```

Evaluation always uses the real dev and devtest episodes, unsolvable ones
included. Training uses the `real` pool (11,035 episodes that some question
sequence can resolve) or, for SFT, `real+synth` (29,633).

## Training

Each trainer takes a YAML config, and any field can be overridden on the
command line (`--seed 1`, `--iterations 4`, ...). The configs hold the
reported recipes.

```bash
python -m promcr.methods.sft.train    --config configs/sft.yaml    --output checkpoints/sft-real-synth
python -m promcr.methods.grpo.train   --config configs/grpo.yaml   --seed 0 --output checkpoints/grpo-seed0
python -m promcr.methods.archer.train --config configs/archer.yaml --seed 0 --output checkpoints/archer-seed0
python -m promcr.methods.bace.train   --config configs/bace.yaml   --seed 0 --output checkpoints/bace-seed0
```

`scripts/train.sh` runs SFT and then seeds 0, 1, 7, 17 and 123 of each RL
method. Every RL method starts from `checkpoints/sft-real-synth`: its adapter
is merged into the backbone, the method trains new adapters on top, and the
SFT policy is the reference for the KL penalties.

| Method | What is trained | Data | Time on one RTX PRO 6000 |
| --- | --- | --- | --- |
| SFT | policy LoRA (r=16) | one epoch of `real+synth`, 54,405 examples | 3.3 h |
| GRPO | policy LoRA (r=16) | one pass of `real`, groups of 4 rollouts | 3.3 h |
| ArCHer | actor LoRA, critic LoRA (r=8), two Q/V head pairs, token baseline | one pass of `real`, 28 iterations | 5.2 h |
| BACE | critic LoRA (r=8), two Q/V head pairs | 150 warm-start + 160 RL episodes | 0.5 h |

Runs save a resumable state as they go (SFT every 100 optimizer steps, GRPO
after every batch, ArCHer and BACE after every iteration). After a crash,
rerun the same command with `--resume`; the run continues where the state was
saved, with the same random stream, and ends as an uninterrupted run would.

A checkpoint directory holds the trained adapters (`actor/` and `critic/` for
ArCHer and BACE, with their Q heads), `warm_start.json` (the SFT checkpoint
it sits on), `config.json`, `train_metrics.jsonl` and
`snapshot_progress.json` (episodes and optimizer steps so far). Snapshots
taken during training are in `snapshots/` with the same layout.

BACE checks its warm-started chooser before RL: its greedy ask rate on the
warm-start states must lie in [0.25, 0.60] and its Q gap must agree in sign
with the measured advantage on at least 60% of them; otherwise the run stops.
The earlier BACE recipe, which scores spatial questions as giving no
information, is `--psi0 permitted --belief-sizes overlap --exact-feature true`.

## Evaluation

```bash
python -m promcr.evaluation.evaluate --policy natural-behavior --split dev
python -m promcr.evaluation.evaluate --checkpoint checkpoints/sft-real-synth --split dev
python -m promcr.evaluation.evaluate --checkpoint checkpoints/bace-seed0 --snapshots --split dev
```

Policies decode greedily; BACE's sampled candidates are seeded per episode, so
every checkpoint sees the same draw on a given episode. Each evaluation writes
`results/<run>/<point>/<split>.json` and `<split>.episodes.jsonl`.

The metrics:

| Group | Metrics |
| --- | --- |
| Accuracy and cost | reward, accuracy, questions per episode |
| When to ask | precision, recall and F1 of asking on ambiguous episodes, over-asking, under-asking, hallucinated certainty |
| When to stop | hit (as many questions as the oracle), early, late, hit-and-correct, per ask-k group |
| Questions | entropy reduction, redundancy, IG ratio and IG-best rate, unparsed asks |
| Calibration | answer ECE, AUROC of confidence for wrong and ambiguous decisions, ask-score AUROC |
| Critic | step-0 gap SD, gap AUROC, explained variance, value calibration error, twin gap, out-of-range Q |

A method's reported checkpoint is chosen on dev, from all of its seeds'
snapshots, then scored on devtest:

```bash
python -m promcr.evaluation.select checkpoints/bace-seed{0,1,7,17,123} --reference results/sft-real-synth/final
python -m promcr.evaluation.evaluate --checkpoint checkpoints/bace-seed0/snapshots/step_004 --split devtest
```

The rule keeps points that pass health gates (Ask F1 >= 0.95, over-asking
<= 0.02, unparsed asks <= 0.03, critic Q in range, step-0 gap SD >= 0.10),
scores six metric families by percentile among the candidates, and ranks
points by their weakest family. It reports the best single point and the best
point shared by all seeds.

Tables and tests from the episode records, mean (SD) over seeds:

```bash
python -m promcr.evaluation.report table \
    --row "Natural Behavior=results/natural-behavior/final" \
    --row "SFT=results/sft-real-synth/final" \
    --row "BACE=results/bace-seed0/step_004,results/bace-seed1/step_004"
python -m promcr.evaluation.report groups --row ...
python -m promcr.evaluation.report compare results/sft-real-synth/final results/bace-seed0/step_004 --split devtest
python -m promcr.evaluation.report curves --run checkpoints/bace-seed0 --metric ask_f1 --out ask_f1.png
```

## Reproducing the reported results

```bash
scripts/download_simmc.sh
scripts/train.sh all
scripts/evaluate.sh
```

then score each method's chosen point on devtest and build the tables with
`report`. Results are not bit-identical across GPU models: on the compound
slice of multi-question episodes, differences under about 2 points are within
that noise.

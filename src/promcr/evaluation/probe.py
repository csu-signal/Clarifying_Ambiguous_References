"""What the policy believed at each decision, read before it acts.

- Referent belief: the policy's own model scored under one fixed prompt, the
  length-normalised log-likelihood of " object_id: <id>" for each candidate,
  softmaxed (the same reading Uncertainty-gated decides on).
- Ask score: P("ASK:") / (P("ASK:") + P("ANSWER:")) under the decision
  prompt for the decision-line policies (SFT, GRPO). Critic policies record
  their Q values instead (`StepLog.decision_q`), and prompted policies have none.

The probe does no sampling, so it doesn't change the actions or any RNG.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..models.generation import option_logprobs_cached
from ..policies.contract import DECISION_INSTRUCTION
from ..policies.decision import DecisionPolicy
from ..policies.prompted import CONFIDENCE_INSTRUCTION


@dataclass
class StepConfidence:
    referent_probs: dict[int, float]
    ask_score: float | None = None


def _softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    return [e / sum(exps) for e in exps]


def confidence_probe(policy, episode, observation: str, candidate_ids: list[int]) -> StepConfidence | None:
    backbone = getattr(policy, "backbone", None)
    if backbone is None:
        return None
    model, tokenizer, device = backbone.model, backbone.tokenizer, backbone.device
    options = [f" object_id: {cid}" for cid in candidate_ids]
    scores = option_logprobs_cached(model, tokenizer, device, f"{CONFIDENCE_INSTRUCTION}\n\n{observation}", options)
    lengths = [len(tokenizer(option, add_special_tokens=False).input_ids) for option in options]
    confidence = StepConfidence(dict(zip(candidate_ids, _softmax([s / max(1, n) for s, n in zip(scores, lengths)]))))
    if isinstance(policy, DecisionPolicy):
        ask, answer = option_logprobs_cached(model, tokenizer, device, f"{DECISION_INSTRUCTION}\n\n{observation}",
                                             ["ASK:", "ANSWER:"])
        confidence.ask_score = 1.0 / (1.0 + math.exp(answer - ask))
    return confidence

"""Sampling and scoring helpers shared by the trainers and the policies.

Generation settings matter for reproducing the reported numbers. Qwen's
`generation_config.json` sets a repetition penalty of 1.05 (and top-p/top-k
for sampling). `chat` and the constrained-attribute decoder leave it in
place; `sample_completions`, which every RL trainer samples branches with,
sets it to 1.0.
"""

from __future__ import annotations

import copy

import torch

# Options scored per batched forward in `option_logprobs_cached`.
OPTION_CHUNK = 16


def chat_prompt(tokenizer, content: str) -> str:
    """A single user message in the chat template, ready for the assistant's reply."""
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True)


def chat(model, tokenizer, device: str, content: str, max_new_tokens: int = 128) -> str:
    """Greedy reply to one user message."""
    inputs = tokenizer(chat_prompt(tokenizer, content), return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def sample_completions(model, tokenizer, device: str, prompt: str, num_samples: int, max_new_tokens: int,
                       temperature: float) -> list[tuple[str, list[int]]]:
    """(text, token ids) for `num_samples` continuations of `prompt`, in one
    batched call. Greedy when `temperature` is 0. Ids stop at the first pad."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
        )
    prompt_len = inputs["input_ids"].shape[1]
    results = []
    for row in output:
        ids = row[prompt_len:].tolist()
        if tokenizer.pad_token_id in ids:
            ids = ids[: ids.index(tokenizer.pad_token_id)]
        results.append((tokenizer.decode(ids, skip_special_tokens=True).strip(), ids))
    return results


def completion_token_logprobs(model, tokenizer, device: str, prompt: str, completion_ids: list[int]) -> torch.Tensor:
    """Per-token log-probabilities of `completion_ids` after `prompt`, with gradients."""
    completion = torch.tensor([completion_ids], dtype=torch.long, device=device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([prompt_ids, completion], dim=1)
    log_probs = torch.log_softmax(model(full_ids).logits[0, :-1].float(), dim=-1)
    n = completion.shape[1]
    return log_probs[-n:].gather(-1, full_ids[0, 1:][-n:].unsqueeze(-1)).squeeze(-1)


def reference_token_logprobs(model, tokenizer, device: str, prompt: str, completion_ids: list[int]) -> torch.Tensor:
    """Per-token log-probabilities under the reference policy: the model with
    every LoRA adapter disabled, which is the merged SFT policy for a
    warm-started run. No gradients, no dropout."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), model.disable_adapter():
            return completion_token_logprobs(model, tokenizer, device, prompt, completion_ids)
    finally:
        model.train(was_training)


def kl_k3(new_logprobs: torch.Tensor, ref_logprobs: torch.Tensor, clip: float = 5.0, reduction: str = "mean") -> torch.Tensor:
    """Schulman's k3 estimate of KL(pi || pi_ref) per token, log-ratios clipped to +-`clip`."""
    log_ratio = torch.clamp(ref_logprobs - new_logprobs, min=-clip, max=clip)
    per_token = torch.exp(log_ratio) - log_ratio - 1
    return per_token.sum() if reduction == "sum" else per_token.mean()


def option_logprobs(model, tokenizer, device: str, content: str, options: list[str]) -> list[float]:
    """Summed log-probability of each option as the reply to `content`, one
    full forward per option. Uncertainty-gated decides on these."""
    prefix_ids = tokenizer(chat_prompt(tokenizer, content), return_tensors="pt").input_ids.to(device)
    scores = []
    for option in options:
        option_ids = tokenizer(option, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        full_ids = torch.cat([prefix_ids, option_ids], dim=1)
        with torch.no_grad():
            log_probs = torch.log_softmax(model(full_ids).logits[0, :-1].float(), dim=-1)
        n = option_ids.shape[1]
        scores.append(log_probs[-n:].gather(-1, full_ids[0, 1:][-n:].unsqueeze(-1)).squeeze(-1).sum().item())
    return scores


def option_logprobs_cached(model, tokenizer, device: str, content: str, options: list[str]) -> list[float]:
    """The same quantity as `option_logprobs` from one prefix forward and a
    shared KV cache, for the evaluation probe and SFT's ask/answer reading."""
    prefix_ids = tokenizer(chat_prompt(tokenizer, content), return_tensors="pt").input_ids.to(device)
    option_ids = [tokenizer(option, add_special_tokens=False).input_ids for option in options]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    scores: list[float] = []
    with torch.no_grad():
        prefix = model(prefix_ids, use_cache=True, logits_to_keep=1)
        first = torch.log_softmax(prefix.logits[0, -1].float(), dim=-1)
        for start in range(0, len(options), OPTION_CHUNK):
            chunk = option_ids[start : start + OPTION_CHUNK]
            width = max(len(ids) for ids in chunk)
            chunk_scores = [first[ids[0]].item() for ids in chunk]
            if width > 1:
                cache = copy.deepcopy(prefix.past_key_values)
                cache.batch_repeat_interleave(len(chunk))
                ids = torch.tensor([row + [pad_id] * (width - len(row)) for row in chunk], device=device)
                mask = torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in chunk], device=device)
                attention = torch.cat([torch.ones(len(chunk), prefix_ids.shape[1], dtype=mask.dtype, device=device), mask], 1)
                logits = model(ids, past_key_values=cache, attention_mask=attention, use_cache=True).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                for i, row in enumerate(chunk):
                    for j in range(1, len(row)):
                        chunk_scores[i] += log_probs[i, j - 1, row[j]].item()
            scores.extend(chunk_scores)
    return scores


def last_hidden_state(model, tokenizer, device: str, prompt: str, completion_ids: list[int] | None = None) -> torch.Tensor:
    """Final-layer hidden state at the last token of `prompt` (+ `completion_ids`), as float32.
    Gradients flow when the caller doesn't wrap it in no_grad."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    if completion_ids is not None:
        ids = torch.cat([ids, torch.tensor([completion_ids], dtype=torch.long, device=device)], dim=1)
    return model(ids, output_hidden_states=True).hidden_states[-1][0, -1].float()

"""Loading the backbone, LoRA adapters and the SFT warm start they sit on.

Every RL trainer merges the SFT adapter into the frozen base before
attaching its own adapters, and saves only those. A checkpoint therefore
records the SFT checkpoint it was trained on in `warm_start.json`, and every
loader merges it back in before attaching the checkpoint's adapters.
Relative paths are resolved against the working directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import MODEL_NAME
from . import generation

WARM_START_FILE = "warm_start.json"
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_config(r: int, alpha: int, dropout: float) -> LoraConfig:
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=r, lora_alpha=alpha, lora_dropout=dropout,
                      target_modules=list(LORA_TARGET_MODULES))


def load_tokenizer(path: str | Path = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(device: str, sft_checkpoint: str | Path | None = None):
    """The bf16 backbone, with `sft_checkpoint`'s adapter merged in when given."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    if sft_checkpoint is not None:
        model = PeftModel.from_pretrained(model, str(sft_checkpoint)).merge_and_unload()
    return model


def record_warm_start(output_dir: str | Path, sft_checkpoint: str | Path | None) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"backbone": MODEL_NAME, "sft_checkpoint": str(sft_checkpoint) if sft_checkpoint is not None else None}
    (output_dir / WARM_START_FILE).write_text(json.dumps(payload, indent=2) + "\n")


def warm_start_chain(checkpoint_dir: str | Path) -> list[Path]:
    """Adapters to merge into the backbone, in training order, before `checkpoint_dir`'s own."""
    chain: list[Path] = []
    current = Path(checkpoint_dir)
    while (current / WARM_START_FILE).exists():
        parent = json.loads((current / WARM_START_FILE).read_text()).get("sft_checkpoint")
        if not parent:
            break
        current = Path(parent)
        if current in chain or len(chain) > 8:
            raise ValueError(f"cycle in the warm-start chain of {checkpoint_dir}")
        chain.append(current)
    return list(reversed(chain))


def load_warm_started_base(checkpoint_dir: str | Path, device: str):
    """The base model `checkpoint_dir`'s adapters were trained on."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    for adapter in warm_start_chain(checkpoint_dir):
        if not adapter.exists():
            raise FileNotFoundError(f"{checkpoint_dir} was trained on top of {adapter}, which does not exist")
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    return model


@dataclass
class Backbone:
    """A model and its tokenizer, as the policies use them."""

    model: object
    tokenizer: object
    device: str

    def chat(self, content: str, max_new_tokens: int = 128) -> str:
        return generation.chat(self.model, self.tokenizer, self.device, content, max_new_tokens)

    def option_logprobs(self, content: str, options: list[str]) -> list[float]:
        return generation.option_logprobs(self.model, self.tokenizer, self.device, content, options)


def load_backbone(device: str = "cuda:0") -> Backbone:
    """The untrained instruct model, for the prompted policies."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to(device)
    model.eval()
    return Backbone(model, AutoTokenizer.from_pretrained(MODEL_NAME), device)


def load_merged_adapter(checkpoint_dir: str | Path, device: str = "cuda:0") -> Backbone:
    """A single-adapter checkpoint (SFT, GRPO) merged into its warm-started base."""
    base = load_warm_started_base(checkpoint_dir, device)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir)).merge_and_unload()
    model.eval()
    return Backbone(model, AutoTokenizer.from_pretrained(str(checkpoint_dir)), device)


def load_two_adapters(checkpoint_dir: str | Path, device: str = "cuda:0"):
    """The `actor` and `critic` adapters of an ArCHer or BACE checkpoint,
    attached side by side so the policy can switch between them."""
    checkpoint_dir = Path(checkpoint_dir)
    base = load_warm_started_base(checkpoint_dir, device)
    model = PeftModel.from_pretrained(base, str(checkpoint_dir / "actor"), adapter_name="actor")
    model.load_adapter(str(checkpoint_dir / "critic"), adapter_name="critic")
    model.set_adapter("actor")
    model.eval()
    return model, AutoTokenizer.from_pretrained(str(checkpoint_dir))

"""ArCHer's value heads and token-level baseline, with their save format."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..chooser import NormedHead

HEAD_FILES = ("critic.pt", "critic_config.json")


def save_head(head: NormedHead, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), directory / "critic.pt")
    config = {"kind": "normed", "hidden_size": head.hidden_size, "mlp_hidden": head.mlp_hidden}
    (directory / "critic_config.json").write_text(json.dumps(config))


def load_head(directory: str | Path, device: str) -> NormedHead:
    directory = Path(directory)
    config = json.loads((directory / "critic_config.json").read_text())
    if config.get("kind") != "normed":
        raise ValueError(f"{directory} holds a {config.get('kind', 'plain')!r} head; only 'normed' heads are supported")
    head = NormedHead(config["hidden_size"], config["mlp_hidden"]).to(device)
    head.load_state_dict(torch.load(directory / "critic.pt", map_location=device))
    head.eval()
    return head


class TokenBaseline(torch.nn.Module):
    """V~(s, a^{1:i-1}): a per-token baseline for the actor's policy gradient (ArCHer Sec. 3.5)."""

    def __init__(self, hidden_size: int, mlp_hidden: int = 256) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(hidden_size, mlp_hidden), torch.nn.ReLU(),
                                       torch.nn.Linear(mlp_hidden, 1))

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        return self.net(token_features).squeeze(-1)


def token_features(model, tokenizer, device: str, prompt: str, completion_ids: list[int]) -> torch.Tensor:
    """For completion token i, the hidden state just before it (so the baseline
    never sees the token it is subtracted from). No gradients."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = torch.cat([prompt_ids, torch.tensor([completion_ids], dtype=torch.long, device=device)], dim=1)
    with torch.no_grad():
        hidden = model(full_ids, output_hidden_states=True).hidden_states[-1][0]
    return hidden[-len(completion_ids) - 1 : -1].detach().float()

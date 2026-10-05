"""The chooser's value heads: a hidden state and a small block of belief features.

Concatenated directly, the 3,584 hidden dimensions drown out the eight
belief features, and the chooser can't tell ambiguous states from resolved
ones. Each head therefore layer-normalises the hidden state and projects the
scalar block to `scalar_width` through its own layer before the two meet.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class BeliefHead(nn.Module):
    """Scalar head over `[hidden ; scalars]`; `hidden_size` is the total input width."""

    def __init__(self, llm_hidden: int, n_scalar: int, mlp_hidden: int = 512, scalar_width: int = 64) -> None:
        super().__init__()
        self.llm_hidden, self.n_scalar = llm_hidden, n_scalar
        self.hidden_size = llm_hidden + n_scalar
        self.mlp_hidden, self.scalar_width = mlp_hidden, scalar_width
        self.llm_norm = nn.LayerNorm(llm_hidden)
        self.scalar_proj = nn.Sequential(nn.Linear(n_scalar, scalar_width), nn.ReLU())
        self.net = nn.Sequential(nn.Linear(llm_hidden + scalar_width, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
        nn.init.normal_(self.net[-1].weight, std=1e-3)  # start near Q = 0, but not exactly constant
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        llm, scalar = features[..., : self.llm_hidden], features[..., self.llm_hidden :]
        return self.net(torch.cat([self.llm_norm(llm), self.scalar_proj(scalar)], dim=-1)).squeeze(-1)


def save_head(head: BeliefHead, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), directory / "critic.pt")
    config = {"kind": "belief", "hidden_size": head.hidden_size, "llm_hidden": head.llm_hidden,
              "n_scalar": head.n_scalar, "mlp_hidden": head.mlp_hidden, "scalar_width": head.scalar_width}
    (directory / "critic_config.json").write_text(json.dumps(config))


def load_head(directory: str | Path, device: str) -> BeliefHead:
    directory = Path(directory)
    config = json.loads((directory / "critic_config.json").read_text())
    if config.get("kind") != "belief":
        raise ValueError(f"{directory} holds a {config.get('kind', 'plain')!r} head; only 'belief' heads are supported")
    head = BeliefHead(config["llm_hidden"], config["n_scalar"], config["mlp_hidden"], config["scalar_width"]).to(device)
    head.load_state_dict(torch.load(directory / "critic.pt", map_location=device))
    head.eval()
    return head

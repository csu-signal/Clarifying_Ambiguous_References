"""The two high-level actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Ask:
    attribute: str  # what the question is about, from ASK_ATTRIBUTES (or UNRESOLVED_ASK_ATTRIBUTE)
    question: str


@dataclass(frozen=True)
class Answer:
    referent_id: int


Action = Union[Ask, Answer]

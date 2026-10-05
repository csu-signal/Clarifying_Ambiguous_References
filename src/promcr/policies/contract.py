"""The output contract every trained policy writes, and the parsers that turn text into actions.

A decision is one line: "ANSWER: <id>" or "ASK: <attribute> | <question>",
with the attribute copied from the menu. When an ask doesn't follow the
contract, its attribute is guessed from keywords in the question; a question
nothing matches becomes an unresolved ask, which spends a turn and reveals
nothing.
"""

from __future__ import annotations

import logging
import re

from ..env.actions import Action, Answer, Ask
from ..env.attributes import (
    ASK_ATTRIBUTES,
    SPATIAL_BIN_ATTRS,
    SPATIAL_GRAPH_ATTRS,
    TEXT_ASKABLE_ATTRS,
    UNRESOLVED_ASK_ATTRIBUTE,
    question_for_attribute,
)

logger = logging.getLogger(__name__)

ASK_ATTRIBUTE_MENU = """\
  availableSizes, brand, customerReview, price, size
      -- properties of the object itself
  left_right, up_down
      -- which half of the scene the object sits in
  left, right, up, down
      -- whether another candidate sits on that side of it"""

_MENU_NAMES = {name.strip() for line in ASK_ATTRIBUTE_MENU.splitlines()
               if not line.lstrip().startswith("--") for name in line.split(",")}
assert _MENU_NAMES == set(TEXT_ASKABLE_ATTRS), "ASK_ATTRIBUTE_MENU and TEXT_ASKABLE_ATTRS disagree"

ASK_CONTRACT = f"""\
Write the clarifying question in exactly this form:
    <attribute> | <question>
<attribute> must be copied character-for-character from this list:
{ASK_ATTRIBUTE_MENU}
Choose the attribute whose values differ most among the candidates. <question> \
must be one natural question to the customer about that attribute. Never invent \
an attribute name, and never name an object id or number inside the question.
Format example: color | What colour is the jacket you mean?"""

# The instruction SFT is trained on and GRPO keeps.
DECISION_INSTRUCTION = f"""\
You are resolving which object a user is referring to among several candidates. \
Decide whether the request is ambiguous among the candidates below and, if so, \
whether asking would help. Respond with exactly one line and nothing else:
- "ANSWER: <object id>" if you can resolve it (or must just guess) -- the id number alone.
- "ASK: <attribute> | <clarification question>" if it is genuinely ambiguous and asking would help.

{ASK_CONTRACT}"""

# Checked in order, so "available sizes" is caught before "size".
_ATTRIBUTE_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("availableSizes", ("available size", "sizes available", "what sizes", "in stock")),
    ("size", ("size",)),
    ("brand", ("brand", "make")),
    ("price", ("price", "cost", "expensive", "cheap", "how much")),
    ("customerReview", ("review", "rating", "rated")),
    ("color", ("color", "colour")),
    ("pattern", ("pattern", "print")),
    ("sleeveLength", ("sleeve",)),
    ("type", ("type", "kind", "category")),
    ("assetType", ("look like", "appearance")),
    ("left_right", ("left", "right")),
    ("up_down", ("up", "down", "top", "bottom", "higher", "lower")),
)
_ATTRIBUTE_NAMES = {name.lower(): name for name in ASK_ATTRIBUTES}
_ID_PATTERN = re.compile(r"\b(\d+)\b")
_DECISION_PATTERN = re.compile(r"(ASK|ANSWER)\s*:\s*(.+)", re.IGNORECASE)

# Asks whose attribute had to be guessed from keywords and found none. Read
# around each `act` by the evaluation rollout (the "unparsed asks" metric).
_unparsed_asks = 0


def unparsed_ask_count() -> int:
    return _unparsed_asks


def _keyword_attribute(text: str) -> str | None:
    lowered = text.lower()
    for attribute, keywords in _ATTRIBUTE_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return attribute
    return None


def _coherent_question(attribute: str, question: str) -> str:
    """The question, unless its wording plainly asks about another attribute;
    then the template question for `attribute`, so the reply the policy reads
    answers the question it sees. Spatial wordings are interchangeable."""
    inferred = _keyword_attribute(question)
    if inferred is None or inferred == attribute:
        return question
    if {inferred, attribute} <= set(SPATIAL_BIN_ATTRS) | set(SPATIAL_GRAPH_ATTRS):
        return question
    return question_for_attribute(attribute)


def resolve_ask(text: str) -> tuple[str, str]:
    """(attribute, question) from an ask: the "<attribute> | <question>"
    contract when it parses, the keyword guess otherwise."""
    global _unparsed_asks
    attribute, _, question = text.partition("|")
    declared = _ATTRIBUTE_NAMES.get(attribute.strip().rstrip(":").lower())
    if declared is not None and question.strip():
        return declared, _coherent_question(declared, question.strip())
    guessed = _keyword_attribute(text)
    if guessed is None:
        _unparsed_asks += 1
        logger.debug("no attribute found in ask %r", text)
        return UNRESOLVED_ASK_ATTRIBUTE, text
    return guessed, text


def parse_referent(text: str, candidate_ids: list[int]) -> int:
    """The first candidate id in `text`, or the first candidate if none appears."""
    for match in _ID_PATTERN.finditer(text):
        if int(match.group(1)) in candidate_ids:
            return int(match.group(1))
    return candidate_ids[0]


def parse_decision(text: str, candidate_ids: list[int]) -> Action:
    """The last ASK:/ANSWER: line of `text`. Text with neither is read as an answer."""
    decisions = _DECISION_PATTERN.findall(text)
    if not decisions:
        return Answer(parse_referent(text, candidate_ids))
    kind, payload = decisions[-1]
    if kind.upper() == "ANSWER":
        return Answer(parse_referent(payload.strip(), candidate_ids))
    return Ask(*resolve_ask(payload.strip()))

"""The attributes a clarifying question can be about, and their template questions."""

from __future__ import annotations

from ..data.simmc import NON_VISUAL_PERMITTED, VISUAL_FORBIDDEN

# Intrinsic attributes include the visual ones: the environment can answer
# them, but the text-only task refuses them (see MCREnv).
INTRINSIC_ATTRS = tuple(sorted(NON_VISUAL_PERMITTED | VISUAL_FORBIDDEN))
SPATIAL_BIN_ATTRS = ("left_right", "up_down")
SPATIAL_GRAPH_ATTRS = ("left", "right", "up", "down")
ASK_ATTRIBUTES = INTRINSIC_ATTRS + SPATIAL_BIN_ATTRS + SPATIAL_GRAPH_ATTRS

# What a text-only policy is offered: the permitted attributes and the two spatial groups.
TEXT_ASKABLE_ATTRS = tuple(sorted(NON_VISUAL_PERMITTED)) + SPATIAL_BIN_ATTRS + SPATIAL_GRAPH_ATTRS

# An ask the harness couldn't map to an attribute. It spends a turn and reveals nothing.
UNRESOLVED_ASK_ATTRIBUTE = "__unresolved__"

QUESTION_TEMPLATES = {
    "brand": "What brand is it?",
    "price": "What's the price?",
    "customerReview": "How is it rated by customers?",
    "availableSizes": "What sizes are available?",
    "size": "What size is it?",
    "color": "What color is it?",
    "pattern": "What pattern does it have?",
    "type": "What type of item is it?",
    "assetType": "What does it look like?",
    "sleeveLength": "What sleeve length does it have?",
    "left_right": "Is it on the left or the right?",
    "up_down": "Is it higher up or lower down?",
    "left": "Is it to the left of another one?",
    "right": "Is it to the right of another one?",
    "up": "Is it above another one?",
    "down": "Is it below another one?",
}


def question_for_attribute(attribute: str) -> str:
    return QUESTION_TEMPLATES.get(attribute, "Could you clarify which one you mean?")

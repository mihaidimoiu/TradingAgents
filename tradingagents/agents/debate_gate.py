"""Optional early stop for the bull/bear debate, judged by TypeSafe's Jev (#1404).

The debate runs a fixed number of rounds. With ``jev_debate_gate`` on and
``TYPESAFE_API_KEY`` set, after each complete round below the cap Jev answers
one typed question about the transcript so far: is the case already
decision-ready for the Research Manager? A confident yes ends the debate early,
saving a round of two LLM calls. Anything else (no key, a failed request, low
confidence) leaves the fixed-round flow untouched, as ``post_screen`` degrades
for the Sentiment Analyst.

What leaves the machine, per judgement: the number of rounds completed and the
last ``MAX_CHARS_PER_SIDE`` characters of each side's history.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from tradingagents.agents.post_screen import TypeSafeError, system_one

logger = logging.getLogger(__name__)

# Jev must be at least this confident the case is decision-ready.
_CONVERGED_ABOVE = 0.65
# Keeps the transcript inside System One's state budget and bounds what is sent.
MAX_CHARS_PER_SIDE = 4000

_QUESTIONS = {
    "decision_ready": {
        "type": "noul",
        "instructions": (
            "After this many rounds of debate, is the case between the bull "
            "and bear arguments already decision-ready for a research manager: "
            "each side's core claim, the evidence behind it, and the other "
            "side's rebuttal are all on the table?"
        ),
        "criteria": {
            "true": "Both positions and their key rebuttals are stated; another round would "
                    "repeat or refine wording, not add substance.",
            "false": "A core claim or its rebuttal is still missing, vague, or unaddressed, "
                     "or the sides are talking past each other.",
        },
    }
}


@dataclass(frozen=True)
class DebateGateVerdict:
    """One judgement: after ``round`` complete rounds, whether the debate ``stop``\\s there."""

    round: int
    score: float | None
    stop: bool
    failure: str = ""


DebateGate = Callable[[dict], bool | None]


def jev_debate_gate(
    config: Mapping, on_verdict: Callable[[DebateGateVerdict], None] | None = None,
) -> DebateGate | None:
    """A gate for ``ConditionalLogic``; None unless ``config`` enables it and a TypeSafe key is set.

    The gate takes the investment-debate state and returns True to end the
    debate now, False or None to keep the fixed-round flow; ``on_verdict``
    hears every judgement, a failed one with no score.
    """
    if not config.get("jev_debate_gate") or not os.environ.get("TYPESAFE_API_KEY"):
        return None

    def gate(debate_state: dict) -> bool | None:
        rounds = int(debate_state.get("count", 0)) // 2
        state = {
            "rounds_completed": rounds,
            "bull_argument": str(debate_state.get("bull_history") or "")[-MAX_CHARS_PER_SIDE:],
            "bear_argument": str(debate_state.get("bear_history") or "")[-MAX_CHARS_PER_SIDE:],
        }
        try:
            score = float(system_one(state, _QUESTIONS)["decision_ready"]["noul"])
        except (TypeSafeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Jev debate gate unavailable (%s); debate continues", exc)
            if on_verdict is not None:
                on_verdict(DebateGateVerdict(round=rounds, score=None, stop=False, failure=str(exc)))
            return None
        stop = score >= _CONVERGED_ABOVE
        if on_verdict is not None:
            on_verdict(DebateGateVerdict(round=rounds, score=score, stop=stop))
        return stop

    return gate

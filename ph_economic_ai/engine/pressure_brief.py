"""The Pressure Monitor's output contract — the *hero* of a run.

A `PressureBrief` is a present-state read (a nowcast), not a forecast: per sector,
which way pressure is leaning right now, a magnitude, how much the agents agree,
what drove it, and which frozen sources informed it. The forecast (M4) consumes
this brief as its prior; here it stands on its own as the thing the app leads with.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class SectorReading:
    sector: str                 # 'gas' | 'food' | 'electricity'
    direction: str              # 'rising' | 'easing' | 'flat' | 'unknown'
    estimate: Optional[float]   # present-state signed change, in `unit`
    unit: str                   # '₱/L' | '%' | '₱/kWh'
    confidence: int             # agent agreement %, NOT a probability
    # Share of agents agreeing on the DIRECTION (sign) of the move. Coarser than
    # `confidence`, which needs magnitudes within a tight band: live runs showed
    # eight agents unanimous that prices rise while split between +1.0 and +2.5,
    # reading 100% here and 50% there. Both are true; they answer different
    # questions, and the direction one is the question a household actually asks.
    direction_agreement: int = 0
    drivers: list[str] = field(default_factory=list)   # salient present-tense points
    sources: list[str] = field(default_factory=list)   # snapshot/RAG sources used
    # The estimates the percentage was computed FROM. Carried because a
    # percentage cannot distinguish a room of twenty from two survivors, nor
    # agents who independently agreed from agents who copied: 32 agents produced
    # TWO distinct estimates spanning 0.26 PHP/L and scored 100 percent. The
    # values are checkable against the agent cards on the same screen; the
    # percentage is not.
    #
    # `kw_only` because this class is built positionally in places and a field
    # added in the middle silently reassigns every argument after it. That is not
    # hypothetical here: adding `direction_agreement` left three call sites
    # passing `drivers` into it, and nothing failed because nothing did
    # arithmetic on the value until this field arrived. Same shape as the
    # `AgentResponse` rebuild that dropped `retrieval` (`RSK-019`). Keyword-only
    # makes the position irrelevant, so the next field cannot repeat it.
    estimates: list[float] = field(default_factory=list, kw_only=True)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PressureBrief:
    as_of: str                  # ISO date the read is "as of"
    window: str                 # 'today' | 'this_week' | 'this_month'
    readings: list[SectorReading]
    narrative: str = ''         # 2-3 sentence present-tense summary

    def to_dict(self) -> dict:
        return {
            'as_of': self.as_of,
            'window': self.window,
            'narrative': self.narrative,
            'readings': [r.to_dict() for r in self.readings],
        }

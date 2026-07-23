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
    drivers: list[str] = field(default_factory=list)   # salient present-tense points
    sources: list[str] = field(default_factory=list)   # snapshot/RAG sources used

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

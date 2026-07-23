"""Stage 2 — the Outlook: a bounded, monthly forecast seeded by the Pressure Brief.

This is the *secondary* output. Its whole job is to stay humble where the audit
says it must. Per sector it:

1. reads the benchmark's own verdict (the *gate*) — efficient / mechanical /
   own-dynamics — from the frozen report;
2. seeds the tournament with the brief's present read as its prior;
3. **bounds** the tournament's number to the anchor and attaches an interval; and
4. frames it: on an efficient sector the headline is "naive + interval, no
   exploitable edge" and the tournament number is shown only as a bounded
   exploratory scenario — never a confident prediction.

The tournament itself is injectable, so this honest logic is unit-testable without
running the 39-call swarm. `make_swarm_tournament(rag)` wires the real thing.
Horizon is always **monthly** — no weekly/daily forecast number is ever produced.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ph_economic_ai.engine import anchoring
from ph_economic_ai.engine.pressure_brief import PressureBrief

_REPORT = (Path(__file__).resolve().parents[1] / 'benchmark' / 'artifacts'
           / 'accuracy_report.json')

# Fallback ±band per sector (monthly), used when the report carries no conformal
# width for the sector. Gas prefers the frozen fuel conformal 90% half-width.
_DEFAULT_BAND = {'gas': 3.0, 'food': 1.0, 'electricity': 1.0}

_NOTE = {
    'efficient': ('Benchmark: no exploitable edge at 1 month. The point is the naive '
                  'path and the interval is the honest deliverable; the tournament '
                  'number is a bounded exploratory scenario, not a prediction.'),
    'mechanical': ('Validated within-month channel (a formulaic generation-charge '
                   'pass-through). Forecast is bounded by the physical anchor.'),
    'own-dynamics': ('Predictable via the series\' own short-run dynamics, not a driver '
                     'signal. Forecast bounded by the anchor.'),
}


@dataclass
class ForecastResult:
    point: Optional[float]
    agreement: int = 0          # tournament agreement %, NOT a probability
    raw: Optional[float] = None  # pre-bounding number, for transparency


# A tournament maps (sector, prior_estimate, scenario) -> ForecastResult.
Tournament = Callable[[str, Optional[float], dict], ForecastResult]


@dataclass
class SectorOutlook:
    sector: str
    basis: str                  # 'efficient' | 'mechanical' | 'own-dynamics'
    point: Optional[float]      # bounded next-month change, in `unit`
    interval: Optional[list]    # [low, high]
    unit: str
    agreement: int              # tournament agreement %, labeled not-a-probability
    note: str
    tournament_estimate: Optional[float] = None  # raw seeded-tournament number (bounded view)
    horizon: str = 'next month'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Outlook:
    as_of: str
    sectors: list[SectorOutlook] = field(default_factory=list)
    horizon: str = 'next month'

    def to_dict(self) -> dict:
        return {'as_of': self.as_of, 'horizon': self.horizon,
                'sectors': [s.to_dict() for s in self.sectors]}


def sector_basis(sector: str, report: dict) -> str:
    """The verdict gate: what does the frozen benchmark say about this sector?
    Absent/ambiguous report -> 'efficient' (the conservative default)."""
    if sector == 'gas':
        audit = {a.get('target'): a for a in report.get('audit', []) if isinstance(a, dict)}
        v = audit.get('fuel', {}).get('verdict')
        # Conservative: efficient unless the report EXPLICITLY says fuel is forecastable.
        return 'own-dynamics' if v not in (None, 'efficient') else 'efficient'
    if sector == 'electricity':
        return 'mechanical' if report.get('electricity_nowcast', {}).get('driver_edge_robust') \
            else 'efficient'
    if sector == 'food':
        v = ((report.get('food_nowcast', {}) or {}).get('mom', {}) or {}).get('verdict')
        return 'own-dynamics' if v == 'beats_best_naive' else 'efficient'
    return 'efficient'


def _band(sector: str, report: dict) -> float:
    if sector == 'gas':
        w = (report.get('conformal_widths', {}) or {}).get('0.9')
        if w:
            return float(w)
    return _DEFAULT_BAND.get(sector, 1.0)


def forecast_outlook(brief: PressureBrief, report: Optional[dict] = None,
                     tournament: Optional[Tournament] = None) -> Outlook:
    """Bounded monthly Outlook from the brief. `tournament` runs the seeded forecast;
    if None, the forecast is the naive persistence of the present read."""
    report = report or {}
    sectors = []
    for r in brief.readings:
        basis = sector_basis(r.sector, report)
        prior = r.estimate
        scenario = {'as_of': brief.as_of, 'window': brief.window, 'sector': r.sector}

        if tournament is not None:
            fr = tournament(r.sector, prior, scenario)
        else:
            fr = ForecastResult(point=prior, agreement=r.confidence, raw=prior)

        raw = fr.point
        point = raw
        # Bound the forecast to the present read (the anchor): next month should
        # not diverge wildly from what is happening now unless the tournament
        # justifies it — reconcile clamps a drifting number back toward the anchor.
        if raw is not None and prior is not None:
            try:
                point = anchoring.reconcile_estimate(raw, prior).value
            except Exception:
                point = raw

        band = _band(r.sector, report)
        interval = ([round(point - band, 2), round(point + band, 2)]
                    if point is not None else None)

        sectors.append(SectorOutlook(
            sector=r.sector, basis=basis,
            point=(round(point, 2) if point is not None else None),
            interval=interval, unit=r.unit, agreement=int(fr.agreement),
            note=_NOTE.get(basis, _NOTE['efficient']),
            tournament_estimate=(round(raw, 2) if raw is not None else None)))
    return Outlook(as_of=brief.as_of, sectors=sectors)


def make_swarm_tournament(rag) -> Tournament:
    """Wire the real fuel tournament (swarm), seeded by the present read. Only gas
    has a tournament; food/electricity fall back to naive persistence. Imported
    lazily so a pure Outlook test never pulls in the swarm/PyQt stack."""
    def _tournament(sector: str, prior: Optional[float], scenario: dict) -> ForecastResult:
        if sector != 'gas':
            return ForecastResult(point=prior, agreement=0, raw=prior)
        from ph_economic_ai.engine.swarm import SwarmOrchestrator
        ml = f"Present-read baseline: {prior:+.2f} PHP/L" if prior is not None else ''
        try:
            mv = SwarmOrchestrator(rag, dict(scenario), ml_baseline=ml).run()
        except Exception:
            return ForecastResult(point=prior, agreement=0, raw=prior)
        return ForecastResult(point=mv.final_estimate,
                              agreement=int(mv.confidence_pct or 0),
                              raw=mv.final_estimate)
    return _tournament


def load_report(report_path: Path = _REPORT) -> dict:
    try:
        return json.loads(Path(report_path).read_text(encoding='utf-8'))
    except Exception:
        return {}


def run_outlook(brief: PressureBrief, rag=None, report_path: Path = _REPORT,
                run_tournament: bool = True) -> Outlook:
    """Stage-2 entry point: bounded monthly Outlook. With `run_tournament`, seeds
    and runs the real swarm for gas; otherwise naive persistence throughout."""
    report = load_report(report_path)
    tournament = make_swarm_tournament(rag) if (run_tournament and rag is not None) else None
    return forecast_outlook(brief, report, tournament)

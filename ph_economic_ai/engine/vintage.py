"""What makes two runs THE SAME run.

The app had no answer to that question, and it showed. Eight runs on
2026-07-27 returned -0.52, -0.10, -1.03, -2.94, -0.97, -2.14, -1.12 and -0.60
₱/L. Nothing about the Philippine fuel market changed between them: the DOE
adjusts on Tuesdays, the sliders were untouched, and three of those runs had
byte-identical stored scenarios. A user who runs the app twice in an afternoon
is entitled to the same answer, and got eight.

ADR-002 already tried to fix this with derived seeds and mostly failed, for two
reasons this module addresses:

1. **The seed key was live.** `_scenario_seed` hashed the whole scenario dict,
   including `oil_pct` and `usd_pct` recomputed from Yahoo ticks on every run. A
   0.01 percent move in Brent produced a different seed for every agent in the
   swarm. The seed was stable with respect to an input that never held still.

2. **The prompt carried a clock.** `LiveDataBrief.fetched_at` is minute
   resolution and was printed into the header of the DATA BRIEF block that
   prefixes every agent, judge and master prompt. A seed only reproduces a call
   when the prompt is identical, so two runs a minute apart could not agree even
   in principle. This is why runs 21, 22 and 23 disagree by ₱1.54 despite
   identical scenarios and identical seeds.

The fix is to define a VINTAGE: the window during which the inputs are
considered unchanged, and inside which a run is expected to reproduce. Both the
seed and the recall lookup key on the vintage rather than on the wall clock.

Everything here is pure and deterministic. No network, no clock unless one is
passed in, so it is fully testable.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Any, Optional

from ph_economic_ai.engine import price_calendar

#: How far a live input may move and still be "unchanged", per key.
#:
#: These are TOLERANCES, compared as |a - b| <= tol, not a rounding grid. That
#: distinction is the whole design and it was arrived at the hard way: an earlier
#: version quantised each value onto a fixed grid and hashed the result, which
#: fails on boundaries. Runs 27 and 28 recorded oil_pct of -3.38 and -3.31, seven
#: hundredths of a percentage point apart and unambiguously the same market read,
#: and a 0.25 grid put them in different buckets because they straddle -3.375. A
#: tolerance has no boundaries to straddle.
#:
#: Each threshold sits below the resolution the answer can express, so a
#: difference this small cannot change the verdict, only cause it to be
#: recomputed for nothing:
#:
#: `oil_pct` / `usd_pct` at 0.25 percentage points — the mechanical anchor moves
#: about 8 centavos per 0.25pp of oil (crude cost per litre is roughly ₱35.7, so
#: 0.25 percent of it is ₱0.089, times VAT and the 0.79 calibration). The models
#: quote on a rough half-peso grid, so 8 centavos is invisible to them.
#:
#: `demand_index` at 1.0 — derived from Manila's forecast max temperature, which
#: Open-Meteo reissues hourly and moves by tenths of a degree.
#:
#: `current_price` at 0.05 — the DOE publishes to the centavo, but the value is a
#: median over whichever brands happened to render on fuelprice.ph, so the last
#: centavo is scrape noise rather than signal.
_TOLERANCE: dict[str, float] = {
    'oil_pct': 0.25,
    'usd_pct': 0.25,
    'demand_index': 1.0,
    'current_price': 0.05,
    'bsp_rate': 0.05,
}

#: Market fields from the data brief that are compared, and their tolerances.
#: Deliberately NOT compared: `fetched_at` (a clock), `psei` (usually None and
#: never used in a prompt), the four news feeds (Google News reorders them
#: without new articles, so they would defeat every recall), and `weather_manila`
#: beyond its contribution through `demand_index`.
_BRIEF_TOLERANCE: dict[str, float] = {
    'brent': 0.50,
    'wti': 0.50,
    'usd_php': 0.05,
}


def _f(value: Any) -> Optional[float]:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def fuel_cycle_start(now: Optional[dt.datetime] = None) -> dt.datetime:
    """The Tuesday 06:00 PHT adjustment that opened the CURRENT fuel cycle.

    `price_calendar.next_fuel_adjustment` looks forward, which is what the report
    needs to answer "when". A vintage needs the opposite: the boundary already
    crossed. Retail fuel prices in the Philippines are a step function with a
    weekly step, so this is the natural window inside which the inputs to a fuel
    forecast genuinely have not changed.
    """
    return price_calendar.next_fuel_adjustment(now) - dt.timedelta(days=7)


def vintage(now: Optional[dt.datetime] = None) -> dict[str, str]:
    """The window a run belongs to.

    Two parts, because they answer different questions and both matter:

    - `fuel_cycle` is the pricing week. A run from last Tuesday is describing a
      different price level and must never be recalled for this one.
    - `day` is the calendar date in Philippine time. Within a cycle the news, the
      social snapshot and the weather still turn over daily, and the user's own
      expectation is stated in days: run it twice today, get today's answer.
    """
    current = price_calendar._now(now)
    return {
        'fuel_cycle': fuel_cycle_start(now).date().isoformat(),
        'day': current.date().isoformat(),
    }


def input_snapshot(scenario: Optional[dict], brief: Optional[Any] = None) -> dict:
    """The inputs a run was answered on, recorded so a later run can compare.

    The scenario alone is not enough. `oil_pct` and `usd_pct` are derived from
    the brief's five-day history, but `brent`, `wti` and `usd_php` reach the
    agents directly through the DATA BRIEF block, so two runs can share a
    scenario and still have shown the agents different market numbers.

    `brief_present` is part of the record because the run path has a nine second
    timeout that hands the swarm `data_brief=None` and carries on. A run that
    saw no brief and a run that saw one are not the same run, whatever their
    scenarios say.
    """
    snap: dict[str, Any] = {'brief_present': brief is not None}
    for key in _TOLERANCE:
        snap[key] = _f((scenario or {}).get(key))
    for key in _BRIEF_TOLERANCE:
        snap[key] = _f(getattr(brief, key, None)) if brief is not None else None
    return snap


def inputs_unchanged(before: Optional[dict], now_: Optional[dict]) -> bool:
    """Have the inputs moved enough to be worth re-answering?

    Compared field by field as |a - b| <= tolerance. A missing value on either
    side is treated as "changed": absence is not evidence that nothing moved,
    and the cost of being wrong in this direction is one honest re-run.
    """
    if not isinstance(before, dict) or not isinstance(now_, dict):
        return False
    if bool(before.get('brief_present')) != bool(now_.get('brief_present')):
        return False
    for key, tol in {**_TOLERANCE, **_BRIEF_TOLERANCE}.items():
        a, b = _f(before.get(key)), _f(now_.get(key))
        if a is None and b is None:
            # Both absent is genuinely unchanged: neither run had the field.
            continue
        if a is None or b is None or abs(a - b) > tol:
            return False
    return True


def describe_drift(before: Optional[dict], now_: Optional[dict]) -> str:
    """The largest tolerated movement between two snapshots, in words.

    `inputs_unchanged` answers "close enough to reuse the answer?", which is the
    right question and is NOT the same as "nothing moved". The recall note used to
    tell the reader "the inputs have not moved since", and that is false whenever
    any field moved inside its tolerance.

    Demonstrated: Brent 74.20 to 74.69, WTI 70.10 to 70.59, USD/PHP 58.4000 to
    58.4400 all pass, while three lines of the DATA BRIEF that prefixes every
    agent, judge and master prompt are different. Ollama reproduces a call exactly
    given the same seed, verified directly, so a fresh run on those inputs would
    have returned a different number. Recall was right to reuse the answer and
    wrong about why.

    Empty string when nothing moved at all, so the caller can still say so
    truthfully on the runs where it is true.
    """
    if not isinstance(before, dict) or not isinstance(now_, dict):
        return ''
    worst_key, worst_delta, worst_tol = '', 0.0, 0.0
    for key, tol in {**_TOLERANCE, **_BRIEF_TOLERANCE}.items():
        a, b = _f(before.get(key)), _f(now_.get(key))
        if a is None or b is None:
            continue
        delta = abs(a - b)
        # Ranked by share of tolerance used, not by raw size: 0.04 of a 0.05
        # band is a bigger deal than 0.30 of a 0.50 one, and comparing pesos
        # against index points against percents would rank them by unit.
        if tol > 0 and delta / tol > (worst_delta / worst_tol if worst_tol else 0):
            worst_key, worst_delta, worst_tol = key, delta, tol
    if not worst_key or worst_delta <= 0:
        return ''
    return f'{worst_key} moved {worst_delta:.4g} of a tolerated {worst_tol:.4g}'


def vintage_key(model_id: str = '', now: Optional[dt.datetime] = None) -> str:
    """The bucket a run belongs to: this day, this pricing week, these models.

    Deliberately coarse. It narrows the recall search to runs that could
    plausibly answer the same question; `inputs_unchanged` then decides whether
    one of them actually does. Splitting it this way is what avoids the boundary
    problem — a hash cannot express "close enough", and a tolerance cannot be
    looked up in an index, so each does the half it is good at.

    `model_id` is in the key on purpose. The same question answered by a 3b model
    on Ollama and by a hosted model on Groq is not the same answer, and the
    provider is resolved from the environment, so it can change between two runs
    on one machine without anything in the app changing.
    """
    payload = {'vintage': vintage(now), 'model': model_id}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode('utf-8'), digest_size=8).hexdigest()


def describe_vintage(now: Optional[dt.datetime] = None) -> str:
    """One line for the report, naming the window rather than the minute.

    This is what replaces `fetched_at` inside the prompt. The agents need to know
    which pricing week and which day they are reasoning about; they have never
    needed to know the minute, and printing it is what made every prompt unique.
    """
    v = vintage(now)
    return f"{v['day']} (fuel pricing week of {v['fuel_cycle']})"

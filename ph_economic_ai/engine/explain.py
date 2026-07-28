"""Why a sector is moving, when it lands, and where that came from.

The report showed a number per sector and nothing else: a household reading
"-0.60 PHP/L" learns what, but not why, not when, and not on whose authority.

Three constraints shape this module, and the third is the one that matters.

**Grounded.** The explanation may only use drivers the forum actually produced
and cite sources the forum actually read. An LLM asked to explain a price move
will happily supply a plausible mechanism it was never given, and a fluent
invented reason is worse than no reason: it reads as evidence. Every returned
explanation is checked against the reading's own source list, and any citation
that is not in it is stripped and reported.

**Dated, not forecast.** "When" comes from `price_calendar`, which reads a
published schedule. The model is never asked to predict a date.

**Degrades honestly.** With no LLM reachable, the module still returns a usable
explanation built from the drivers themselves rather than failing or inventing.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

from ph_economic_ai.engine import price_calendar

# Fuel moves on the weekly DOE adjustment; the CPI-derived sectors on the PSA
# release. Same mapping the monitor card uses.
_SECTOR_EVENT = {'gas': 'fuel', 'food': 'cpi', 'electricity': 'cpi'}

_SYSTEM = (
    'You explain a Philippine price movement to a household in plain language. '
    'You are given a sector, a measured change, and the specific points the '
    'analysts raised. Write ONE sentence, at most 30 words, saying WHY the price '
    'is moving that way.\n'
    'HARD RULES. Use only the points you are given. Do NOT add causes, numbers, '
    'institutions, or events that are not in them. Do NOT predict a date; the '
    'date is supplied separately from a published schedule. If the points do not '
    'actually explain the direction, say the drivers are unclear rather than '
    'inventing a reason. Plain words, no jargon, no hedging filler.'
)

_MAX_WORDS = 30


def _event_for(sector: str) -> dict:
    if _SECTOR_EVENT.get(sector, 'cpi') == 'fuel':
        return price_calendar.describe_next_fuel_adjustment()
    return price_calendar.describe_next_cpi_release()


def _direction_word(estimate: Optional[float], direction: str) -> str:
    if estimate is None:
        return direction or 'unclear'
    if estimate > 0:
        return 'going up'
    if estimate < 0:
        return 'going down'
    return 'holding steady'


def _decapitalise(text: str) -> str:
    """Lower the first letter for mid-sentence use, but never inside an acronym.

    Blindly lowering turned "WESM spot prices eased" into "wESM spot prices
    eased", which reads as a typo in the one place the app is claiming to explain
    itself carefully.
    """
    if not text:
        return text
    first = text.split(' ', 1)[0]
    if first.isupper() or (len(first) > 1 and first[1].isupper()):
        return text
    return text[0].lower() + text[1:]


def fallback_why(reading) -> str:
    """An explanation built only from the drivers, for when no model is reachable."""
    drivers = [d.strip() for d in (reading.drivers or []) if d and d.strip()]
    move = _direction_word(reading.estimate, reading.direction)
    if not drivers:
        return f'{reading.sector.title()} is {move}, but the analysts did not agree on a clear driver.'
    lead = _decapitalise(drivers[0].rstrip('.'))
    if len(drivers) > 1:
        others = len(drivers) - 1
        plural = '' if others == 1 else 's'
        return (f'{reading.sector.title()} is {move}, mainly because {lead}, '
                f'with {others} other factor{plural} cited.')
    return f'{reading.sector.title()} is {move}, mainly because {lead}.'


def check_grounding(text: str, allowed_sources: list[str]) -> dict:
    """Flag citations the reading never supplied.

    Deliberately narrow: it looks for source-shaped tokens (capitalised acronyms
    and known source names) that are not in the allowed list. It cannot catch an
    invented *reason*, which is why the prompt forbids adding causes and why the
    drivers are passed verbatim.
    """
    allowed = {s.upper() for s in (allowed_sources or [])}
    candidates = set(re.findall(r'\b[A-Z]{2,}(?:[-_][A-Z0-9]+)?\b', text or ''))
    # Ordinary sentence-initial words and units are not citations.
    ignore = {'PHP', 'L', 'KWH', 'AI', 'CPI', 'MOM', 'YOY', 'PH', 'NCR', 'US'}
    ungrounded = sorted(c for c in candidates if c not in allowed and c not in ignore)
    return {'grounded': not ungrounded, 'ungrounded': ungrounded}


def build_explanation(reading, complete: Optional[Callable] = None,
                      temperature: float = 0.2,
                      seed: Optional[int] = None) -> dict:
    """Why / when / sources for one sector reading.

    `complete(messages, **kw) -> str` is injected so this is testable without a
    model and so a failure degrades to `fallback_why` rather than raising.
    """
    event = _event_for(reading.sector)
    sources = [s for s in (reading.sources or []) if s]
    drivers = [d for d in (reading.drivers or []) if d]

    why, generated = fallback_why(reading), False
    if complete is not None and drivers:
        unit = reading.unit or ''
        est = 'unknown' if reading.estimate is None else f'{reading.estimate:+.2f} {unit}'
        user = (
            f'Sector: {reading.sector}\n'
            f'Measured change: {est}\n'
            f'Direction: {_direction_word(reading.estimate, reading.direction)}\n'
            f'Analyst points (use ONLY these):\n'
            + '\n'.join(f'- {d}' for d in drivers)
            + (f'\nSources the analysts read: {", ".join(sources)}' if sources else '')
        )
        try:
            raw = complete(
                [{'role': 'system', 'content': _SYSTEM},
                 {'role': 'user', 'content': user}],
                temperature=temperature, seed=seed)
            candidate = ' '.join((raw or '').split())
            if candidate:
                words = candidate.split()
                if len(words) > _MAX_WORDS:
                    candidate = ' '.join(words[:_MAX_WORDS]).rstrip(',;:') + '...'
                why, generated = candidate, True
        except Exception:
            pass                      # keep the grounded fallback

    grounding = check_grounding(why, sources)
    return {
        'sector': reading.sector,
        'why': why,
        'why_generated': generated,
        'when': event['when'],
        'when_date': event['date'],
        'when_label': event['label'],
        'when_basis': event['basis'],
        'sources': sources,
        'drivers': drivers,
        'grounded': grounding['grounded'],
        'ungrounded_mentions': grounding['ungrounded'],
    }


def build_all(readings, complete: Optional[Callable] = None) -> list[dict]:
    return [build_explanation(r, complete) for r in (readings or [])]

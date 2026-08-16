"""Meralco's monthly generation charge, in PHP/kWh.

This is the level `engine.anchoring.electricity_passthrough_anchor` scales a
fuel/FX shock against, and it used to be the frozen constant 5.50. By 2026-07
the actual charge was 9.2504 -- 41% higher -- and because the constant
multiplies straight through, every electricity estimate the app produced was
biased low, silently. A number that has to be *remembered* is a number that goes
stale, so it is read from data here and the constant survives only as a
last-resort fallback.

**Why this is not a fetcher.** Meralco publishes one "Generation" PDF per month
in its rates archive and returns HTTP 403 to automated requests, so the series
cannot be refreshed over the network the way `psa_cpi` refreshes PSA. What is
automated instead is the *reading*: `parse_generation_charge_text` turns a
published PDF into a row, so adding a month is running
`tools/refresh_meralco.py` on a downloaded file rather than editing a constant
and hoping the next person notices.

The same series is the outcome side of grading electricity. The sector's
estimate is a PHP/kWh change in the generation charge, and
`load_generation_charge_mom` is exactly that quantity -- which is what
`engine/interval.py` says electricity needs before it can join `GRADED_SECTORS`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
GENERATION_CHARGE_CSV = HERE / 'data' / 'meralco_generation_charge.csv'

#: Used only when the series cannot be read at all. Set to the last verified
#: published level (2026-07) rather than a round number, so a fallback that
#: silently becomes load-bearing is still approximately right instead of 41%
#: wrong. Update it whenever the series is refreshed.
FALLBACK_PHP_KWH = 9.2504

_MONTHS = {m: i for i, m in enumerate(
    ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY',
     'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER'], start=1)}


def load_generation_charge(csv_path: Path = GENERATION_CHARGE_CSV) -> pd.Series:
    """Monthly generation charge (PHP/kWh) indexed by 'YYYY-MM', sorted."""
    df = pd.read_csv(csv_path, dtype={'date': str})
    s = pd.Series(df['generation_charge_php_kwh'].astype(float).values,
                  index=df['date'].astype(str).values)
    return s[~s.index.duplicated(keep='last')].sort_index()


def load_generation_charge_mom(csv_path: Path = GENERATION_CHARGE_CSV) -> pd.Series:
    """Month-over-month change in PHP/kWh -- NOT a percentage.

    This is the unit the electricity estimate is expressed in, which is the
    whole point: grading it against the PSA electricity CPI would compare a
    PHP/kWh forecast to a percentage move of a different basket.
    """
    return load_generation_charge(csv_path).diff().dropna()


def latest_generation_charge(csv_path: Path = GENERATION_CHARGE_CSV,
                             default: float = FALLBACK_PHP_KWH) -> float:
    """Newest published level, or `default` if the series cannot be read.

    Never raises. The anchor is a magnitude guard on a live screen and may not
    be taken down by a missing or malformed data file.
    """
    try:
        s = load_generation_charge(csv_path)
        if len(s):
            return float(s.iloc[-1])
    except Exception:
        pass
    return float(default)


def parse_generation_charge_text(text: str) -> float:
    """The billed generation charge from a rate-table PDF's extracted text.

    Takes the figure AFTER Other Generation Adjustments, not the TOTAL line
    above it. In July 2026 those were 9.2504 and 9.2656 -- close enough that
    reading the wrong one would look entirely plausible and quietly bias the
    anchor. The OGA block also varies in length between months (August 2026
    added a GOUR Recovery line), so the headline is found by position relative
    to the footnote rather than by counting rows.
    """
    marker = text.find('*Includes adjustment')
    body = text[:marker] if marker != -1 else text
    numbers = [ln.strip() for ln in body.splitlines()
               if re.fullmatch(r'\(?\d+\.\d{4}\)?', ln.strip())]
    if not numbers:
        raise ValueError(
            'no adjusted generation-charge headline found; expected a bare '
            '4-decimal figure after the Other Generation Adjustments block')
    return float(numbers[-1].replace('(', '-').replace(')', ''))


def parse_generation_charge_month(text: str) -> str:
    """The 'YYYY-MM' this table describes, from its own heading."""
    m = re.search(r'([A-Z]+)\s+(\d{4})\s+GENERATION CHARGE', text.upper())
    if not m or m.group(1) not in _MONTHS:
        raise ValueError('no "<MONTH> <YEAR> GENERATION CHARGE" heading found')
    return f'{int(m.group(2)):04d}-{_MONTHS[m.group(1)]:02d}'


def parse_generation_charge_pdf(pdf_path) -> tuple[str, float]:
    """(month, PHP/kWh) from a downloaded Meralco Generation PDF."""
    import pypdf
    text = '\n'.join((p.extract_text() or '')
                     for p in pypdf.PdfReader(str(pdf_path)).pages)
    return parse_generation_charge_month(text), parse_generation_charge_text(text)

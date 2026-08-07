"""Freeze a FRED-MD snapshot for the vulnerability survey.

FRED-MD (McCracken & Ng, 2016) is the standard monthly US macro panel used to
benchmark nowcasting methods — including by Beck, Dovern & Vogl (2025), whose
"mind the naive forecast" result this thesis complements. The survey in
`benchmark/vulnerability_survey.py` measures how many of its series sit in the
regime where omitting the mean from a naive pool produces spurious significance.

Like every other live pull in this project, the fetch lives in `tools/` and
writes a frozen CSV; the benchmark reads only the frozen file, so the survey
reproduces offline and cannot drift under a reviewer.

    python -m ph_economic_ai.tools.refresh_fredmd
"""
from __future__ import annotations

import sys

import pandas as pd

from ph_economic_ai.benchmark.paths import data
from ph_economic_ai.benchmark.provenance import write_record

URL = ('https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/'
       'fred-md/monthly/current.csv')
OUT = data('fredmd_snapshot.csv')


def refresh(url: str = URL, out=OUT) -> int:
    """Download FRED-MD and freeze it. Returns rows written (0 = skipped).

    Nothing partial is ever written: the frame is validated before it replaces
    the committed snapshot.
    """
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f'fredmd: fetch failed ({type(e).__name__}: {e}) — nothing written')
        return 0
    if df.empty or df.columns[0] != 'sasdate' or str(df.iloc[0, 0]).strip() != 'Transform:':
        print('fredmd: unexpected layout (no Transform: row) — nothing written')
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator='\n')
    write_record(out, source='FRED-MD monthly database (McCracken and Ng 2016)',
                 params={'vintage': 'current'},
                 transformations=['keep the tcode header row',
                                  'no transform applied at snapshot time; the '
                                  'recommended tcode transform is applied by '
                                  'vulnerability_survey at analysis time'],
                 units='mixed, per series; see the tcode row',
                 notes='Frozen snapshot backing the exposure census.')
    print(f'fredmd: wrote {len(df) - 1} months x {df.shape[1] - 1} series -> {out}')
    return len(df) - 1


if __name__ == '__main__':
    sys.exit(0 if refresh() else 1)

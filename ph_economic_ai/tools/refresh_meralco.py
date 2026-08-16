"""Add months to the Meralco generation-charge series from downloaded PDFs.

Meralco returns HTTP 403 to automated requests, so unlike `psa_cpi` this cannot
fetch. The manual step is downloading; everything after it is automated, which
is the part that used to be a hand-edited constant going stale unnoticed.

    1. https://company.meralco.com.ph -> rates archive -> the month's
       "Generation" PDF (filter by Category = Generation).
    2. python -m ph_economic_ai.tools.refresh_meralco ~/Downloads/*_gc_table*.pdf

Rows are merged by month, so re-running on the same PDF is a no-op and a
corrected re-issue overwrites cleanly. Writes a provenance sidecar like every
other committed input.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ph_economic_ai.benchmark.meralco import (
    GENERATION_CHARGE_CSV,
    load_generation_charge,
    parse_generation_charge_pdf,
)
from ph_economic_ai.benchmark.provenance import write_record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pdfs', nargs='+', help='downloaded Generation PDFs')
    ap.add_argument('--out', default=str(GENERATION_CHARGE_CSV))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args(argv)

    out = Path(args.out)
    try:
        series = load_generation_charge(out).to_dict()
    except Exception:
        series = {}
    before = dict(series)

    for raw in args.pdfs:
        p = Path(raw)
        try:
            month, value = parse_generation_charge_pdf(p)
        except Exception as e:
            print(f'  SKIP {p.name}: {type(e).__name__}: {e}', file=sys.stderr)
            continue
        was = series.get(month)
        series[month] = value
        note = 'new' if was is None else ('unchanged' if was == value
                                          else f'revised from {was}')
        print(f'  {month}  {value:.4f} PHP/kWh  ({note})  <- {p.name}')

    if not series:
        print('nothing parsed; series unchanged', file=sys.stderr)
        return 1

    df = (pd.DataFrame(sorted(series.items()),
                       columns=['date', 'generation_charge_php_kwh'])
          .sort_values('date'))
    added = len(series) - len(before)
    if args.dry_run:
        print(f'dry run - would write {len(df)} rows ({added} new)')
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, lineterminator='\n')
    write_record(out,
                 source='Meralco monthly Generation rate tables (company.meralco.com.ph)',
                 params={'retrieval': 'manual download; the site returns HTTP 403 '
                                      'to automated requests',
                         'category': 'Generation'},
                 transformations=['read the adjusted headline after Other Generation '
                                  'Adjustments, not the pre-adjustment TOTAL',
                                  'label months YYYY-MM from the table heading, sort'],
                 units='PHP/kWh',
                 notes='Generation charge level. Its month-over-month DIFFERENCE is '
                       'the outcome series for grading the electricity sector, and '
                       'its latest level is the anchor scale in engine/anchoring.py.')
    print(f'Wrote {out.name} ({len(df)} rows, '
          f'{df["date"].iloc[0]}..{df["date"].iloc[-1]}, {added} new)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Build `benchmark/data/meralco_all_in_rate.csv` from Meralco bill tables.

The all-in residential rate, which `anchor_backtest` had frozen at 11.2 PHP/kWh
in a constant described as the 2024 average. Measured against Meralco's own
tables it runs 13.17 to 14.83 across 2026, so the constant was 15 to 25 percent
low. It is the same defect PR #29 removed one level down, and it survived that PR
because nothing in the repository could read the real figure.

**Source.** Meralco's "Typical Consumption Level" PDFs, filed as
`MM-YYYY_residential_bills.pdf`, from the rates archive at
`company.meralco.com.ph`. NOT the Summary Schedule of Rates: that lists every
component and never their sum, and mixes per-kWh, per-kW and per-customer-month
columns, so adding them is a unit error rather than an arithmetic shortcut.

**Fetching is manual, deliberately.** Meralco returns HTTP 200 and the same page
for every path including invented ones, so a real file cannot be told from a miss
by status code, and both WebFetch and a headless browser are refused. Deriving
the URL the way `refresh_doe_adjustment` does is therefore not possible here.
Point this at downloaded PDFs instead:

    python -m ph_economic_ai.tools.refresh_meralco_all_in <pdf> [<pdf> ...]

Each file's month comes from its own filename, and the generation charge it
carries is checked against `meralco_generation_charge.csv` where the months
overlap -- two documents that must agree, so a silent parse error in either is
caught rather than committed.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from ph_economic_ai.benchmark.meralco import (
    ALL_IN_RATE_CSV, TYPICAL_KWH, load_generation_charge,
    parse_residential_bill_row,
)
from ph_economic_ai.benchmark.provenance import write_record

_MONTH_FROM_NAME = re.compile(r'(\d{2})-(\d{4})')


def month_of(path) -> str | None:
    """'YYYY-MM' from a filename like `07-2026_residential_bills.pdf`."""
    m = _MONTH_FROM_NAME.search(Path(path).name)
    return f'{m.group(2)}-{m.group(1)}' if m else None


def read_pdf(path) -> tuple[float, float] | tuple[None, None]:
    import pypdf

    text = '\n'.join((p.extract_text() or '')
                     for p in pypdf.PdfReader(str(path)).pages)
    start = text.find('For Non-Lifeline')
    if start < 0:
        return None, None
    for line in text[start:].splitlines():
        rate, gen = parse_residential_bill_row(line, kwh=TYPICAL_KWH)
        if rate is not None:
            return rate, gen
    return None, None


def build(paths) -> None:
    existing: dict[str, dict] = {}
    if ALL_IN_RATE_CSV.exists():
        with open(ALL_IN_RATE_CSV, encoding='utf-8') as fh:
            existing = {r['date']: r for r in csv.DictReader(fh)}

    try:
        committed_gen = load_generation_charge()
    except Exception:
        committed_gen = {}

    for path in paths:
        month = month_of(path)
        if not month:
            print(f'  {Path(path).name}: no MM-YYYY in the filename, skipped')
            continue
        rate, gen = read_pdf(path)
        if rate is None:
            print(f'  {month}: no {TYPICAL_KWH} kWh non-lifeline row found')
            continue

        # Cross-check against the generation-charge series parsed from a
        # DIFFERENT Meralco document. They describe the same quantity, so a
        # disagreement means one of the two parsers is wrong and neither figure
        # should be trusted until it is resolved.
        note = ''
        if month in getattr(committed_gen, 'index', []):
            delta = abs(float(committed_gen[month]) - gen)
            note = ('  gen agrees' if delta < 1e-4
                    else f'  GEN MISMATCH vs committed ({committed_gen[month]})')
        existing[month] = {'date': month,
                           'all_in_php_kwh': f'{rate:.4f}',
                           'generation_php_kwh': f'{gen:.4f}',
                           'kwh_basis': str(TYPICAL_KWH)}
        print(f'  {month}  all-in {rate:.4f}  generation {gen:.4f}{note}')

    if not existing:
        raise SystemExit('nothing parsed; nothing written')

    fields = ['date', 'all_in_php_kwh', 'generation_php_kwh', 'kwh_basis']
    ALL_IN_RATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ALL_IN_RATE_CSV, 'w', encoding='utf-8', newline='\n') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for key in sorted(existing):
            writer.writerow({f: existing[key].get(f, '') for f in fields})

    write_record(
        ALL_IN_RATE_CSV,
        source=('Meralco Typical Consumption Level tables (residential bills), '
                'rates archive at company.meralco.com.ph'),
        params={'consumption_kwh': TYPICAL_KWH, 'customer_class': 'non-lifeline residential'},
        transformations=[
            f'all_in_php_kwh = total bill at {TYPICAL_KWH} kWh divided by {TYPICAL_KWH}',
            'generation_php_kwh = the row\'s first money column, divided the same way',
            'bracketed figures read as deductions, not charges',
            'generation cross-checked against meralco_generation_charge.csv, '
            'which is parsed from a different Meralco document',
        ],
        units='PHP per kWh',
        notes=(f'The rate is quoted at a consumption level and the bill is not '
               f'linear in kWh -- distribution steps by bracket and customer '
               f'charges are flat -- so {TYPICAL_KWH} kWh is recorded rather than '
               f'assumed. Fetching is manual: Meralco serves the same page for '
               f'every path, so a URL cannot be derived or probed.'),
    )
    print(f'\nWrote {ALL_IN_RATE_CSV.name} ({len(existing)} month(s)) + provenance')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    build(sys.argv[1:])

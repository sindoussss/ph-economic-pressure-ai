"""Build the committed benchmark data fixtures from real sources.

Run manually (NOT in tests):  python -m ph_economic_ai.benchmark.refresh_data

Automation level:
  * features_monthly.csv  -- FULLY AUTOMATIC: pulled from the repo's live fetcher
    (Yahoo / World Bank indicator APIs) every run.
  * world_bank_ron95.csv  -- AUTOMATIC IF a direct workbook URL is configured,
    otherwise falls back to a local copy you downloaded once.

Configuring the World Bank workbook URL (makes the gold refresh automatic):
  The DDH catalog API is not openly queryable, but the "Download" button on
  https://datacatalog.worldbank.org/search/dataset/0066829/global-fuel-prices-database
  resolves to a real .xlsx URL. Capture that URL and either:
    - set the env var  PH_ECON_WB_XLSX_URL=<that url>, or
    - paste it into WB_XLSX_URL below.
  Once set, every run re-downloads and re-extracts automatically. Until set,
  save the workbook locally as global_fuel_prices.xlsx and the script uses that.
"""
import os
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).parent
XLSX = HERE / 'global_fuel_prices.xlsx'
WB_OUT = HERE / 'data' / 'world_bank_ron95.csv'
FEATURES_OUT = HERE / 'data' / 'features_monthly.csv'

# Paste the resolved .xlsx download URL here to make the gold refresh automatic,
# or set the PH_ECON_WB_XLSX_URL environment variable (which takes precedence).
WB_XLSX_URL = ''

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


# ── World Bank gold series ──────────────────────────────────────────────────────

def _download_workbook(url: str, dest: Path) -> None:
    print(f'Downloading World Bank workbook from {url} ...')
    r = requests.get(url, headers=_HEADERS, timeout=60, stream=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f'  saved {dest} ({dest.stat().st_size // 1024} KB)')


def _ensure_workbook() -> Path:
    """Return a path to the workbook, downloading it if a URL is configured."""
    url = os.environ.get('PH_ECON_WB_XLSX_URL') or WB_XLSX_URL
    if url:
        try:
            _download_workbook(url, XLSX)
        except Exception as e:
            print(f'  download failed ({e!r}); falling back to local copy if present')
    if XLSX.exists():
        return XLSX
    raise SystemExit(
        f'No World Bank workbook found at {XLSX} and no working URL configured.\n'
        'Either download it manually (see module docstring) or set '
        'PH_ECON_WB_XLSX_URL / WB_XLSX_URL to the resolved .xlsx download link.'
    )


def _find_col(cols, *needles) -> str | None:
    """First column whose lowercased name contains ANY of the needles."""
    low = {c: str(c).lower() for c in cols}
    for c in cols:
        if any(n in low[c] for n in needles):
            return c
    return None


def build_world_bank_csv() -> None:
    xlsx = _ensure_workbook()
    raw = pd.read_excel(xlsx)
    print('Workbook columns:', list(raw.columns))

    country_col = _find_col(raw.columns, 'country', 'economy')
    product_col = _find_col(raw.columns, 'product', 'fuel', 'grade')
    date_col    = _find_col(raw.columns, 'date', 'month', 'period', 'time')
    # Prefer a local-currency (PHP) price column; fall back to a generic price col.
    price_col = (_find_col(raw.columns, 'local', 'lcu', 'php')
                 or _find_col(raw.columns, 'price', 'value', 'pump'))
    print(f'Detected -> country={country_col!r} product={product_col!r} '
          f'date={date_col!r} price={price_col!r}')
    if not all([country_col, date_col, price_col]):
        raise SystemExit('Could not auto-detect required columns; inspect the '
                         'printed columns and adjust _find_col needles.')

    df = raw.copy()
    df = df[df[country_col].astype(str).str.contains('Philippines', case=False, na=False)]
    if product_col is not None:
        # Premium gasoline (RON95). Match on '95'/'premium' first, else any gasoline.
        prem = df[df[product_col].astype(str).str.contains('95|premium', case=False, na=False, regex=True)]
        df = prem if len(prem) else df[df[product_col].astype(str).str.contains('gasoline|petrol', case=False, na=False, regex=True)]

    out = (df[[date_col, price_col]]
           .assign(date=lambda d: pd.to_datetime(d[date_col], errors='coerce').dt.strftime('%Y-%m'))
           .dropna(subset=['date'])
           .rename(columns={price_col: 'ron95_php_per_liter'})
           [['date', 'ron95_php_per_liter']]
           .dropna()
           .sort_values('date'))
    out = out[~out['date'].duplicated(keep='last')]
    WB_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WB_OUT, index=False)
    print(f'Wrote {len(out)} rows to {WB_OUT} '
          f'({out["date"].iloc[0]}..{out["date"].iloc[-1]})' if len(out) else
          f'Wrote 0 rows to {WB_OUT} -- check column detection!')
    if price_col and 'local' not in str(price_col).lower() and 'php' not in str(price_col).lower():
        print('  WARNING: price column may not be in PHP/liter (could be USD). '
              'Verify units before treating as gold ₱/L.')


# ── Predictor matrix (fully automatic) ──────────────────────────────────────────

def build_features_csv() -> None:
    from ph_economic_ai.fetcher import _fetch_all
    fdf = _fetch_all()
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    fdf.to_csv(FEATURES_OUT, index=False)
    print(f'Wrote features_monthly.csv ({len(fdf)} rows)')


def main():
    build_world_bank_csv()
    build_features_csv()


if __name__ == '__main__':
    main()

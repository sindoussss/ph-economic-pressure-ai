"""One-off: build the committed benchmark data fixtures from real sources.

Run manually (NOT in tests):  python -m ph_economic_ai.benchmark.refresh_data

Two fixtures are produced:

1. data/world_bank_ron95.csv  -- the gold ground-truth series.
   The World Bank Global Fuel Prices DB (dataset id 0066829) ships as an Excel
   workbook that is only available via the catalog UI (the DDH API is not openly
   queryable). Download it once from:
     https://datacatalogapi.worldbank.org/... (use the "Download" button at)
     https://datacatalogapi.worldbank.org  ->
     https://datacatalog.worldbank.org/search/dataset/0066829/global-fuel-prices-database
   Save the workbook next to this file as 'global_fuel_prices.xlsx', then run
   this script to extract the Philippines premium-gasoline (RON95) monthly column.

2. data/features_monthly.csv  -- the aligned real predictor matrix, produced by
   the repo's existing live fetcher (Yahoo/World-Bank), so run.py is offline-
   reproducible.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
XLSX = HERE / 'global_fuel_prices.xlsx'
WB_OUT = HERE / 'data' / 'world_bank_ron95.csv'
FEATURES_OUT = HERE / 'data' / 'features_monthly.csv'


def build_world_bank_csv() -> None:
    if not XLSX.exists():
        raise SystemExit(
            f'Download the World Bank workbook to {XLSX} first '
            '(see module docstring for the URL).'
        )
    # Column names vary by release; inspect once and adjust the filters below.
    raw = pd.read_excel(XLSX)
    print('Columns:', list(raw.columns))
    ph = raw[raw['country'].str.contains('Philippines', case=False, na=False)]
    ph = ph[ph['product'].str.contains('gasoline', case=False, na=False)]
    out = (ph[['date', 'price']]
           .assign(date=lambda d: pd.to_datetime(d['date']).dt.strftime('%Y-%m'))
           .rename(columns={'price': 'ron95_php_per_liter'})
           .sort_values('date'))
    WB_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WB_OUT, index=False)
    print(f'Wrote {len(out)} rows to {WB_OUT}')


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

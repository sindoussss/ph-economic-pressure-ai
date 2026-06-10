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


def _find_premium_sheet(xl: pd.ExcelFile) -> str:
    """The local-currency premium-gasoline (RON95+) sheet, excluding the USD one."""
    for sh in xl.sheet_names:
        s = sh.lower().replace(' ', '')
        if 'premium' in s and 'ron95' in s and 'usd' not in s:
            return sh
    raise SystemExit('Could not find a "Premium Gasoline RON95" LCU sheet; '
                     f'available sheets: {xl.sheet_names}')


def build_world_bank_csv() -> None:
    """Extract the PH premium-gasoline (RON95+) monthly series in PHP/litre.

    The workbook is wide-format: one sheet per fuel, country rows, month columns.
    We use the LCU (local-currency) premium-gasoline sheet so prices are in
    PHP/litre directly.
    """
    xlsx = _ensure_workbook()
    xl = pd.ExcelFile(xlsx)
    sheet = _find_premium_sheet(xl)
    df = xl.parse(sheet)
    print(f'Using sheet: {sheet!r}  shape={df.shape}')

    country_col = df.columns[0]
    units_col = df.columns[1] if 'unit' in str(df.columns[1]).lower() else None
    date_cols = [c for c in df.columns if hasattr(c, 'year') and hasattr(c, 'month')]
    if not date_cols:
        raise SystemExit('No datetime month columns found in the sheet.')

    matches = df[df[country_col].astype(str).str.contains('Philippines', case=False, na=False)]
    if matches.empty:
        raise SystemExit('No Philippines row found in the premium-gasoline sheet.')

    # Among PH rows, pick the PHP-units row with the most observed months.
    best, best_n, best_units = None, -1, None
    for _, row in matches.iterrows():
        units = str(row[units_col]) if units_col is not None else ''
        if units_col is not None and 'php' not in units.lower():
            continue
        n = int(row[date_cols].notna().sum())
        if n > best_n:
            best, best_n, best_units = row, n, units
    if best is None:                      # no explicit PHP row; fall back to fullest row
        best = max((r for _, r in matches.iterrows()),
                   key=lambda r: int(r[date_cols].notna().sum()))
        best_units = str(best[units_col]) if units_col is not None else '?'
    print(f'PH row units={best_units!r}, observed months={best_n}')

    s = best[date_cols].dropna()
    out = pd.DataFrame({
        'date': [pd.Timestamp(d).strftime('%Y-%m') for d in s.index],
        'ron95_php_per_liter': [round(float(v), 2) for v in s.values],
    })
    out = out[~out['date'].duplicated(keep='last')].sort_values('date')
    WB_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(WB_OUT, index=False)
    if len(out):
        print(f'Wrote {len(out)} rows to {WB_OUT} '
              f'({out["date"].iloc[0]}..{out["date"].iloc[-1]})')
    else:
        print(f'Wrote 0 rows to {WB_OUT} -- check sheet/row detection!')
    if 'php' not in (best_units or '').lower():
        print('  WARNING: units may not be PHP/litre. Verify before treating as gold ₱/L.')


# ── Predictor matrix (fully automatic) ──────────────────────────────────────────

def _yahoo_monthly(ticker: str, rng: str = '10y') -> pd.Series:
    """Monthly close series indexed 'YYYY-MM' (mirrors fetcher._fetch_yahoo, longer range)."""
    import datetime as _dt
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    r = requests.get(url, params={'interval': '1mo', 'range': rng},
                     headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'},
                     timeout=15)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    ts = res['timestamp']
    closes = res['indicators']['quote'][0]['close']
    dates = [_dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc).strftime('%Y-%m') for t in ts]
    s = pd.Series(closes, index=dates, dtype=float).dropna().round(2)
    return s[~s.index.duplicated(keep='last')]


def build_features_csv() -> None:
    """Real monthly predictors aligned for the backtest: Brent oil, USD/PHP, the
    RBOB-derived gas proxy, and a seasonal demand index. Skips PSEi (Yahoo ^PSEi
    404s and it is not a model predictor)."""
    from ph_economic_ai.fetcher import _compute_demand
    oil = _yahoo_monthly('BZ=F')      # Brent crude, USD/bbl
    usd = _yahoo_monthly('PHP=X')     # PHP per USD
    rbob = _yahoo_monthly('RB=F')     # RBOB gasoline futures, USD/gal
    base = pd.concat([oil.rename('oil_price'), usd.rename('usd_php'),
                      rbob.rename('rbob')], axis=1).dropna()
    # RBOB -> PHP/litre proxy (same formula as fetcher._fetch_doe_prices)
    base['gas_price'] = ((base['rbob'] / 3.785 * base['usd_php']) * 1.35 + 12).round(2)
    base = base.drop(columns=['rbob']).reset_index().rename(columns={'index': 'date'})
    base['demand_index'] = _compute_demand(base['date'].tolist())
    base = base.sort_values('date')
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(FEATURES_OUT, index=False)
    print(f'Wrote features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')


FX_OUT = HERE / 'data' / 'usd_php_monthly.csv'
CPI_OUT = HERE / 'data' / 'ph_cpi_monthly.csv'
FRED_CPI_ID = 'PHLCPIALLMINMEI'   # OECD MEI monthly CPI, Philippines (index)


def build_fx_csv() -> None:
    """USD/PHP monthly close from Yahoo -> data/usd_php_monthly.csv."""
    fx = _yahoo_monthly('PHP=X')
    df = fx.rename('usd_php').reset_index()
    df.columns = ['date', 'usd_php']
    FX_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FX_OUT, index=False)
    print(f'Wrote usd_php_monthly.csv ({len(df)} rows, {df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def build_cpi_csv() -> None:
    """PH monthly CPI index from FRED -> data/ph_cpi_monthly.csv.

    If FRED is unreachable or the id is retired, download manually from DBnomics:
      https://api.db.nomics.world/v22/series/IMF/IFS/M.PH.PCPI_IX?observations=1
    and save a 2-column CSV 'date,cpi_index' (date YYYY-MM) to CPI_OUT.
    """
    import io
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_CPI_ID}'
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    raw.columns = ['date', 'cpi_index']
    raw['date'] = pd.to_datetime(raw['date']).dt.strftime('%Y-%m')
    raw = raw[pd.to_numeric(raw['cpi_index'], errors='coerce').notna()]
    CPI_OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(CPI_OUT, index=False)
    print(f'Wrote ph_cpi_monthly.csv ({len(raw)} rows, {raw["date"].iloc[0]}..{raw["date"].iloc[-1]})')


LONG_FEATURES_OUT = HERE / 'data' / 'features_monthly_long.csv'


def build_long_features(rng: str = 'max') -> None:
    """Longer-history predictor matrix (default Yahoo range='max') for the MoM
    nowcast longer-sample confirmation. Same columns/derivations as
    build_features_csv, just a longer window -> data/features_monthly_long.csv."""
    from ph_economic_ai.fetcher import _compute_demand
    oil = _yahoo_monthly('BZ=F', rng)
    usd = _yahoo_monthly('PHP=X', rng)
    rbob = _yahoo_monthly('RB=F', rng)
    base = pd.concat([oil.rename('oil_price'), usd.rename('usd_php'),
                      rbob.rename('rbob')], axis=1).dropna()
    base['gas_price'] = ((base['rbob'] / 3.785 * base['usd_php']) * 1.35 + 12).round(2)
    base = base.drop(columns=['rbob']).reset_index().rename(columns={'index': 'date'})
    base['demand_index'] = _compute_demand(base['date'].tolist())
    base = base.sort_values('date')
    LONG_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(LONG_FEATURES_OUT, index=False)
    print(f'Wrote features_monthly_long.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')


FOOD_FEATURES_OUT = HERE / 'data' / 'food_features_monthly.csv'


def build_food_features(rng: str = 'max') -> None:
    """Free global food-commodity predictor panel for the Food-CPI nowcast:
    Yahoo agri futures + oil + USD/PHP -> data/food_features_monthly.csv."""
    cols = {'ZR=F': 'rice', 'ZW=F': 'wheat', 'ZC=F': 'corn', 'ZS=F': 'soybean',
            'BZ=F': 'oil_price', 'PHP=X': 'usd_php'}
    parts = [_yahoo_monthly(t, rng).rename(name) for t, name in cols.items()]
    base = pd.concat(parts, axis=1).dropna().reset_index().rename(columns={'index': 'date'})
    base = base.sort_values('date')
    FOOD_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(FOOD_FEATURES_OUT, index=False)
    print(f'Wrote food_features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')


ELECTRICITY_FEATURES_OUT = HERE / 'data' / 'electricity_features_monthly.csv'


def build_electricity_features(rng: str = 'max') -> None:
    """Free energy predictor panel for the Electricity-CPI nowcast:
    Yahoo Brent + natural gas + USD/PHP -> data/electricity_features_monthly.csv."""
    cols = {'BZ=F': 'oil_price', 'NG=F': 'natgas', 'PHP=X': 'usd_php'}
    parts = [_yahoo_monthly(t, rng).rename(name) for t, name in cols.items()]
    base = pd.concat(parts, axis=1).dropna().reset_index().rename(columns={'index': 'date'})
    base = base.sort_values('date')
    ELECTRICITY_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(ELECTRICITY_FEATURES_OUT, index=False)
    print(f'Wrote electricity_features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')


def main():
    build_world_bank_csv()
    build_features_csv()


if __name__ == '__main__':
    main()

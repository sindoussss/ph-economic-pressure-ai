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
from ph_economic_ai.benchmark.provenance import write_record

HERE = Path(__file__).parent
XLSX = HERE / 'global_fuel_prices.xlsx'
# Provenance constants. These describe what the builders below actually do, so the
# committed sidecars record the endpoint decision rather than leaving it buried in
# a function argument, which is how the interval=1mo defect stayed invisible.
YAHOO_CHART = 'https://query1.finance.yahoo.com/v8/finance/chart/<ticker>'
DAILY_TRANSFORMS = [
    'request interval=1d; never interval=1mo, which omits months non-randomly',
    'resample to month start, value = last observed close of the month',
    'round to 2dp; drop duplicated months keeping last',
    'inner join across tickers; dropna',
    'assert_complete_months before writing',
]
CHUNKED_TRANSFORMS = [
    'request interval=1d in explicit five-year period1/period2 windows; '
    'range=max is insufficient because Yahoo thins old daily history',
    'skip windows preceding the ticker (HTTP 400 or empty payload)',
    'concatenate, sort, drop duplicate days keeping last',
] + DAILY_TRANSFORMS[1:]

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
    write_record(WB_OUT, source='World Bank Global Fuel Prices Database workbook '
                                 '(datacatalog.worldbank.org/search/dataset/0066829/'
                                 'global-fuel-prices-database), Open Database License (ODbL)',
                 params={'workbook_url': os.environ.get('PH_ECON_WB_XLSX_URL') or WB_XLSX_URL
                                         or 'not configured; built from the committed copy',
                         'workbook_path': str(xlsx), 'sheet': sheet},
                 transformations=['parse the LCU premium-gasoline (RON95+) sheet',
                                  'select the Philippines row with the most observed months',
                                  'label months YYYY-MM, round to 2dp',
                                  'drop duplicate months keeping last'],
                 units='PHP per litre',
                 notes='RON95 gold target. RSK-007: the workbook is now committed at '
                       'ph_economic_ai/benchmark/global_fuel_prices.xlsx, so this record '
                       'both identifies and makes it retrievable. ODbL requires attribution '
                       'on redistribution -- given here and in the manuscript References.')
    if len(out):
        print(f'Wrote {len(out)} rows to {WB_OUT} '
              f'({out["date"].iloc[0]}..{out["date"].iloc[-1]})')
    else:
        print(f'Wrote 0 rows to {WB_OUT} -- check sheet/row detection!')
    if 'php' not in (best_units or '').lower():
        print('  WARNING: units may not be PHP/litre. Verify before treating as gold ₱/L.')


# ── Predictor matrix (fully automatic) ──────────────────────────────────────────

def _yahoo_monthly(ticker: str, rng: str = '10y') -> pd.Series:
    """Monthly close series indexed 'YYYY-MM', built by resampling DAILY data.

    Do not use Yahoo's `interval=1mo` endpoint here. It silently omits months, and
    not at random: measured over ten years, `PHP=X` returned no October at all in
    10 of 10 years, while `BZ=F` and `RB=F` each dropped 17 scattered months. The
    three series are then inner-joined, so every gap compounds and the feature
    panel lost 25 of 120 months. Because the backtest lags by ROW, those holes
    turned a one-row lag into a two or three month lag without any error.

    Resampling `interval=1d` ourselves gives 121 of 121 months for all three
    tickers, with the month labelled by its start and valued at its last observed
    close.

    `rng='max'` is not enough on its own: Yahoo thins old history even on the daily
    endpoint, so a single max-range call still lost 22 to 46 months per ticker
    (`PHP=X` again dropped every October from 2004 to 2007). Long history is
    therefore fetched in explicit five-year epoch windows, which returns true daily
    data throughout -- `PHP=X` goes from 22 missing months to 0.
    """
    if rng == 'max':
        daily = _yahoo_daily_chunked(ticker)
    else:
        daily = _yahoo_daily(ticker, params={'interval': '1d', 'range': rng})
    if daily.empty:
        return pd.Series(dtype=float)
    monthly = daily.resample('MS').last().dropna().round(2)
    monthly.index = monthly.index.strftime('%Y-%m')
    return monthly[~monthly.index.duplicated(keep='last')]


def _yahoo_daily(ticker: str, params: dict) -> pd.Series:
    """Daily closes for one chart request. Empty series if the window predates the
    ticker (Yahoo answers 400) or holds no observations."""
    import datetime as _dt
    url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}'
    r = requests.get(url, params=params,
                     headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'},
                     timeout=30)
    if r.status_code == 400:
        return pd.Series(dtype=float)
    r.raise_for_status()
    res = r.json()['chart']['result'][0]
    if 'timestamp' not in res:
        return pd.Series(dtype=float)
    days = pd.to_datetime(
        [_dt.datetime.fromtimestamp(t, tz=_dt.timezone.utc).date()
         for t in res['timestamp']])
    return pd.Series(res['indicators']['quote'][0]['close'],
                     index=days, dtype=float).dropna()


def _yahoo_daily_chunked(ticker: str, start_year: int = 1999,
                         span_years: int = 5) -> pd.Series:
    """Full daily history, requested in fixed epoch windows and concatenated.

    Windows before the ticker's first trade come back 400 or empty and are skipped,
    so the caller does not need to know when each contract starts.
    """
    import datetime as _dt
    end_year = _dt.datetime.now(_dt.timezone.utc).year + 1
    parts = []
    for y0 in range(start_year, end_year, span_years):
        y1 = min(y0 + span_years, end_year)
        p1 = int(_dt.datetime(y0, 1, 1, tzinfo=_dt.timezone.utc).timestamp())
        p2 = int(_dt.datetime(y1, 1, 1, tzinfo=_dt.timezone.utc).timestamp())
        part = _yahoo_daily(ticker, {'interval': '1d', 'period1': p1, 'period2': p2})
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.Series(dtype=float)
    daily = pd.concat(parts).sort_index()
    return daily[~daily.index.duplicated(keep='last')]


def assert_complete_months(index, label: str) -> None:
    """Raise if `index` skips a calendar month.

    The backtest lags by row position, so a gap silently changes what a lag means.
    This makes that failure loud at build time instead of leaving it to be found in
    the results. Call it on anything written to `data/`.
    """
    idx = pd.PeriodIndex(pd.to_datetime(sorted(index), format='%Y-%m'), freq='M')
    if len(idx) == 0:
        raise SystemExit(f'{label}: empty index')
    expected = pd.period_range(idx.min(), idx.max(), freq='M')
    missing = sorted(set(expected) - set(idx))
    if missing:
        shown = ', '.join(str(m) for m in missing[:12])
        more = '' if len(missing) <= 12 else f' and {len(missing) - 12} more'
        raise SystemExit(
            f'{label}: {len(missing)} calendar months missing from '
            f'{idx.min()}..{idx.max()} ({shown}{more}). Row lags would not be '
            f'month lags. Fix the source before writing this file.')


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
    # The three tickers are inner-joined above, so any month absent from one is
    # absent from all. Fail here rather than write a panel whose row lags are not
    # month lags.
    assert_complete_months(base['date'], 'features_monthly.csv')
    FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(FEATURES_OUT, index=False)
    write_record(FEATURES_OUT, source=YAHOO_CHART,
                 params={'tickers': {'BZ=F': 'oil_price', 'PHP=X': 'usd_php', 'RB=F': 'rbob'},
                         'interval': '1d', 'range': '10y'},
                 transformations=DAILY_TRANSFORMS + [
                     'gas_price = (rbob / 3.785 * usd_php) * 1.35 + 12',
                     'demand_index from fetcher._compute_demand'],
                 units='USD/bbl, PHP per USD, PHP per litre',
                 notes='Short-window predictor panel.')
    print(f'Wrote features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')


FX_OUT = HERE / 'data' / 'usd_php_monthly.csv'
CPI_OUT = HERE / 'data' / 'ph_cpi_monthly.csv'
# RSK-002, resolved 2026-07-28. The committed ph_cpi_monthly.csv is byte-identical
# to IMF IFS M.PH.PCPI_IX served by DBnomics: 821 months, 1957-01..2025-05, maximum
# absolute difference 0.0000 across every overlapping month. The manuscript's stated
# provenance (IMF IFS via DBnomics) was therefore correct and this module's label was
# not. The FRED id below additionally now returns HTTP 404, so it could not have been
# the live source either. DBnomics is the primary path; FRED is retained only as a
# fallback and is expected to fail.
DBNOMICS_CPI_URL = ('https://api.db.nomics.world/v22/series/IMF/IFS/'
                    'M.PH.PCPI_IX?observations=1')
FRED_CPI_ID = 'PHLCPIALLMINMEI'   # retired: returns 404 as of 2026-07-28


def build_fx_csv() -> None:
    """USD/PHP monthly close from Yahoo -> data/usd_php_monthly.csv."""
    fx = _yahoo_monthly('PHP=X')
    df = fx.rename('usd_php').reset_index()
    df.columns = ['date', 'usd_php']
    assert_complete_months(df['date'], 'usd_php_monthly.csv')
    FX_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FX_OUT, index=False)
    write_record(FX_OUT, source=YAHOO_CHART,
                 params={'ticker': 'PHP=X', 'interval': '1d', 'range': '10y'},
                 transformations=DAILY_TRANSFORMS,
                 units='PHP per USD',
                 notes='Before 2026-07-28 this file was built from interval=1mo and '
                       'was missing all ten Octobers.')
    print(f'Wrote usd_php_monthly.csv ({len(df)} rows, {df["date"].iloc[0]}..{df["date"].iloc[-1]})')


def _cpi_from_dbnomics() -> pd.DataFrame:
    """PH monthly CPI index from IMF IFS via DBnomics. The authoritative path."""
    r = requests.get(DBNOMICS_CPI_URL, headers=_HEADERS, timeout=45)
    r.raise_for_status()
    doc = r.json()['series']['docs'][0]
    frame = pd.DataFrame({'date': doc['period'], 'cpi_index': doc['value']})
    frame = frame[pd.to_numeric(frame['cpi_index'], errors='coerce').notna()]
    frame['date'] = frame['date'].astype(str).str.slice(0, 7)
    return frame.reset_index(drop=True)


def _cpi_from_fred() -> pd.DataFrame:
    """Retired fallback. Kept so the failure is explicit rather than silent."""
    import io
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_CPI_ID}'
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    raw.columns = ['date', 'cpi_index']
    raw['date'] = pd.to_datetime(raw['date']).dt.strftime('%Y-%m')
    return raw[pd.to_numeric(raw['cpi_index'], errors='coerce').notna()].reset_index(drop=True)


def build_cpi_csv() -> None:
    """PH monthly CPI index -> data/ph_cpi_monthly.csv, from IMF IFS via DBnomics.

    This used to fetch FRED `PHLCPIALLMINMEI` and label it OECD MEI, which created
    `RSK-002`: the manuscript claimed IMF IFS via DBnomics and the code claimed
    something else. The conflict is resolved in favour of the manuscript, on
    evidence rather than preference. The committed file matches IMF IFS
    `M.PH.PCPI_IX` exactly over all 821 shared months, and the FRED id now 404s, so
    it cannot have been the source of the committed data.

    Because the two agree exactly, this repair changes no downstream result.
    """
    try:
        frame = _cpi_from_dbnomics()
        source = 'IMF IFS M.PH.PCPI_IX via DBnomics'
        params = {'url': DBNOMICS_CPI_URL, 'provider': 'DBnomics',
                  'dataset': 'IMF/IFS', 'series': 'M.PH.PCPI_IX'}
    except Exception as primary_error:
        print(f'  DBnomics unavailable ({primary_error!r}); trying the retired FRED path')
        frame = _cpi_from_fred()
        source = f'FRED series {FRED_CPI_ID} (retired fallback)'
        params = {'url': f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_CPI_ID}',
                  'series_id': FRED_CPI_ID}

    frame = frame[~frame['date'].duplicated(keep='last')].sort_values('date')
    assert_complete_months(frame['date'], 'ph_cpi_monthly.csv')
    CPI_OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CPI_OUT, index=False)
    write_record(CPI_OUT, source=source, params=params,
                 transformations=['take the observations array',
                                  'drop non-numeric observations',
                                  'label months YYYY-MM',
                                  'drop duplicate months keeping last',
                                  'assert_complete_months before writing'],
                 units='CPI index',
                 notes='RSK-002 RESOLVED 2026-07-28. Verified byte-identical to the '
                       'previously committed file over all 821 months.')
    print(f'Wrote ph_cpi_monthly.csv ({len(frame)} rows, '
          f'{frame["date"].iloc[0]}..{frame["date"].iloc[-1]}) from {source}')


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
    assert_complete_months(base['date'], 'features_monthly_long.csv')
    LONG_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(LONG_FEATURES_OUT, index=False)
    write_record(LONG_FEATURES_OUT, source=YAHOO_CHART,
                 params={'tickers': {'BZ=F': 'oil_price', 'PHP=X': 'usd_php', 'RB=F': 'rbob'},
                         'interval': '1d', 'window': 'five-year epoch chunks'},
                 transformations=CHUNKED_TRANSFORMS + [
                     'gas_price = (rbob / 3.785 * usd_php) * 1.35 + 12',
                     'demand_index from fetcher._compute_demand'],
                 units='USD/bbl, PHP per USD, PHP per litre',
                 notes='Long-window panel.')
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
    assert_complete_months(base['date'], 'food_features_monthly.csv')
    FOOD_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(FOOD_FEATURES_OUT, index=False)
    write_record(FOOD_FEATURES_OUT, source=YAHOO_CHART,
                 params={'tickers': {'ZR=F': 'rice', 'ZW=F': 'wheat', 'ZC=F': 'corn',
                                     'ZS=F': 'soybean', 'BZ=F': 'oil_price', 'PHP=X': 'usd_php'},
                         'interval': '1d', 'window': 'five-year epoch chunks'},
                 transformations=CHUNKED_TRANSFORMS,
                 units='futures settlement, PHP per USD',
                 notes='Food-CPI predictor panel.')
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
    assert_complete_months(base['date'], 'electricity_features_monthly.csv')
    ELECTRICITY_FEATURES_OUT.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(ELECTRICITY_FEATURES_OUT, index=False)
    write_record(ELECTRICITY_FEATURES_OUT, source=YAHOO_CHART,
                 params={'tickers': {'BZ=F': 'oil_price', 'NG=F': 'natgas', 'PHP=X': 'usd_php'},
                         'interval': '1d', 'window': 'five-year epoch chunks'},
                 transformations=CHUNKED_TRANSFORMS,
                 units='USD/bbl, USD/MMBtu, PHP per USD',
                 notes='Electricity-CPI predictor panel.')
    print(f'Wrote electricity_features_monthly.csv ({len(base)} rows, '
          f'{base["date"].iloc[0]}..{base["date"].iloc[-1]})')


def main():
    build_world_bank_csv()
    build_features_csv()


if __name__ == '__main__':
    main()

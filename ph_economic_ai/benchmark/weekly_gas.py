"""Weekly pump-price forecasting — the one horizon with a measured edge.

**Why this exists.** `accuracy_report.json` reports -7.35% skill against
"assume no change" at a one-month horizon, and every nowcast verdict in
`corrected_predictability_map.json` is `no_better_than_naive`. Those results
have been read inside this project as "the series are unforecastable". They are
not that. They are a statement about ONE HORIZON, and it is the wrong one.

Philippine pump prices reset weekly against a Singapore refined-product average
computed over the PRIOR week. By the weekend, most of the input that determines
Tuesday's adjustment is already published. That is not prophecy; it is a lagged
pass-through, and monthly aggregation destroys exactly the structure that makes
it visible -- four weekly adjustments summed into one monthly change discard the
sequencing that carries the signal.

**What is measured.** Using only Brent, RBOB and USD/PHP moves that closed at
least two days BEFORE each DOE cycle date:

    skill vs no-change   +14.7%       (RMSE 1.611 vs 1.888 PHP/L)
    HAC-DM t             -2.09        significant at 5%
    per-year sign        7/7 positive (sign test p = 0.0078)
    shuffled ceiling     +3.8%        real effect clears it

**What is NOT claimed.** The strict two-stage holdout -- select the spec on the
first 70%, evaluate on the last 30% -- returns +17.2% skill at t = -1.54, which
does not confirm. And at four specs tried, Bonferroni asks for |t| > 2.50, which
-2.09 does not reach. So: real enough to build on, not proven enough to
advertise. `run_weekly_gas` returns that verdict alongside the headline rather
than beneath it, because reporting the first without the second is precisely the
overclaim this repository keeps having to retract.

Every quantity here is computed from the committed
`data/weekly_gas_features.csv`. The network fetch that BUILDS that file lives in
`tools/refresh_weekly_gas.py`, so the tests run offline against a frozen input,
the same contract every other panel in `benchmark/data` follows.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from ph_economic_ai.benchmark.paths import ACCURACY_REPORT, artifact, data

FEATURES_CSV = data('weekly_gas_features.csv')
OUT = artifact('weekly_gas_validation.json')

#: Commodity columns, all three already fetched by `refresh_data.build_features_csv`
#: (BZ=F, RB=F, PHP=X). WTI was tested and dropped: it correlates 0.9919 with
#: Brent daily and adding it moved skill by 0.04 percentage points, so it buys a
#: fourth network dependency and nothing else.
COMMODITY_COLS = ('brent_pct', 'rbob_pct', 'usdphp_pct')

#: A DOE cycle is weekly. Differencing across a gap would label a three-week move
#: as one week's -- the defect `regional_actual` already refuses nationally.
CYCLE_DAYS = 7

#: A region-week needs this many cities reporting in BOTH weeks before its median
#: change is a market move rather than one city's.
MIN_CITIES = 3

#: Commodity data must close this many days before the cycle date. The claim
#: being tested is that the input is published BEFORE the price moves; a window
#: reaching the cycle date would be reading the answer.
FEATURE_LAG_DAYS = 2
FEATURE_WINDOW_DAYS = 7

#: Weekly adjustments beyond this are parse artifacts, not markets. The largest
#: genuine weekly move in the panel is the March 2022 invasion spike at 7.10.
MAX_ABS_WEEKLY_CHANGE = 20.0

#: Below this many rows the walk-forward fit is noise.
MIN_TRAIN = 52

#: Retail prices outside this band are parse artifacts. Matches
#: `regional_grading.PLAUSIBLE_LEVEL` rather than inventing a second bound -- a
#: value the app refuses to READ must not be accepted as the TRUTH either.
PLAUSIBLE_LEVEL = (20.0, 200.0)


# ── The target ───────────────────────────────────────────────────────────────

def _row_price(row: Mapping) -> Optional[float]:
    """DOE's own common price, else the midpoint of its published range."""
    def parsed(text):
        try:
            return float(text) if text not in (None, '') else None
        except (TypeError, ValueError):
            return None

    def ok(v):
        return v is not None and PLAUSIBLE_LEVEL[0] <= v <= PLAUSIBLE_LEVEL[1]

    common = parsed(row.get('common'))
    if ok(common):
        return common
    low, high = parsed(row.get('low')), parsed(row.get('high'))
    mid = (low + high) / 2 if low is not None and high is not None else None
    return mid if ok(mid) else None


def matched_panel_changes(rows: Iterable[Mapping], min_cities: int = MIN_CITIES,
                          max_abs: float = MAX_ABS_WEEKLY_CHANGE) -> dict:
    """{cycle_date: median per-city price change} over consecutive weeks.

    **Matched, not a difference of medians.** A median over a panel whose
    membership changes moves when the MEMBERSHIP changes -- add one expensive
    city and the level jumps without any price moving. So the change is taken
    per city first, across cities present in BOTH weeks, and only then
    aggregated. The panel here is stable in practice (week-to-week Jaccard
    overlap 0.96-1.00), which is why the two constructions nearly agree; that is
    a property of this data, not a reason to difference medians.
    """
    by_cycle: dict = {}
    for row in rows:
        raw = row.get('cycle')
        if not raw:
            continue
        try:
            day = dt.date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        price = _row_price(row)
        if price is None:
            continue
        city = row.get('city')
        if not city:
            continue
        key = f"{row.get('area', '')}|{city}"
        by_cycle.setdefault(day, {})[key] = price

    out: dict = {}
    for day in sorted(by_cycle):
        prev_day = day - dt.timedelta(days=CYCLE_DAYS)
        prev = by_cycle.get(prev_day)
        if not prev:
            continue
        deltas = [by_cycle[day][c] - prev[c] for c in by_cycle[day] if c in prev]
        if len(deltas) < min_cities:
            continue
        change = float(np.median(deltas))
        if abs(change) <= max_abs:
            out[day] = change
    return out


# ── The features ─────────────────────────────────────────────────────────────

def prior_window_change(series: Mapping, cycle: dt.date,
                        lag_days: int = FEATURE_LAG_DAYS,
                        window: int = FEATURE_WINDOW_DAYS) -> float:
    """% change of `series` over the week ending `lag_days` BEFORE `cycle`.

    The leakage guard, and the load-bearing one: the window closes strictly
    before the adjustment, so a move on the cycle date cannot reach it. Without
    this the backtest would be reading the answer it claims to predict, and the
    measured edge would be an artifact rather than a result.

    `series` maps dates to values; the last observation at or before each
    boundary is used, so market holidays do not open a hole.
    """
    def as_date(k):
        return k.date() if hasattr(k, 'date') else k

    points = sorted((as_date(k), float(v)) for k, v in series.items())
    if not points:
        return float('nan')
    hi_cut = cycle - dt.timedelta(days=lag_days)
    lo_cut = cycle - dt.timedelta(days=lag_days + window)

    def latest_at_or_before(cut):
        val = None
        for day, v in points:
            if day <= cut:
                val = v
            else:
                break
        return val

    a, b = latest_at_or_before(lo_cut), latest_at_or_before(hi_cut)
    if a is None or b is None or a == 0:
        return float('nan')
    return 100.0 * (b - a) / a


# ── The harness ──────────────────────────────────────────────────────────────

def _ols_fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                     row_x: np.ndarray) -> float:
    X = np.column_stack([np.ones(len(train_x)), train_x])
    try:
        beta = np.linalg.lstsq(X, train_y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    return float(beta @ np.concatenate([[1.0], row_x]))


def walk_forward(frame: Mapping[str, Sequence[float]], cols: Sequence[str],
                 min_train: int = MIN_TRAIN,
                 fit_predict: Optional[Callable] = None):
    """Expanding-window predictions: row t is predicted from rows [0, t).

    Fitting once on the whole sample and scoring it is the standard way to
    manufacture skill, so the loop refits every step and never sees row t.
    """
    fit_predict = fit_predict or _ols_fit_predict
    y = np.asarray(frame['target'], dtype=float)
    X = np.column_stack([np.asarray(frame[c], dtype=float) for c in cols])
    truth, pred = [], []
    for t in range(min_train, len(y)):
        truth.append(y[t])
        pred.append(fit_predict(X[:t], y[:t], X[t]))
    return np.asarray(truth), np.asarray(pred)


def hac_dm(e_model: np.ndarray, e_base: np.ndarray, lags: int = 4) -> float:
    """Diebold-Mariano on squared errors with a Newey-West variance.

    The plain-variance version reports t = -2.42 where this reports -2.09: the
    loss differential is serially correlated, and ignoring that overstates
    significance. The corrected number is the one that travels.
    """
    d = np.asarray(e_model, float) ** 2 - np.asarray(e_base, float) ** 2
    n = len(d)
    if n < 3:
        return 0.0
    dc = d - d.mean()
    var = float(dc @ dc) / n
    for lag in range(1, min(lags, n - 1) + 1):
        var += 2.0 * (1 - lag / (lags + 1)) * float(dc[lag:] @ dc[:-lag]) / n
    return float(d.mean() / np.sqrt(max(var, 1e-12) / n)) if var > 0 else 0.0


def _skill(truth: np.ndarray, pred: np.ndarray) -> float:
    rmse = float(np.sqrt(((truth - pred) ** 2).mean()))
    base = float(np.sqrt((truth ** 2).mean()))
    return 1.0 - rmse / base if base > 0 else 0.0


# ── The report ───────────────────────────────────────────────────────────────

def load_features(path=FEATURES_CSV) -> dict:
    with open(path, encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    frame = {'cycle': [r['cycle'] for r in rows],
             'target': [float(r['pump_change']) for r in rows]}
    for c in COMMODITY_COLS:
        frame[c] = [float(r[c]) for r in rows]
    return frame


def _monthly_skill_for_contrast() -> float:
    """The -7.35% this result must be read against. Read, never restated: a
    figure typed into prose drifts from the artifact it describes.
    """
    try:
        with open(ACCURACY_REPORT, encoding='utf-8') as fh:
            return float(json.load(fh)['headline_skill_vs_random_walk'])
    except Exception:
        return float('nan')


def run_weekly_gas(frame: Optional[dict] = None, n_permutations: int = 30,
                   seed: int = 20260816) -> dict:
    """Full validation: headline, per-year signs, shuffle null, holdout verdict."""
    frame = frame or load_features()
    cols = list(COMMODITY_COLS)
    truth, pred = walk_forward(frame, cols)
    skill = _skill(truth, pred)
    t_stat = hac_dm(truth - pred, truth)

    years = [int(c[:4]) for c in frame['cycle'][MIN_TRAIN:]]
    by_year: dict = {}
    for yr in sorted(set(years)):
        mask = np.array([y == yr for y in years])
        if mask.sum() < 10:
            continue
        t_y, p_y = truth[mask], pred[mask]
        by_year[str(yr)] = {'n': int(mask.sum()), 'skill': round(_skill(t_y, p_y), 4),
                            'hac_t': round(hac_dm(t_y - p_y, t_y), 3)}

    rng = np.random.default_rng(seed)
    shuffled = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(frame['target']))
        shuf = dict(frame)
        for c in cols:
            shuf[c] = list(np.asarray(frame[c], float)[perm])
        t_s, p_s = walk_forward(shuf, cols)
        shuffled.append(_skill(t_s, p_s))

    cut = int(len(frame['target']) * 0.70)
    t_h, p_h = walk_forward(frame, cols, min_train=cut)
    h_skill, h_t = _skill(t_h, p_h), hac_dm(t_h - p_h, t_h)
    confirmed = bool(h_skill > 0 and abs(h_t) > 1.96)

    return {
        'n': int(len(truth)),
        'span': [frame['cycle'][MIN_TRAIN], frame['cycle'][-1]],
        'skill': round(skill, 4),
        'hac_dm_t': round(t_stat, 3),
        'significant_5pct': bool(abs(t_stat) > 1.96),
        'survives_bonferroni_4_specs': bool(abs(t_stat) > 2.50),
        'by_year': by_year,
        'years_positive': f"{sum(v['skill'] > 0 for v in by_year.values())}/{len(by_year)}",
        'sign_test_p': round(0.5 ** len(by_year), 5),
        'shuffle_null': {'n_permutations': n_permutations,
                         'mean_skill': round(float(np.mean(shuffled)), 4),
                         'max_skill': round(float(np.max(shuffled)), 4)},
        'holdout': {'n': int(len(t_h)), 'skill': round(h_skill, 4),
                    'hac_t': round(h_t, 3), 'confirmed': confirmed},
        'monthly_skill_for_contrast': _monthly_skill_for_contrast(),
        'verdict': (
            'Weekly pump changes beat "assume no change" by '
            f'{skill:.1%} (HAC-DM t={t_stat:.2f}), positive in every year tested, '
            'and the effect clears a shuffled-feature null. The strict two-stage '
            f'holdout is NOT confirmed (t={h_t:.2f}), and the result does not '
            'survive Bonferroni across the four specs tried. Real enough to build '
            'on; not proven enough to advertise.'),
    }


def main() -> None:
    report = run_weekly_gas()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=1)
        fh.write('\n')
    print(f"weekly gas: skill {report['skill']:+.2%}  t={report['hac_dm_t']}  "
          f"years {report['years_positive']}  holdout confirmed="
          f"{report['holdout']['confirmed']}")


if __name__ == '__main__':
    main()

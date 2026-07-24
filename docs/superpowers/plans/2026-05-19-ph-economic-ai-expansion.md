# PH Economic AI Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new Philippine economic indicators (PSEi, CPI, BSP rate, OFW remittances), upgrade from RandomForest to HistGradientBoosting, and surface new data + model insights via two new tabs in the existing Dashboard.

**Architecture:** New data flows from `fetcher.py` → expanded DataFrame (9 columns) → `preprocessing.py` picks up available extra columns dynamically → `model.py` trains HGB and exposes feature importances + CV-RMSE → `main.py` wires CV-RMSE through → `DashboardPage` renders Indicators and Model Insights tabs alongside the existing Overview tab.

**Tech Stack:** PyQt6 6.10.0, scikit-learn `HistGradientBoostingRegressor`, matplotlib FigureCanvasQTAgg, requests, pandas, World Bank JSON API (no key), Yahoo Finance v8 JSON API (no key).

---

## File Map

| File | Change |
|------|--------|
| `ph_economic_ai/fetcher.py` | Add `_parse_world_bank_response`, `_forward_fill_annual`, `_fetch_world_bank`, `_fetch_psei`; update `_fetch_all` |
| `ph_economic_ai/utils/preprocessing.py` | `build_features` uses available extra columns dynamically |
| `ph_economic_ai/model.py` | Swap RF → HGB; update `predict`, `get_training_predictions`; add `get_feature_importances`, `cross_val_rmse`, `forecast` |
| `ph_economic_ai/main.py` | Compute `cv_rmse`; pass to `MainWindow` |
| `ph_economic_ai/ui/main_window.py` | Accept `cv_rmse`; add `_make_last_features`; add `cv_rmse`, `feature_importances`, `forecast_prices`, `df_raw` to result dict |
| `ph_economic_ai/ui/dashboard.py` | Refactor into `QTabWidget`; add Indicators tab and Model Insights tab |
| `ph_economic_ai/tests/test_fetcher.py` | Add `test_forward_fill_annual_basic`, `test_world_bank_parse_returns_series` |
| `ph_economic_ai/tests/test_model.py` | Update `test_train_returns_fitted_model`; add `test_feature_importances_shape`, `test_cross_val_rmse_positive` |

---

## Task 1: New Fetcher Functions (World Bank + PSEi)

**Files:**
- Modify: `ph_economic_ai/fetcher.py`
- Modify: `ph_economic_ai/tests/test_fetcher.py`

- [ ] **Step 1: Write the failing tests**

Add to `ph_economic_ai/tests/test_fetcher.py` (after the existing imports and `_sample_df`):

```python
from ph_economic_ai.fetcher import (
    _compute_demand, _load_cache, _save_cache,
    _parse_world_bank_response, _forward_fill_annual,
)


def test_forward_fill_annual_basic():
    annual = pd.Series({'2022': 5.0, '2023': 6.0})
    monthly = ['2022-06', '2022-12', '2023-03', '2023-09']
    result = _forward_fill_annual(annual, monthly)
    assert result['2022-06'] == 5.0
    assert result['2022-12'] == 5.0
    assert result['2023-03'] == 6.0
    assert result['2023-09'] == 6.0


def test_forward_fill_annual_carries_forward():
    annual = pd.Series({'2021': 3.0})
    monthly = ['2021-01', '2022-06']  # 2022 not in annual
    result = _forward_fill_annual(annual, monthly)
    assert result['2021-01'] == 3.0
    assert result['2022-06'] == 3.0  # carries last known value


def test_world_bank_parse_returns_series():
    sample_payload = [
        {'page': 1, 'pages': 1, 'per_page': 100, 'total': 2},
        [
            {'date': '2023', 'value': 5.97},
            {'date': '2022', 'value': 3.21},
            {'date': '2021', 'value': None},  # missing — should be dropped
        ]
    ]
    result = _parse_world_bank_response(sample_payload)
    assert isinstance(result, pd.Series)
    assert '2023' in result.index
    assert '2022' in result.index
    assert '2021' not in result.index  # None dropped
    assert abs(result['2023'] - 5.97) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_fetcher.py::test_forward_fill_annual_basic ph_economic_ai/tests/test_fetcher.py::test_world_bank_parse_returns_series -v
```

Expected: `ImportError` or `FAILED` — functions don't exist yet.

- [ ] **Step 3: Add `_parse_world_bank_response` and `_forward_fill_annual` to `fetcher.py`**

Add after the `_YAHOO_HEADERS` block (before `fetch_dataset`):

```python
def _parse_world_bank_response(payload: list) -> pd.Series:
    """Parse World Bank JSON array → Series indexed by 'YYYY', NaN dropped."""
    records = payload[1]
    data = {}
    for r in records:
        year = r.get('date')
        value = r.get('value')
        if year and value is not None:
            data[year] = float(value)
    series = pd.Series(data, dtype=float).dropna()
    series.index.name = None
    return series


def _forward_fill_annual(annual: pd.Series, monthly_index: list) -> pd.Series:
    """Broadcast annual values (keyed 'YYYY') to monthly 'YYYY-MM' strings."""
    sorted_years = sorted(annual.index)
    result = {}
    for ym in monthly_index:
        year = ym[:4]
        earlier = [y for y in sorted_years if y <= year]
        if earlier:
            result[ym] = annual[max(earlier)]
    return pd.Series(result, dtype=float)
```

- [ ] **Step 4: Run the two new tests to verify they pass**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_fetcher.py::test_forward_fill_annual_basic ph_economic_ai/tests/test_fetcher.py::test_forward_fill_annual_carries_forward ph_economic_ai/tests/test_fetcher.py::test_world_bank_parse_returns_series -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Add `_fetch_world_bank` and `_fetch_psei` to `fetcher.py`**

Add after `_forward_fill_annual`:

```python
def _fetch_world_bank(indicator_id: str) -> pd.Series:
    """Fetch annual World Bank indicator for Philippines. Returns Series indexed by 'YYYY'."""
    url = f'https://api.worldbank.org/v2/country/PHL/indicator/{indicator_id}'
    r = requests.get(
        url,
        params={'format': 'json', 'per_page': '100'},
        timeout=FETCH_TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
        raise ValueError(f'World Bank returned no data for {indicator_id!r}')
    return _parse_world_bank_response(payload)


def _fetch_psei() -> pd.Series:
    """Fetch PSEi monthly close prices from Yahoo Finance."""
    return _fetch_yahoo('^PSEi')
```

- [ ] **Step 6: Update `_fetch_all` in `fetcher.py` to build 9-column DataFrame**

Replace the existing `_fetch_all` function:

```python
def _fetch_all() -> pd.DataFrame:
    oil = _fetch_yahoo('BZ=F')
    usd = _fetch_yahoo('PHP=X')
    gas = _fetch_doe_prices(usd_php=usd)
    psei = _fetch_psei()

    cpi_annual = _fetch_world_bank('FP.CPI.TOTL.ZG')
    bsp_annual = _fetch_world_bank('FR.INR.LEND')
    rem_annual = _fetch_world_bank('BX.TRF.PWKR.CD.DT')

    base = pd.DataFrame({'oil_price': oil, 'usd_php': usd, 'gas_price': gas}).dropna()
    monthly_index = base.index.tolist()

    cpi = _forward_fill_annual(cpi_annual, monthly_index)
    bsp_rate = _forward_fill_annual(bsp_annual, monthly_index)
    remittances = (_forward_fill_annual(rem_annual, monthly_index) / 1e9).round(2)

    df = pd.DataFrame({
        'oil_price': oil,
        'usd_php': usd,
        'gas_price': gas,
        'psei': psei,
        'cpi': cpi,
        'bsp_rate': bsp_rate,
        'remittances': remittances,
    }).dropna()

    df.index.name = 'date'
    df = df.reset_index()
    df['demand_index'] = _compute_demand(df['date'].tolist())
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'oil_price', 'usd_php', 'demand_index', 'gas_price',
               'psei', 'cpi', 'bsp_rate', 'remittances']]
```

- [ ] **Step 7: Run the full test suite to confirm no regressions**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -q
```

Expected: 35 passed (32 existing + 3 new).

- [ ] **Step 8: Commit**

```
git add ph_economic_ai/fetcher.py ph_economic_ai/tests/test_fetcher.py
git commit -m "feat: add World Bank + PSEi fetchers and 9-column _fetch_all"
```

---

## Task 2: Adaptive `build_features`

**Files:**
- Modify: `ph_economic_ai/utils/preprocessing.py`

The current `build_features` hardcodes 4 feature columns. After this change it picks up whichever of the 4 new columns exist in the DataFrame — so it works with both `generate_dataset()` (old schema, tests) and `fetch_dataset()` (new 9-column schema, app).

- [ ] **Step 1: Replace `build_features` in `preprocessing.py`**

Replace the entire `build_features` function:

```python
def build_features(df: pd.DataFrame):
    """Add lagged gas price, drop first row (NaN), return X, y, col names, df."""
    df = df.copy()
    df['prev_gas_price'] = df['gas_price'].shift(1)
    df = df.dropna().reset_index(drop=True)
    base_cols = ['oil_price', 'usd_php', 'demand_index']
    extra_cols = ['psei', 'cpi', 'bsp_rate', 'remittances']
    available_extra = [c for c in extra_cols if c in df.columns]
    feature_cols = base_cols + available_extra + ['prev_gas_price']
    X = df[feature_cols].values.astype(float)
    y = df['gas_price'].values.astype(float)
    return X, y, feature_cols, df
```

- [ ] **Step 2: Run the full test suite**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -q
```

Expected: 35 passed. `generate_dataset()` only has the base 3 + prev_gas_price → 4 features. Tests still pass.

- [ ] **Step 3: Commit**

```
git add ph_economic_ai/utils/preprocessing.py
git commit -m "feat: build_features picks up new indicator columns when present"
```

---

## Task 3: HGB Model + New Functions

**Files:**
- Modify: `ph_economic_ai/model.py`
- Modify: `ph_economic_ai/tests/test_model.py`

`HistGradientBoostingRegressor` does not have `.estimators_`, so `predict` and `get_training_predictions` must change. The `forecast` function does a rolling 6-step prediction.

- [ ] **Step 1: Write new tests and update existing ones in `test_model.py`**

Replace the entire file content:

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.ensemble import HistGradientBoostingRegressor

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai.model import (
    train, predict, get_training_predictions, simulate_scenarios,
    get_feature_importances, cross_val_rmse, forecast,
)


def _trained():
    df = generate_dataset()
    X, y, feature_cols, df_feat = build_features(df)
    regressor = train(X, y)
    last = df_feat.iloc[-1]
    last_features = np.array([last[c] if c != 'prev_gas_price' else last['gas_price']
                               for c in feature_cols])
    return regressor, X, y, df_feat, last_features, float(last['gas_price']), feature_cols


def test_train_returns_fitted_model():
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    reg = train(X, y)
    assert isinstance(reg, HistGradientBoostingRegressor)
    assert hasattr(reg, 'feature_importances_')


def test_predict_returns_tuple():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    result = predict(reg, last_features)
    assert len(result) == 3
    predicted_price, confidence, pred_std = result
    assert 50.0 < predicted_price < 90.0
    assert 0.0 <= confidence <= 100.0
    assert pred_std >= 0.0


def test_get_training_predictions_shape():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    means, stds = get_training_predictions(reg, X)
    assert means.shape == (len(X),)
    assert stds.shape == (len(X),)
    assert (stds >= 0).all()


def test_simulate_scenarios_keys():
    reg, X, y, df_feat, last_features, baseline, _ = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert set(scenarios.keys()) == {'oil_shock', 'usd_shock', 'demand_drop'}


def test_simulate_oil_shock_raises_price():
    reg, X, y, df_feat, last_features, baseline, _ = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['oil_shock'] > 0, 'Higher oil should raise price'


def test_simulate_demand_drop_lowers_price():
    reg, X, y, df_feat, last_features, baseline, _ = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['demand_drop'] < 0, 'Lower demand should reduce price'


def test_feature_importances_shape():
    reg, X, y, df_feat, last_features, _, feature_cols = _trained()
    importances = get_feature_importances(reg, feature_cols)
    assert len(importances) == len(feature_cols)
    assert abs(sum(importances.values()) - 1.0) < 1e-6


def test_cross_val_rmse_positive():
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    rmse = cross_val_rmse(X, y)
    assert rmse > 0.0


def test_forecast_returns_n_months():
    reg, X, y, df_feat, last_features, _, _ = _trained()
    prices = forecast(reg, last_features, n_months=6)
    assert len(prices) == 6
    assert all(30.0 < p < 150.0 for p in prices)
```

- [ ] **Step 2: Run tests to verify new ones fail**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_model.py -v
```

Expected: `test_feature_importances_shape`, `test_cross_val_rmse_positive`, `test_forecast_returns_n_months` FAIL with ImportError. `test_train_returns_fitted_model` FAIL (RF has no `feature_importances_` like HGB does, and isinstance check fails).

- [ ] **Step 3: Replace `model.py` entirely**

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score


def train(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    """Train on first 80% of rows (time-ordered). Returns fitted regressor."""
    split = int(len(X) * 0.8)
    regressor = HistGradientBoostingRegressor(random_state=42)
    regressor.fit(X[:split], y[:split])
    return regressor


def predict(regressor: HistGradientBoostingRegressor, last_features: np.ndarray) -> tuple:
    """
    Predict next price from a 1-D feature vector.
    Returns (predicted_price, confidence_0_100, pred_std).
    pred_std is 0.0 — use cv_rmse from cross_val_rmse() for uncertainty bands.
    """
    X = last_features.reshape(1, -1)
    predicted_price = float(regressor.predict(X)[0])
    return predicted_price, 90.0, 0.0


def get_training_predictions(regressor: HistGradientBoostingRegressor, X: np.ndarray) -> tuple:
    """Return (means, stds) for all rows. stds are zeros — HGB has no per-tree variance."""
    means = regressor.predict(X)
    stds = np.zeros(len(X))
    return means, stds


def get_feature_importances(model: HistGradientBoostingRegressor,
                            feature_names: list) -> dict:
    """Return feature name → normalized importance (0–1), sorted descending."""
    importances = model.feature_importances_
    total = importances.sum()
    normalized = importances / total if total > 0 else importances
    return dict(sorted(zip(feature_names, normalized), key=lambda x: x[1], reverse=True))


def cross_val_rmse(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """5-fold CV on a fresh HGB. Returns mean RMSE (positive, PHP/liter)."""
    model = HistGradientBoostingRegressor(random_state=42)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    return float(-scores.mean())


def forecast(regressor: HistGradientBoostingRegressor, last_features: np.ndarray,
             n_months: int = 6) -> np.ndarray:
    """
    Roll n_months forward from last_features using flat projection.
    The last element of last_features is prev_gas_price — updated each step.
    Returns array of shape (n_months,) with predicted prices.
    """
    prices = []
    features = last_features.copy()
    for _ in range(n_months):
        price, _, _ = predict(regressor, features)
        prices.append(price)
        features[-1] = price  # update prev_gas_price for next step
    return np.array(prices)


def simulate_scenarios(regressor: HistGradientBoostingRegressor,
                       last_features: np.ndarray, baseline_price: float) -> dict:
    """
    Perturb last_features for 3 scenarios and return price deltas vs baseline.
    Feature layout: [oil_price(0), usd_php(1), demand_index(2), ..., prev_gas_price(-1)]
    """
    def _delta(features):
        p, _, _ = predict(regressor, features)
        return p - baseline_price

    oil_f = last_features.copy(); oil_f[0] *= 1.05
    usd_f = last_features.copy(); usd_f[1] *= 1.02
    dem_f = last_features.copy(); dem_f[2] = max(0.0, dem_f[2] - 10.0)

    return {
        'oil_shock': _delta(oil_f),
        'usd_shock': _delta(usd_f),
        'demand_drop': _delta(dem_f),
    }
```

- [ ] **Step 4: Run all model tests**

```
.venv\Scripts\pytest ph_economic_ai/tests/test_model.py -v
```

Expected: 10 PASSED.

- [ ] **Step 5: Run the full test suite**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -q
```

Expected: 35 passed.

- [ ] **Step 6: Commit**

```
git add ph_economic_ai/model.py ph_economic_ai/tests/test_model.py
git commit -m "feat: upgrade model to HistGradientBoosting; add feature_importances, cv_rmse, forecast"
```

---

## Task 4: Wire `cv_rmse` Through the App

**Files:**
- Modify: `ph_economic_ai/main.py`
- Modify: `ph_economic_ai/ui/main_window.py`

`main.py` computes `cv_rmse` after training and passes it to `MainWindow`. `MainWindow` adds a `_make_last_features()` helper (replaces the inline array), captures `feature_cols` from `build_features`, and adds `cv_rmse`, `feature_importances`, `forecast_prices`, and `df_raw` to the result dict for the Dashboard.

- [ ] **Step 1: Replace `main.py`**

```python
import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

from ph_economic_ai.data import fetch_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        df, data_source = fetch_dataset()
    except RuntimeError as e:
        QMessageBox.critical(None, 'Data Error', str(e))
        sys.exit(1)

    X, y, feature_cols, _ = build_features(df)
    regressor = ml.train(X, y)
    cv_rmse = ml.cross_val_rmse(X, y)

    window = MainWindow(df=df, regressor=regressor, data_source=data_source, cv_rmse=cv_rmse)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Replace `MainWindow.__init__` in `main_window.py`**

Change the constructor signature and body. Replace lines 26–71 of `main_window.py` (the `MainWindow.__init__` method):

```python
class MainWindow(QMainWindow):
    def __init__(self, df, regressor, data_source: str = 'Live Data',
                 cv_rmse: float = 0.0, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._cv_rmse = cv_rmse
        self._oil_shock_active = False

        X, y, self._feature_cols, self._df_feat = build_features(df)
        self._X = X
        self._last_features = self._make_last_features()

        self.setWindowTitle('Philippine Economic Pressure AI')
        self.setMinimumSize(1100, 680)
        self.setStyleSheet('background:#FFFFFF;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = SidebarWidget(data_source=data_source)
        self._sidebar.page_changed.connect(self._on_page_changed)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._dashboard = DashboardPage()
        self._dashboard.recalculate_requested.connect(self._on_recalculate)
        self._dashboard.oil_shock_requested.connect(self._on_oil_shock)
        self._pressure_page = PressureGaugePage()
        self._agent_page = AgentGraphPage()
        self._settings_page = _SettingsPage()

        for page in (self._dashboard, self._pressure_page,
                     self._agent_page, self._settings_page):
            self._stack.addWidget(page)

        root.addWidget(self._stack, stretch=1)

        self._refresh()
```

- [ ] **Step 3: Add `_make_last_features` method to `MainWindow`**

Add after the `__init__` method (before `_on_page_changed`):

```python
    def _make_last_features(self) -> np.ndarray:
        """Build feature vector for next-step prediction from last row of df_feat."""
        last = self._df_feat.iloc[-1]
        values = []
        for col in self._feature_cols:
            if col == 'prev_gas_price':
                values.append(float(last['gas_price']))
            else:
                values.append(float(last[col]))
        return np.array(values)
```

- [ ] **Step 4: Update `_on_recalculate` to use `_make_last_features`**

Replace the existing `_on_recalculate` method:

```python
    def _on_recalculate(self):
        self._oil_shock_active = False
        self._last_features = self._make_last_features()
        self._refresh()
```

- [ ] **Step 5: Update `_build_result` to include new keys**

In `_build_result`, add these four entries to the returned dict (after `'train_stds'`):

```python
        feature_importances = ml.get_feature_importances(
            self._regressor, self._feature_cols
        )
        forecast_prices = ml.forecast(self._regressor, self._last_features)

        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'trend': trend,
            'confidence': confidence,
            'pressure_index': pressure_index,
            'pressure_band': band,
            'oil_delta': oil_delta,
            'usd_delta': usd_delta,
            'demand_norm': demand_norm,
            'explanation': explanation,
            'scenarios': scenarios,
            'pred_std': pred_std,
            'train_means': train_means,
            'train_stds': train_stds,
            'df': self._df_feat,
            'cv_rmse': self._cv_rmse,
            'feature_importances': feature_importances,
            'forecast_prices': forecast_prices,
            'df_raw': self._df,
        }
```

- [ ] **Step 6: Run the app to verify it starts without error**

```
.venv\Scripts\python -m ph_economic_ai.main
```

Expected: app launches, Dashboard Overview tab visible, no traceback.

- [ ] **Step 7: Run the full test suite**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -q
```

Expected: 35 passed.

- [ ] **Step 8: Commit**

```
git add ph_economic_ai/main.py ph_economic_ai/ui/main_window.py
git commit -m "feat: wire cv_rmse through app; add feature_importances and forecast to result dict"
```

---

## Task 5: Dashboard Indicators Tab

**Files:**
- Modify: `ph_economic_ai/ui/dashboard.py`

Wrap the existing `_build()` content into `_build_overview_tab()`. Add a `QTabWidget` as the top-level container. Add the "Indicators" tab with 7 matplotlib line charts in a 2-column scrollable grid.

- [ ] **Step 1: Update imports at the top of `dashboard.py`**

Replace the existing PyQt6 import line:

```python
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QPushButton, QScrollArea, QSizePolicy,
                              QProgressBar, QTabWidget, QGridLayout)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
```

- [ ] **Step 2: Add `QTabWidget` to `DashboardPage.__init__` and rename `_build` → `_build_overview_tab`**

Replace the `DashboardPage` class `__init__` and `_build` methods (lines 407–483 of `dashboard.py`):

```python
class DashboardPage(QWidget):
    recalculate_requested = pyqtSignal()
    oil_shock_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#FAFAFA;')

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet(
            'QTabBar::tab { padding: 8px 22px; font-size: 12px; color: #888888; }'
            'QTabBar::tab:selected { color: #111111; font-weight: 700; '
            '  border-bottom: 2px solid #4A90E2; }'
            'QTabWidget::pane { border: none; }'
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._tabs)

        self._build_overview_tab()
        self._build_indicators_tab()
```

- [ ] **Step 2: Rename `_build` to `_build_overview_tab` and wrap its content in a new QWidget**

Replace the existing `_build` method with `_build_overview_tab`. The only change is: instead of `main = QHBoxLayout(self)`, create a widget and use that as the layout parent, then register it as a tab.

Replace the method body:

```python
    def _build_overview_tab(self):
        overview = QWidget()
        overview.setStyleSheet('background:#FAFAFA;')
        main = QHBoxLayout(overview)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # ── Center column ─────────────────────────────────────────────────────
        center = QWidget()
        center.setStyleSheet('background:#FAFAFA;')
        clyt = QVBoxLayout(center)
        clyt.setContentsMargins(22, 20, 22, 20)
        clyt.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        pg_title = QLabel('Gasoline Price Dashboard')
        pg_title.setStyleSheet('font-size:18px; font-weight:700; color:#111111;')
        pg_sub = QLabel('Philippines · Live data · Trained on startup')
        pg_sub.setStyleSheet('font-size:11px; color:#AAAAAA;')
        title_col.addWidget(pg_title)
        title_col.addWidget(pg_sub)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self._recalc_btn = QPushButton('↺  Recalculate')
        self._recalc_btn.setStyleSheet(
            'padding:7px 14px; font-size:11px; font-weight:600; border-radius:8px;'
            'border:1px solid #4A90E2; background:#FFFFFF; color:#4A90E2;'
        )
        self._recalc_btn.clicked.connect(self.recalculate_requested)
        self._shock_btn = QPushButton('⚡  Oil Shock +10%')
        self._shock_btn.setStyleSheet(
            'padding:7px 14px; font-size:11px; font-weight:600; border-radius:8px;'
            'border:1px solid #E07A4A; background:#FFFFFF; color:#E07A4A;'
        )
        self._shock_btn.clicked.connect(self.oil_shock_requested)
        btn_row.addWidget(self._recalc_btn)
        btn_row.addWidget(self._shock_btn)

        hdr.addLayout(title_col)
        hdr.addStretch()
        hdr.addLayout(btn_row)
        clyt.addLayout(hdr)

        self._chart = PriceChart()
        clyt.addWidget(self._chart)

        mini_row = QHBoxLayout()
        mini_row.setSpacing(10)
        self._price_card  = _MiniCard('Predicted Price')
        self._trend_card  = _MiniCard('Trend Direction')
        self._index_card  = _MiniCard('Pressure Index')
        for card in (self._price_card, self._trend_card, self._index_card):
            mini_row.addWidget(card)
        clyt.addLayout(mini_row)

        self._sim_panel = SimulationPanel()
        clyt.addWidget(self._sim_panel)

        self._right = _RightPanel()

        main.addWidget(center, stretch=1)
        main.addWidget(self._right)

        self._tabs.addTab(overview, 'Overview')
```

- [ ] **Step 3: Add `_build_indicators_tab` method**

Add after `_build_overview_tab`:

```python
    def _build_indicators_tab(self):
        _INDICATORS = [
            ('oil_price',   'Oil Price',          'USD/bbl',    '#4A90E2'),
            ('usd_php',     'USD/PHP Rate',        'PHP/USD',    '#E0A84A'),
            ('demand_index','Demand Index',        'Index',      '#27AE60'),
            ('psei',        'PSEi',                'Points',     '#9B59B6'),
            ('cpi',         'CPI Inflation',       '% p.a.',     '#E74C3C'),
            ('bsp_rate',    'BSP Lending Rate',    '% p.a.',     '#1ABC9C'),
            ('remittances', 'OFW Remittances',     'USD bn',     '#E07A4A'),
        ]

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('QScrollArea { border:none; } QScrollBar { width:0px; }')

        container = QWidget()
        container.setStyleSheet('background:#FAFAFA;')
        grid = QGridLayout(container)
        grid.setContentsMargins(22, 20, 22, 20)
        grid.setSpacing(14)

        self._indicator_axes = {}
        self._indicator_canvases = {}

        for i, (key, title, ylabel, color) in enumerate(_INDICATORS):
            fig = Figure(figsize=(4, 2), dpi=96)
            fig.patch.set_facecolor('#FFFFFF')
            ax = fig.add_subplot(111)
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=10, fontweight='bold', color='#111111', pad=4)
            ax.set_ylabel(ylabel, fontsize=8, color='#888888')
            ax.tick_params(labelsize=7, colors='#888888')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#EEEEEE')
            ax.grid(axis='y', color='#F5F5F5', linewidth=0.5)
            fig.tight_layout(pad=1.2)

            canvas = FigureCanvasQTAgg(fig)
            canvas.setFixedHeight(200)

            row_i, col_i = divmod(i, 2)
            grid.addWidget(canvas, row_i, col_i)

            self._indicator_axes[key] = ax
            self._indicator_canvases[key] = canvas

        scroll.setWidget(container)
        self._tabs.addTab(scroll, 'Indicators')
```

- [ ] **Step 4: Add `_update_indicators` method and call it from `refresh`**

Add `_update_indicators` after `_build_indicators_tab`:

```python
    def _update_indicators(self, result: dict):
        df_raw = result.get('df_raw')
        if df_raw is None:
            return

        _LABEL_MAP = {
            'oil_price':    ('Oil Price',       'USD/bbl',  '#4A90E2'),
            'usd_php':      ('USD/PHP Rate',    'PHP/USD',  '#E0A84A'),
            'demand_index': ('Demand Index',    'Index',    '#27AE60'),
            'psei':         ('PSEi',            'Points',   '#9B59B6'),
            'cpi':          ('CPI Inflation',   '% p.a.',   '#E74C3C'),
            'bsp_rate':     ('BSP Lending Rate','% p.a.',   '#1ABC9C'),
            'remittances':  ('OFW Remittances', 'USD bn',   '#E07A4A'),
        }

        dates = df_raw['date'].tolist()
        tick_pos = list(range(0, len(dates), max(1, len(dates) // 5)))
        tick_labels = [dates[i] for i in tick_pos]

        for key, ax in self._indicator_axes.items():
            if key not in df_raw.columns:
                continue
            title, ylabel, color = _LABEL_MAP[key]
            values = df_raw[key].values.astype(float)
            x = list(range(len(values)))

            ax.clear()
            ax.set_facecolor('#FFFFFF')
            ax.set_title(title, fontsize=10, fontweight='bold', color='#111111', pad=4)
            ax.set_ylabel(ylabel, fontsize=8, color='#888888')
            ax.plot(x, values, color=color, linewidth=1.5)
            ax.fill_between(x, float(values.min()), values, color=color, alpha=0.08)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=30, fontsize=7)
            ax.tick_params(labelsize=7, colors='#888888')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#EEEEEE')
            ax.grid(axis='y', color='#F5F5F5', linewidth=0.5)

            canvas = self._indicator_canvases[key]
            canvas.figure.tight_layout(pad=1.2)
            canvas.draw()
```

In the existing `refresh` method, add a call to `_update_indicators` at the end:

```python
    def refresh(self, result: dict):
        # ... existing chart, mini-cards, sim_panel, right refresh calls ...
        self._right.refresh(result)
        self._update_indicators(result)
```

- [ ] **Step 5: Run the app and check both tabs**

```
.venv\Scripts\python -m ph_economic_ai.main
```

Expected:
- Dashboard opens on "Overview" tab — all existing content visible and unchanged.
- Click "Indicators" tab — 7 charts render (or are blank on first load if cache is present with old schema; that's OK — Task 1 adds the new columns for fresh fetches).

- [ ] **Step 6: Commit**

```
git add ph_economic_ai/ui/dashboard.py
git commit -m "feat: add QTabWidget to Dashboard — Overview and Indicators tabs"
```

---

## Task 6: Dashboard Model Insights Tab

**Files:**
- Modify: `ph_economic_ai/ui/dashboard.py`

Add the third tab with a feature importance horizontal bar chart and a 6-month forecast line with ±CV-RMSE shaded band.

- [ ] **Step 1: Add `_build_model_insights_tab` method to `DashboardPage`**

Add after `_update_indicators` and add the call in `__init__`:

First, in `__init__`, add the third tab call after `self._build_indicators_tab()`:

```python
        self._build_model_insights_tab()
```

Then add the method:

```python
    def _build_model_insights_tab(self):
        container = QWidget()
        container.setStyleSheet('background:#FAFAFA;')
        lyt = QVBoxLayout(container)
        lyt.setContentsMargins(22, 20, 22, 20)
        lyt.setSpacing(16)

        # ── Feature importance card ───────────────────────────────────────────
        fi_frame = QFrame()
        fi_frame.setStyleSheet(
            'background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;'
        )
        fi_inner = QVBoxLayout(fi_frame)
        fi_inner.setContentsMargins(16, 14, 16, 14)
        fi_inner.setSpacing(8)

        fi_title = QLabel('Feature Importance')
        fi_title.setStyleSheet('font-size:13px; font-weight:700; color:#111111; border:none;')
        fi_inner.addWidget(fi_title)

        fi_sub = QLabel('Which signals drive the gas price model most?  Blue = original · Purple = new')
        fi_sub.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        fi_inner.addWidget(fi_sub)

        self._fi_fig = Figure(figsize=(6, 2.8), dpi=96)
        self._fi_fig.patch.set_facecolor('#FFFFFF')
        self._fi_ax = self._fi_fig.add_subplot(111)
        self._fi_canvas = FigureCanvasQTAgg(self._fi_fig)
        self._fi_canvas.setFixedHeight(220)
        fi_inner.addWidget(self._fi_canvas)
        lyt.addWidget(fi_frame)

        # ── Forecast card ─────────────────────────────────────────────────────
        fc_frame = QFrame()
        fc_frame.setStyleSheet(
            'background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;'
        )
        fc_inner = QVBoxLayout(fc_frame)
        fc_inner.setContentsMargins(16, 14, 16, 14)
        fc_inner.setSpacing(8)

        fc_title = QLabel('6-Month Gas Price Forecast  (indicative)')
        fc_title.setStyleSheet('font-size:13px; font-weight:700; color:#111111; border:none;')
        fc_inner.addWidget(fc_title)

        self._fc_fig = Figure(figsize=(6, 2.8), dpi=96)
        self._fc_fig.patch.set_facecolor('#FFFFFF')
        self._fc_ax = self._fc_fig.add_subplot(111)
        self._fc_canvas = FigureCanvasQTAgg(self._fc_fig)
        self._fc_canvas.setFixedHeight(220)
        fc_inner.addWidget(self._fc_canvas)

        self._rmse_label = QLabel()
        self._rmse_label.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        fc_inner.addWidget(self._rmse_label)

        lyt.addWidget(fc_frame)
        lyt.addStretch()

        self._tabs.addTab(container, 'Model Insights')
```

- [ ] **Step 2: Add `_update_model_insights` method**

Add after `_build_model_insights_tab`:

```python
    def _update_model_insights(self, result: dict):
        _ORIGINAL = {'oil_price', 'usd_php', 'demand_index', 'prev_gas_price'}

        # ── Feature importance ────────────────────────────────────────────────
        fi = result.get('feature_importances', {})
        if fi:
            ax = self._fi_ax
            ax.clear()
            ax.set_facecolor('#FFFFFF')

            names = list(fi.keys())
            values = list(fi.values())
            colors = ['#4A90E2' if n in _ORIGINAL else '#9B59B6' for n in names]

            bars = ax.barh(names, values, color=colors, height=0.55)
            ax.set_xlabel('Normalized importance', fontsize=8, color='#888888')
            ax.tick_params(labelsize=8, colors='#333333')
            ax.spines[['top', 'right', 'bottom']].set_visible(False)
            ax.spines['left'].set_color('#EEEEEE')
            ax.set_xlim(0, max(values) * 1.18 if values else 1)
            for bar, val in zip(bars, values):
                ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                        f'{val:.3f}', va='center', fontsize=7, color='#888888')

            self._fi_fig.tight_layout(pad=1.2)
            self._fi_canvas.draw()

        # ── Forecast ──────────────────────────────────────────────────────────
        forecast_prices = result.get('forecast_prices')
        cv_rmse = result.get('cv_rmse', 0.0)
        df = result.get('df')
        if forecast_prices is not None and df is not None:
            ax = self._fc_ax
            ax.clear()
            ax.set_facecolor('#FFFFFF')

            hist_dates = df['date'].tolist()[-12:]
            hist_prices = df['gas_price'].values[-12:].astype(float)
            n_hist = len(hist_dates)

            ax.plot(range(n_hist), hist_prices, color='#AAAAAA', linewidth=1.5,
                    label='Historical (last 12 mo)')

            fc_x = list(range(n_hist - 1, n_hist + len(forecast_prices)))
            fc_y = np.concatenate([[hist_prices[-1]], forecast_prices])
            ax.plot(fc_x, fc_y, color='#4A90E2', linewidth=2, label='Forecast')
            ax.fill_between(fc_x, fc_y - cv_rmse, fc_y + cv_rmse,
                            color='#4A90E2', alpha=0.15,
                            label=f'±₱{cv_rmse:.2f} CV-RMSE')

            tick_pos = list(range(0, n_hist, max(1, n_hist // 4)))
            ax.set_xticks(tick_pos)
            ax.set_xticklabels([hist_dates[i] for i in tick_pos], rotation=30, fontsize=7)
            ax.set_ylabel('₱ / liter', fontsize=8, color='#888888')
            ax.tick_params(colors='#888888', labelsize=7)
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#EEEEEE')
            ax.grid(axis='y', color='#F5F5F5', linewidth=0.5)
            ax.legend(fontsize=7, framealpha=0, loc='upper left')

            self._fc_fig.tight_layout(pad=1.2)
            self._fc_canvas.draw()

            self._rmse_label.setText(
                f'CV-RMSE: ₱{cv_rmse:.2f}/liter  ·  '
                'Forecast uses flat projection of latest known input values'
            )
```

- [ ] **Step 3: Call `_update_model_insights` from `refresh`**

In `DashboardPage.refresh`, add after `self._update_indicators(result)`:

```python
        self._update_model_insights(result)
```

- [ ] **Step 4: Run the app and verify the Model Insights tab**

```
.venv\Scripts\python -m ph_economic_ai.main
```

Expected:
- "Model Insights" tab appears.
- Feature Importance chart shows 4–8 bars (depends on how many new columns are in cached data).
- Forecast chart shows last 12 months of actual prices + 6-month forward line with shaded band.
- RMSE label reads something like "CV-RMSE: ₱2.45/liter · Forecast uses flat projection..."

- [ ] **Step 5: Run the full test suite**

```
.venv\Scripts\pytest ph_economic_ai/tests/ -q
```

Expected: 35 passed.

- [ ] **Step 6: Commit**

```
git add ph_economic_ai/ui/dashboard.py
git commit -m "feat: add Model Insights tab — feature importance + 6-month forecast with RMSE band"
```

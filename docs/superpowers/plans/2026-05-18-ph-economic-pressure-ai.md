# Philippine Economic Pressure AI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully offline PyQt6 desktop app that predicts Philippine gasoline prices, computes an Economic Pressure Index, and displays AI-generated advisory output.

**Architecture:** Greenfield `ph_economic_ai/` subfolder in the project root. Pure-Python non-UI modules (`data`, `model`, `preprocessing`, `explanation`) are tested with pytest. UI modules (`ui/`) are composed in `MainWindow` and verified by running the app. Data flows from `data.py → preprocessing → model → explanation → UI` on startup; `MainWindow` reruns steps 3-5 on Recalculate/Oil Shock without retraining.

**Tech Stack:** PyQt6 6.10.0, matplotlib 3.8.4 (embedded chart), scikit-learn 1.7.2 (RandomForestRegressor), numpy 1.26.4, pandas 1.5.3 — all pre-installed in `.venv`.

---

## File Map

| File | Responsibility |
|------|---------------|
| `ph_economic_ai/main.py` | `QApplication` entry, runs startup pipeline, shows `MainWindow` |
| `ph_economic_ai/data.py` | Generate 120-row synthetic DataFrame |
| `ph_economic_ai/model.py` | Train, predict, get training preds, simulate scenarios |
| `ph_economic_ai/utils/__init__.py` | Empty |
| `ph_economic_ai/utils/preprocessing.py` | Build features, compute pressure index, get band label |
| `ph_economic_ai/utils/explanation.py` | Rule-based driver analysis + advisory text |
| `ph_economic_ai/ui/__init__.py` | Empty |
| `ph_economic_ai/ui/main_window.py` | `QMainWindow`: sidebar + stacked pages, signal wiring, result dict assembly |
| `ph_economic_ai/ui/sidebar.py` | Left nav sidebar, emits `page_changed(int)` |
| `ph_economic_ai/ui/charts.py` | `PriceChart(FigureCanvasQTAgg)`: actual/predicted/confidence chart |
| `ph_economic_ai/ui/dashboard.py` | `DashboardPage`: chart + mini-cards + simulation panel + right panel |
| `ph_economic_ai/ui/pressure.py` | `PressureGauge(QWidget)` + `PressureGaugePage(QWidget)` |
| `ph_economic_ai/ui/agent_graph.py` | `AgentGraphPage`: QGraphicsView with 5 nodes + edges |
| `ph_economic_ai/tests/__init__.py` | Empty |
| `ph_economic_ai/tests/test_data.py` | Tests for `data.py` |
| `ph_economic_ai/tests/test_preprocessing.py` | Tests for `preprocessing.py` |
| `ph_economic_ai/tests/test_model.py` | Tests for `model.py` |
| `ph_economic_ai/tests/test_explanation.py` | Tests for `explanation.py` |

---

## Task 1: Project Scaffold

**Files:**
- Create: `ph_economic_ai/__init__.py`
- Create: `ph_economic_ai/ui/__init__.py`
- Create: `ph_economic_ai/utils/__init__.py`
- Create: `ph_economic_ai/tests/__init__.py`

- [ ] **Step 1: Create directory structure**

```powershell
New-Item -ItemType Directory -Force ph_economic_ai/ui
New-Item -ItemType Directory -Force ph_economic_ai/utils
New-Item -ItemType Directory -Force ph_economic_ai/tests
"" | Out-File -Encoding utf8 ph_economic_ai/__init__.py
"" | Out-File -Encoding utf8 ph_economic_ai/ui/__init__.py
"" | Out-File -Encoding utf8 ph_economic_ai/utils/__init__.py
"" | Out-File -Encoding utf8 ph_economic_ai/tests/__init__.py
```

Run from: `C:\Users\user\PycharmProjects\PythonProject`

- [ ] **Step 2: Verify structure**

```powershell
Get-ChildItem ph_economic_ai -Recurse | Select-Object FullName
```

Expected: `ph_economic_ai/`, `ph_economic_ai/ui/`, `ph_economic_ai/utils/`, `ph_economic_ai/tests/` all present with `__init__.py`.

- [ ] **Step 3: Commit**

```bash
git add ph_economic_ai/
git commit -m "feat: scaffold ph_economic_ai package structure"
```

---

## Task 2: Data Generation (`data.py`)

**Files:**
- Create: `ph_economic_ai/data.py`
- Create: `ph_economic_ai/tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

`ph_economic_ai/tests/test_data.py`:
```python
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset


def test_shape():
    df = generate_dataset()
    assert df.shape == (120, 5)


def test_columns():
    df = generate_dataset()
    assert list(df.columns) == ['date', 'oil_price', 'usd_php', 'demand_index', 'gas_price']


def test_no_nulls():
    df = generate_dataset()
    assert df.isnull().sum().sum() == 0


def test_ranges():
    df = generate_dataset()
    assert df['oil_price'].between(75, 105).all()
    assert df['usd_php'].between(54, 62).all()
    assert df['demand_index'].between(55, 90).all()
    assert df['gas_price'].between(62, 82).all()


def test_reproducible():
    df1 = generate_dataset(seed=42)
    df2 = generate_dataset(seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds():
    df1 = generate_dataset(seed=1)
    df2 = generate_dataset(seed=2)
    assert not df1['oil_price'].equals(df2['oil_price'])
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'ph_economic_ai.data'`

- [ ] **Step 3: Implement `data.py`**

`ph_economic_ai/data.py`:
```python
import numpy as np
import pandas as pd


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 120

    dates = pd.date_range('2024-01', periods=n, freq='MS').strftime('%Y-%m').tolist()

    # Oil price random walk with slight upward drift, clipped 75–105
    oil = np.empty(n)
    oil[0] = 85.0
    for i in range(1, n):
        oil[i] = oil[i - 1] + rng.normal(0.2, 1.8)
    oil = np.clip(oil, 75.0, 105.0)

    # USD/PHP correlated walk, clipped 54–62
    usd = np.empty(n)
    usd[0] = 56.5
    for i in range(1, n):
        drift = (oil[i] - oil[i - 1]) * 0.04
        usd[i] = usd[i - 1] + drift + rng.normal(0.0, 0.25)
    usd = np.clip(usd, 54.0, 62.0)

    # Demand index: seasonal sine + noise, clipped 55–90
    t = np.arange(n)
    demand = 72.0 + 9.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0.0, 2.5, n)
    demand = np.clip(demand, 55.0, 90.0)

    # Gas price derived from inputs + small noise
    gas = (
        62.0
        + (oil - 75.0) * 0.38
        + (usd - 54.0) * 0.75
        + (demand - 55.0) * 0.04
        + rng.normal(0.0, 0.4, n)
    )
    gas = np.clip(gas, 62.0, 82.0)

    return pd.DataFrame({
        'date': dates,
        'oil_price': oil.round(2),
        'usd_php': usd.round(2),
        'demand_index': demand.round(1),
        'gas_price': gas.round(2),
    })
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_data.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/data.py ph_economic_ai/tests/test_data.py
git commit -m "feat: add synthetic dataset generator with tests"
```

---

## Task 3: Preprocessing (`utils/preprocessing.py`)

**Files:**
- Create: `ph_economic_ai/utils/preprocessing.py`
- Create: `ph_economic_ai/tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing tests**

`ph_economic_ai/tests/test_preprocessing.py`:
```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features, compute_index, pressure_band


def _df():
    return generate_dataset()


def test_build_features_shape():
    df = _df()
    X, y, cols, df_out = build_features(df)
    assert X.shape == (119, 4)
    assert y.shape == (119,)
    assert cols == ['oil_price', 'usd_php', 'demand_index', 'prev_gas_price']


def test_build_features_no_nan():
    df = _df()
    X, y, _, _ = build_features(df)
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_compute_index_range():
    df = _df()
    last = df.iloc[-1]
    index, *_ = compute_index(last['oil_price'], last['usd_php'], last['demand_index'], df)
    assert 0.0 <= index <= 100.0


def test_compute_index_returns_deltas():
    df = _df()
    last = df.iloc[-1]
    index, oil_delta, usd_delta, demand_norm = compute_index(
        last['oil_price'], last['usd_php'], last['demand_index'], df
    )
    assert isinstance(oil_delta, float)
    assert isinstance(usd_delta, float)
    assert 0.0 <= demand_norm <= 1.0


def test_compute_index_high_oil_raises_pressure():
    df = _df()
    base_idx, *_ = compute_index(df['oil_price'].mean(), df['usd_php'].mean(), 70.0, df)
    high_idx, *_ = compute_index(df['oil_price'].max(), df['usd_php'].mean(), 70.0, df)
    assert high_idx > base_idx


def test_pressure_band_labels():
    assert pressure_band(15.0) == 'Stable'
    assert pressure_band(45.0) == 'Rising'
    assert pressure_band(70.0) == 'High'
    assert pressure_band(90.0) == 'Critical'
    assert pressure_band(0.0) == 'Stable'
    assert pressure_band(100.0) == 'Critical'
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_preprocessing.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `preprocessing.py`**

`ph_economic_ai/utils/preprocessing.py`:
```python
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame):
    """Add lagged gas price, drop first row (NaN), return X, y, col names, df."""
    df = df.copy()
    df['prev_gas_price'] = df['gas_price'].shift(1)
    df = df.dropna().reset_index(drop=True)
    feature_cols = ['oil_price', 'usd_php', 'demand_index', 'prev_gas_price']
    X = df[feature_cols].values.astype(float)
    y = df['gas_price'].values.astype(float)
    return X, y, feature_cols, df


def compute_index(current_oil: float, current_usd: float, current_demand: float,
                  df: pd.DataFrame) -> tuple:
    """Return (pressure_index 0-100, oil_delta, usd_delta, demand_norm)."""
    oil_mean, oil_std = df['oil_price'].mean(), df['oil_price'].std()
    usd_mean, usd_std = df['usd_php'].mean(), df['usd_php'].std()

    oil_deltas = (df['oil_price'] - oil_mean) / oil_std
    usd_deltas = (df['usd_php'] - usd_mean) / usd_std
    demand_norms = df['demand_index'] / 100.0
    raw_series = oil_deltas * 0.50 + usd_deltas * 0.30 + demand_norms * 0.20
    raw_min, raw_max = float(raw_series.min()), float(raw_series.max())

    oil_delta = float((current_oil - oil_mean) / oil_std)
    usd_delta = float((current_usd - usd_mean) / usd_std)
    demand_norm = float(current_demand / 100.0)
    raw = oil_delta * 0.50 + usd_delta * 0.30 + demand_norm * 0.20

    if raw_max > raw_min:
        normalized = (raw - raw_min) / (raw_max - raw_min) * 100.0
    else:
        normalized = 50.0

    return float(np.clip(normalized, 0.0, 100.0)), oil_delta, usd_delta, demand_norm


def pressure_band(index: float) -> str:
    if index <= 30:
        return 'Stable'
    elif index <= 60:
        return 'Rising'
    elif index <= 80:
        return 'High'
    return 'Critical'
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_preprocessing.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/utils/preprocessing.py ph_economic_ai/tests/test_preprocessing.py
git commit -m "feat: add preprocessing — feature builder and pressure index"
```

---

## Task 4: ML Model (`model.py`)

**Files:**
- Create: `ph_economic_ai/model.py`
- Create: `ph_economic_ai/tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

`ph_economic_ai/tests/test_model.py`:
```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai.model import train, predict, get_training_predictions, simulate_scenarios


def _trained():
    df = generate_dataset()
    X, y, _, df_feat = build_features(df)
    regressor = train(X, y)
    last = df_feat.iloc[-1]
    last_features = np.array([
        last['oil_price'], last['usd_php'], last['demand_index'], last['gas_price']
    ])
    return regressor, X, y, df_feat, last_features, last['gas_price']


def test_train_returns_fitted_model():
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    reg = train(X, y)
    assert hasattr(reg, 'estimators_')
    assert len(reg.estimators_) == 100


def test_predict_returns_tuple():
    reg, X, y, df_feat, last_features, _ = _trained()
    result = predict(reg, last_features)
    assert len(result) == 3
    predicted_price, confidence, pred_std = result
    assert 50.0 < predicted_price < 90.0
    assert 0.0 <= confidence <= 100.0
    assert pred_std >= 0.0


def test_get_training_predictions_shape():
    reg, X, y, df_feat, last_features, _ = _trained()
    means, stds = get_training_predictions(reg, X)
    assert means.shape == (len(X),)
    assert stds.shape == (len(X),)
    assert (stds >= 0).all()


def test_simulate_scenarios_keys():
    reg, X, y, df_feat, last_features, baseline = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert set(scenarios.keys()) == {'oil_shock', 'usd_shock', 'demand_drop'}


def test_simulate_oil_shock_raises_price():
    reg, X, y, df_feat, last_features, baseline = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['oil_shock'] > 0, "Higher oil should raise price"


def test_simulate_demand_drop_lowers_price():
    reg, X, y, df_feat, last_features, baseline = _trained()
    scenarios = simulate_scenarios(reg, last_features, baseline)
    assert scenarios['demand_drop'] < 0, "Lower demand should reduce price"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_model.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement `model.py`**

`ph_economic_ai/model.py`:
```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Train on first 80% of rows (time-ordered). Returns fitted regressor."""
    split = int(len(X) * 0.8)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X[:split], y[:split])
    return regressor


def predict(regressor: RandomForestRegressor, last_features: np.ndarray) -> tuple:
    """
    Predict next price from a 1-D feature vector.
    Returns (predicted_price, confidence_0_100, pred_std).
    """
    X = last_features.reshape(1, -1)
    tree_preds = np.array([t.predict(X)[0] for t in regressor.estimators_])
    predicted_price = float(tree_preds.mean())
    pred_std = float(tree_preds.std())
    confidence = float(np.clip(100.0 - (pred_std / max(predicted_price, 1.0) * 100.0), 0.0, 100.0))
    return predicted_price, confidence, pred_std


def get_training_predictions(regressor: RandomForestRegressor, X: np.ndarray) -> tuple:
    """Return (means, stds) arrays over all training rows — used for chart confidence band."""
    tree_matrix = np.array([t.predict(X) for t in regressor.estimators_])
    return tree_matrix.mean(axis=0), tree_matrix.std(axis=0)


def simulate_scenarios(regressor: RandomForestRegressor, last_features: np.ndarray,
                       baseline_price: float) -> dict:
    """
    Perturb last_features for each scenario and return price deltas vs baseline.
    last_features layout: [oil_price, usd_php, demand_index, prev_gas_price]
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

- [ ] **Step 4: Run — expect PASS**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_model.py -v
```

Expected: `6 passed` (may take ~5s for tree predictions)

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/model.py ph_economic_ai/tests/test_model.py
git commit -m "feat: add ML model — train, predict, simulate scenarios"
```

---

## Task 5: Rule-Based Explanation (`utils/explanation.py`)

**Files:**
- Create: `ph_economic_ai/utils/explanation.py`
- Create: `ph_economic_ai/tests/test_explanation.py`

- [ ] **Step 1: Write the failing tests**

`ph_economic_ai/tests/test_explanation.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.utils.explanation import generate


def _gen(oil_delta=0.0, usd_delta=0.0, demand_norm=0.7,
         pressure_index=50.0, current=70.0, predicted=71.5):
    return generate(oil_delta, usd_delta, demand_norm, pressure_index, current, predicted)


def test_returns_required_keys():
    result = _gen()
    for key in ('drivers', 'risk_badge', 'risk_color', 'summary',
                'advisory', 'advisory_icon', 'expected_increase', 'price_direction'):
        assert key in result, f"Missing key: {key}"


def test_drivers_has_three_items():
    assert len(_gen()['drivers']) == 3


def test_driver_keys():
    driver = _gen()['drivers'][0]
    for key in ('icon', 'name', 'value', 'status', 'color'):
        assert key in driver


def test_high_pressure_advisory():
    result = _gen(pressure_index=75.0)
    assert 'Refuel' in result['advisory']
    assert result['advisory_icon'] == '⛽'


def test_rising_pressure_advisory():
    result = _gen(pressure_index=45.0)
    assert 'Monitor' in result['advisory']


def test_stable_advisory():
    result = _gen(pressure_index=15.0)
    assert 'stable' in result['advisory'].lower() or 'No action' in result['advisory']


def test_price_direction_increase():
    result = _gen(current=70.0, predicted=72.0)
    assert result['price_direction'] == 'increase'


def test_price_direction_decrease():
    result = _gen(current=72.0, predicted=70.0)
    assert result['price_direction'] == 'decrease'


def test_risk_colors():
    assert _gen(pressure_index=75.0)['risk_color'] == '#E07A4A'
    assert _gen(pressure_index=45.0)['risk_color'] == '#E0A84A'
    assert _gen(pressure_index=15.0)['risk_color'] == '#4A90E2'
```

- [ ] **Step 2: Run — expect FAIL**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_explanation.py -v
```

- [ ] **Step 3: Implement `explanation.py`**

`ph_economic_ai/utils/explanation.py`:
```python
def generate(oil_delta: float, usd_delta: float, demand_norm: float,
             pressure_index: float, current_price: float, predicted_price: float) -> dict:

    # ── Drivers ──────────────────────────────────────────────────────────────
    if oil_delta > 0.5:
        oil_label, oil_color = '↑ High', '#E07A4A'
    elif oil_delta > 0.0:
        oil_label, oil_color = '↑ Rising', '#E07A4A'
    else:
        oil_label, oil_color = '→ Neutral', '#888888'

    if usd_delta > 0.3:
        usd_label, usd_color = '↑ Rising', '#E07A4A'
    else:
        usd_label, usd_color = '→ Stable', '#888888'

    if demand_norm > 0.7:
        dem_label, dem_color = '↑ High', '#E07A4A'
    else:
        dem_label, dem_color = '→ Neutral', '#888888'

    drivers = [
        {
            'icon': '🛢', 'name': 'Crude Oil',
            'value': f'Δ {oil_delta:+.2f}σ · Weight 50%',
            'status': oil_label, 'color': oil_color,
        },
        {
            'icon': '💱', 'name': 'USD / PHP',
            'value': f'Δ {usd_delta:+.2f}σ · Weight 30%',
            'status': usd_label, 'color': usd_color,
        },
        {
            'icon': '📊', 'name': 'Demand Index',
            'value': f'{demand_norm * 100:.0f}/100 · Weight 20%',
            'status': dem_label, 'color': dem_color,
        },
    ]

    # ── Risk badge ────────────────────────────────────────────────────────────
    if pressure_index > 60:
        risk_badge = '⚠ High Pressure — Price rise likely'
        risk_color = '#E07A4A'
    elif pressure_index > 30:
        risk_badge = '⚡ Rising Pressure — Monitor closely'
        risk_color = '#E0A84A'
    else:
        risk_badge = '✓ Stable — No immediate risk'
        risk_color = '#4A90E2'

    # ── Summary ───────────────────────────────────────────────────────────────
    parts = []
    if oil_delta > 0.3:
        parts.append('Oil prices are pushing import costs higher.')
    if usd_delta > 0.3:
        parts.append('A stronger dollar makes fuel imports more expensive.')
    if demand_norm > 0.7:
        parts.append('High demand is adding to price pressure.')
    if not parts:
        parts.append('Economic indicators are broadly stable.')
    summary = ' '.join(parts)

    # ── Advisory ─────────────────────────────────────────────────────────────
    if pressure_index > 60:
        advisory, advisory_icon = 'Refuel within 48 hours', '⛽'
    elif pressure_index > 30:
        advisory, advisory_icon = 'Monitor prices this week', '👁'
    else:
        advisory, advisory_icon = 'No action needed — prices stable', '✓'

    price_change = predicted_price - current_price
    sign = '+' if price_change >= 0 else ''
    direction = 'increase' if price_change >= 0 else 'decrease'

    return {
        'drivers': drivers,
        'risk_badge': risk_badge,
        'risk_color': risk_color,
        'summary': summary,
        'advisory': advisory,
        'advisory_icon': advisory_icon,
        'expected_increase': f'{sign}₱{abs(price_change):.2f} / liter',
        'price_direction': direction,
    }
```

- [ ] **Step 4: Run — expect PASS**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/test_explanation.py -v
```

Expected: `9 passed`

- [ ] **Step 5: Run full test suite**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add ph_economic_ai/utils/explanation.py ph_economic_ai/tests/test_explanation.py
git commit -m "feat: add rule-based explanation and advisory generator"
```

---

## Task 6: Sidebar Widget (`ui/sidebar.py`)

**Files:**
- Create: `ph_economic_ai/ui/sidebar.py`

- [ ] **Step 1: Implement `sidebar.py`**

`ph_economic_ai/ui/sidebar.py`:
```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtCore import pyqtSignal


class SidebarWidget(QWidget):
    page_changed = pyqtSignal(int)

    _NAV = [
        ('ANALYSIS', None),
        ('◈', 'Dashboard', 0),
        ('◉', 'Pressure Index', 1),
        ('⬡', 'Agent Network', 2),
        ('SYSTEM', None),
        ('⚙', 'Settings', 3),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(190)
        self.setStyleSheet('background: #FFFFFF;')
        self._buttons: list[tuple[QPushButton, int]] = []
        self._active_idx = 0
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Logo block
        logo = QFrame()
        logo.setStyleSheet('background:#FFFFFF; border-bottom: 1px solid #EAEAEA;')
        ll = QVBoxLayout(logo)
        ll.setContentsMargins(18, 14, 18, 14)
        ll.setSpacing(2)
        top = QLabel('PH ECONAI')
        top.setStyleSheet('font-size:11px; font-weight:700; color:#4A90E2; letter-spacing:1px;')
        sub = QLabel('Economic Advisor')
        sub.setStyleSheet('font-size:10px; color:#BBBBBB;')
        ll.addWidget(top)
        ll.addWidget(sub)
        layout.addWidget(logo)

        for item in self._NAV:
            if item[1] is None:
                lbl = QLabel(item[0])
                lbl.setStyleSheet(
                    'font-size:9px; font-weight:700; color:#CCCCCC;'
                    'letter-spacing:1px; padding:12px 18px 4px 18px;'
                )
                layout.addWidget(lbl)
            else:
                icon, text, page_idx = item
                btn = QPushButton(f'{icon}  {text}')
                btn.setFlat(True)
                btn.setCursor(self.cursor())
                btn.setStyleSheet(self._style(page_idx == 0))
                btn.clicked.connect(lambda _, idx=page_idx: self._on_click(idx))
                self._buttons.append((btn, page_idx))
                layout.addWidget(btn)

        layout.addStretch()

        footer = QLabel('  ●  Trained · Offline')
        footer.setStyleSheet(
            'font-size:10px; color:#4A90E2; font-weight:600;'
            'background:#EBF4FF; border-radius:10px;'
            'padding:4px 8px; margin:12px 14px;'
        )
        layout.addWidget(footer)

    def _style(self, active: bool) -> str:
        if active:
            return (
                'text-align:left; padding:9px 18px; font-size:13px;'
                'color:#4A90E2; background:#EBF4FF;'
                'border:none; border-left:3px solid #4A90E2; font-weight:600;'
            )
        return (
            'text-align:left; padding:9px 18px 9px 21px; font-size:13px;'
            'color:#666666; background:transparent; border:none;'
        )

    def _on_click(self, idx: int):
        self._active_idx = idx
        for btn, page_idx in self._buttons:
            btn.setStyleSheet(self._style(page_idx == idx))
        self.page_changed.emit(idx)
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/sidebar.py
git commit -m "feat: add sidebar navigation widget"
```

---

## Task 7: Price Chart (`ui/charts.py`)

**Files:**
- Create: `ph_economic_ai/ui/charts.py`

- [ ] **Step 1: Implement `charts.py`**

`ph_economic_ai/ui/charts.py`:
```python
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PriceChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 2.6), dpi=96)
        self.figure.patch.set_facecolor('#FFFFFF')
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        self.ax.set_facecolor('#FAFAFA')
        for spine in ('top', 'right'):
            self.ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            self.ax.spines[spine].set_color('#EAEAEA')
        self.ax.tick_params(colors='#AAAAAA', labelsize=8)
        self.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'₱{x:.0f}')
        )
        self.ax.grid(axis='y', color='#F0F0F0', linewidth=0.6, zorder=0)
        self.figure.tight_layout(pad=1.2)

    def update_data(self, dates: list, actuals: np.ndarray, train_means: np.ndarray,
                    train_stds: np.ndarray, predicted_price: float, pred_std: float):
        self.ax.clear()
        self._style_axes()

        n = len(actuals)
        x_actual = list(range(n))
        x_pred = list(range(n)) + [n]
        pred_line = list(train_means) + [predicted_price]
        upper = list(train_means + train_stds) + [predicted_price + pred_std]
        lower = list(train_means - train_stds) + [predicted_price - pred_std]

        # Confidence band
        self.ax.fill_between(x_pred, lower, upper, color='#C8DEF5', alpha=0.45, zorder=1)

        # Actual line
        self.ax.plot(x_actual, actuals, color='#999999', linewidth=1.8,
                     label='Actual', zorder=2)

        # Predicted line
        self.ax.plot(x_pred, pred_line, color='#4A90E2', linewidth=2.2,
                     label='Predicted', zorder=3)

        # Forecast divider
        self.ax.axvline(x=n - 1, color='#DDDDDD', linewidth=1.0,
                        linestyle='--', zorder=2)
        self.ax.text(n - 0.5, self.ax.get_ylim()[1], '→ Forecast',
                     fontsize=7, color='#BBBBBB', va='top')

        # Endpoint markers
        self.ax.plot(n - 1, actuals[-1], 'o', color='#FFFFFF',
                     markeredgecolor='#999999', markersize=5, zorder=4)
        self.ax.plot(n, predicted_price, 'o', color='#4A90E2', markersize=5, zorder=4)

        # X-axis labels (every 12 months)
        tick_pos = list(range(0, n, 12))
        tick_labels = [dates[i] for i in tick_pos if i < len(dates)]
        self.ax.set_xticks(tick_pos[:len(tick_labels)])
        self.ax.set_xticklabels(tick_labels, fontsize=8, color='#AAAAAA')

        # Legend
        self.ax.legend(fontsize=8, loc='upper left',
                       framealpha=0.7, edgecolor='#EAEAEA')

        self.figure.tight_layout(pad=1.2)
        self.draw()
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/charts.py
git commit -m "feat: add matplotlib price chart with confidence band"
```

---

## Task 8: Pressure Gauge (`ui/pressure.py`)

**Files:**
- Create: `ph_economic_ai/ui/pressure.py`

- [ ] **Step 1: Implement `pressure.py`**

`ph_economic_ai/ui/pressure.py`:
```python
import math
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QScrollArea, QSizePolicy)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPainterPath


_BAND_CONFIG = [
    ('Stable',   '0 – 30',   '#EBF4FF', '#4A90E2'),
    ('Rising',   '31 – 60',  '#FFF8EE', '#E0A84A'),
    ('High',     '61 – 80',  '#FFF3EE', '#E07A4A'),
    ('Critical', '81 – 100', '#FFEFEE', '#E05040'),
]


class PressureGauge(QWidget):
    def __init__(self, size: int = 120, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._size = size
        self.setFixedSize(size, size)

    def set_value(self, value: float):
        self._value = float(max(0.0, min(100.0, value)))
        self.update()

    def _arc_color(self) -> str:
        if self._value <= 30:
            return '#4A90E2'
        elif self._value <= 60:
            return '#E0A84A'
        elif self._value <= 80:
            return '#E07A4A'
        return '#E05040'

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        m = 10
        side = self._size - 2 * m
        rect = QRectF(m + 6, m + 6, side - 12, side - 12)
        pw = max(8, self._size // 10)

        # Background track
        pen = QPen(QColor('#EAEAEA'), pw, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, -270 * 16)

        # Value arc
        pen.setColor(QColor(self._arc_color()))
        painter.setPen(pen)
        span = int(-270.0 * self._value / 100.0 * 16)
        if span != 0:
            painter.drawArc(rect, 225 * 16, span)

        # Value text (large)
        painter.setPen(QPen(QColor('#111111')))
        f = QFont()
        f.setPointSize(max(10, self._size // 6))
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(
            QRectF(m, m + side * 0.15, side, side * 0.45),
            Qt.AlignmentFlag.AlignCenter,
            f'{int(self._value)}'
        )

        # Sub-label
        f2 = QFont()
        f2.setPointSize(max(7, self._size // 14))
        painter.setFont(f2)
        painter.setPen(QPen(QColor('#BBBBBB')))
        painter.drawText(
            QRectF(m, m + side * 0.55, side, side * 0.25),
            Qt.AlignmentFlag.AlignCenter,
            '/ 100'
        )


def _band_card(label: str, rng: str, bg: str, color: str, active: bool) -> QFrame:
    card = QFrame()
    border = f'border: 1.5px solid {color};' if active else 'border: 1px solid #EAEAEA;'
    card.setStyleSheet(f'background:{bg}; border-radius:6px; {border}')
    layout = QVBoxLayout(card)
    layout.setContentsMargins(8, 6, 8, 6)
    layout.setSpacing(1)
    lbl = QLabel(('⚠ ' if active else '') + label)
    lbl.setStyleSheet(f'font-size:9px; font-weight:700; color:{color};'
                      f'text-transform:uppercase; letter-spacing:0.5px;')
    rng_lbl = QLabel(rng)
    rng_lbl.setStyleSheet(f'font-size:9px; color:{color};')
    if active:
        now = QLabel('← NOW')
        now.setStyleSheet(f'font-size:8px; font-weight:700; color:{color};')
        layout.addWidget(now)
    layout.addWidget(lbl)
    layout.addWidget(rng_lbl)
    return card


class PressureGaugePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._gauge = PressureGauge(size=160)
        self._band_cards: list[QFrame] = []
        self._history_labels: list[QLabel] = []
        self._history: list[float] = []
        self._current_band = 'Stable'
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        title = QLabel('Economic Pressure Index')
        title.setStyleSheet('font-size:20px; font-weight:700; color:#111111;')
        layout.addWidget(title)

        sub = QLabel('Weighted composite of oil price, USD/PHP rate, and fuel demand')
        sub.setStyleSheet('font-size:12px; color:#888888;')
        layout.addWidget(sub)

        # Gauge centered
        gauge_row = QHBoxLayout()
        gauge_row.addStretch()
        gauge_row.addWidget(self._gauge)
        gauge_row.addStretch()
        layout.addLayout(gauge_row)

        # Bands grid (2×2)
        bands_row = QHBoxLayout()
        bands_row.setSpacing(8)
        for label, rng, bg, color in _BAND_CONFIG:
            card = _band_card(label, rng, bg, color, active=(label == self._current_band))
            self._band_cards.append(card)
            bands_row.addWidget(card)
        layout.addLayout(bands_row)

        # History
        hist_title = QLabel('RECENT INDEX VALUES')
        hist_title.setStyleSheet(
            'font-size:10px; font-weight:700; color:#BBBBBB; letter-spacing:0.8px;'
        )
        layout.addWidget(hist_title)

        self._hist_layout = QVBoxLayout()
        self._hist_layout.setSpacing(4)
        layout.addLayout(self._hist_layout)

        layout.addStretch()

    def refresh(self, result: dict):
        index = result['pressure_index']
        band = result['pressure_band']
        self._gauge.set_value(index)
        self._current_band = band

        for i, (label, rng, bg, color) in enumerate(_BAND_CONFIG):
            active = label == band
            card = self._band_cards[i]
            border = f'border: 1.5px solid {color};' if active else 'border: 1px solid #EAEAEA;'
            card.setStyleSheet(f'background:{bg}; border-radius:6px; {border}')

        # Update history
        self._history.append(index)
        self._history = self._history[-5:]
        while self._hist_layout.count():
            item = self._hist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for val in reversed(self._history):
            b = pressure_band_color(val)
            lbl = QLabel(f'  {val:.1f} / 100  —  {result["pressure_band"]}')
            lbl.setStyleSheet(f'font-size:12px; color:{b}; padding:4px 0;')
            self._hist_layout.addWidget(lbl)


def pressure_band_color(index: float) -> str:
    if index <= 30:
        return '#4A90E2'
    elif index <= 60:
        return '#E0A84A'
    elif index <= 80:
        return '#E07A4A'
    return '#E05040'
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/pressure.py
git commit -m "feat: add PressureGauge widget and PressureGaugePage"
```

---

## Task 9: Agent Network (`ui/agent_graph.py`)

**Files:**
- Create: `ph_economic_ai/ui/agent_graph.py`

- [ ] **Step 1: Implement `agent_graph.py`**

`ph_economic_ai/ui/agent_graph.py`:
```python
import math
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel,
                              QGraphicsView, QGraphicsScene, QGraphicsItem)
from PyQt6.QtCore import Qt, QRectF, QPointF, QLineF
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QPolygonF, QPainterPath


_NODES = [
    ('Oil Market',        80,  80),
    ('USD / PHP',         80, 190),
    ('Demand',            80, 300),
    ('ML Model',         280, 185),
    ('Prediction\nOutput', 490,  80),
    ('Economic\nIndex',  490, 300),
]

_EDGES = [
    (0, 3), (1, 3), (2, 3),   # inputs → ML Model
    (3, 4), (3, 5),            # ML Model → outputs
]

_W, _H = 130, 52


class _NodeItem(QGraphicsItem):
    def __init__(self, label: str, x: float, y: float):
        super().__init__()
        self._label = label
        self.setPos(x, y)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, _W, _H)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, _W, _H), 8, 8)
        painter.fillPath(path, QColor('#FFFFFF'))
        painter.setPen(QPen(QColor('#4A90E2'), 1.5))
        painter.drawPath(path)
        painter.setPen(QPen(QColor('#111111')))
        f = QFont()
        f.setPointSize(9)
        painter.setFont(f)
        painter.drawText(
            QRectF(0, 0, _W, _H),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )


class _EdgeItem:
    """Draws a line with an arrowhead from source node edge to target node edge."""

    def __init__(self, scene: QGraphicsScene, src_pos: QPointF, dst_pos: QPointF):
        # Offset to node centres
        sx = src_pos.x() + _W
        sy = src_pos.y() + _H / 2
        dx = dst_pos.x()
        dy = dst_pos.y() + _H / 2

        pen = QPen(QColor('#CCCCCC'), 1.5)
        pen.setStyle(Qt.PenStyle.SolidLine)
        line = scene.addLine(sx, sy, dx, dy, pen)
        line.setZValue(-1)

        # Arrowhead
        angle = math.atan2(dy - sy, dx - sx)
        size = 9
        p1 = QPointF(
            dx - size * math.cos(angle - math.pi / 6),
            dy - size * math.sin(angle - math.pi / 6),
        )
        p2 = QPointF(
            dx - size * math.cos(angle + math.pi / 6),
            dy - size * math.sin(angle + math.pi / 6),
        )
        arrow = QPolygonF([QPointF(dx, dy), p1, p2])
        scene.addPolygon(arrow, QPen(Qt.PenStyle.NoPen), QColor('#CCCCCC'))


class AgentGraphPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = QLabel('Agent Network')
        title.setStyleSheet('font-size:20px; font-weight:700; color:#111111;')
        layout.addWidget(title)

        sub = QLabel('Data flow from economic inputs through the ML model to outputs')
        sub.setStyleSheet('font-size:12px; color:#888888;')
        layout.addWidget(sub)

        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, 700, 420)
        scene.setBackgroundBrush(QColor('#FAFAFA'))

        node_items = []
        for label, x, y in _NODES:
            item = _NodeItem(label, x, y)
            scene.addItem(item)
            node_items.append(QPointF(x, y))

        for src_idx, dst_idx in _EDGES:
            _EdgeItem(scene, node_items[src_idx], node_items[dst_idx])

        view = QGraphicsView(scene)
        view.setRenderHint(QPainter.RenderHint.Antialiasing)
        view.setStyleSheet('border: 1px solid #EAEAEA; border-radius: 10px; background: #FAFAFA;')
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        layout.addWidget(view)
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/agent_graph.py
git commit -m "feat: add agent network QGraphicsView page"
```

---

## Task 10: Dashboard Page (`ui/dashboard.py`)

**Files:**
- Create: `ph_economic_ai/ui/dashboard.py`

- [ ] **Step 1: Implement `dashboard.py`**

`ph_economic_ai/ui/dashboard.py`:
```python
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QFrame, QPushButton, QScrollArea, QSizePolicy,
                              QProgressBar)
from PyQt6.QtCore import pyqtSignal, Qt
from ph_economic_ai.ui.charts import PriceChart
from ph_economic_ai.ui.pressure import PressureGauge, pressure_band_color


# ── Helpers ───────────────────────────────────────────────────────────────────

def _card(parent=None) -> QFrame:
    f = QFrame(parent)
    f.setStyleSheet('background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;')
    return f


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        'font-size:10px; font-weight:700; color:#BBBBBB;'
        'text-transform:uppercase; letter-spacing:0.9px;'
    )
    return lbl


# ── Mini summary card ─────────────────────────────────────────────────────────

class _MiniCard(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(3)
        self._label = QLabel(label.upper())
        self._label.setStyleSheet(
            'font-size:10px; color:#AAAAAA; letter-spacing:0.5px; border:none;'
        )
        self._value = QLabel('—')
        self._value.setStyleSheet('font-size:19px; font-weight:700; color:#111111; border:none;')
        self._badge = QLabel('')
        self._badge.setStyleSheet(
            'font-size:10px; font-weight:600; padding:2px 8px;'
            'border-radius:10px; border:none;'
        )
        layout.addWidget(self._label)
        layout.addWidget(self._value)
        layout.addWidget(self._badge)

    def update(self, value: str, badge: str, val_color: str, badge_bg: str, badge_color: str):
        self._value.setText(value)
        self._value.setStyleSheet(
            f'font-size:19px; font-weight:700; color:{val_color}; border:none;'
        )
        self._badge.setText(badge)
        self._badge.setStyleSheet(
            f'font-size:10px; font-weight:600; padding:2px 8px; border-radius:10px;'
            f'background:{badge_bg}; color:{badge_color}; border:none;'
        )


# ── If-Then Simulation panel ──────────────────────────────────────────────────

class _SimScenario(QFrame):
    def __init__(self, if_text: str, explanation: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet('border:none;')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        if_lbl = QLabel('IF')
        if_lbl.setStyleSheet('font-size:9px; font-weight:700; color:#CCCCCC; border:none;')
        layout.addWidget(if_lbl)

        self._cond = QLabel(if_text)
        self._cond.setStyleSheet('font-size:12px; font-weight:600; color:#333333; border:none;')
        self._cond.setWordWrap(True)
        layout.addWidget(self._cond)

        arrow_row = QHBoxLayout()
        arrow_lbl = QLabel('→')
        arrow_lbl.setStyleSheet('font-size:16px; color:#DDDDDD; border:none;')
        arrow_row.addWidget(arrow_lbl)
        self._impact = QLabel('—')
        self._impact.setStyleSheet('font-size:20px; font-weight:700; border:none;')
        arrow_row.addWidget(self._impact)
        arrow_row.addStretch()
        layout.addLayout(arrow_row)

        self._unit = QLabel('per liter')
        self._unit.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        layout.addWidget(self._unit)

        self._bar = QProgressBar()
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(4)
        self._bar.setStyleSheet(
            'QProgressBar { background:#F0F0F0; border-radius:2px; border:none; }'
            'QProgressBar::chunk { border-radius:2px; }'
        )
        layout.addWidget(self._bar)

        self._desc = QLabel(explanation)
        self._desc.setStyleSheet('font-size:10px; color:#AAAAAA; border:none;')
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

    def set_delta(self, delta: float, max_abs: float, up_color: str, down_color: str):
        color = up_color if delta >= 0 else down_color
        sign = '+' if delta >= 0 else '−'
        self._impact.setText(f'{sign}₱{abs(delta):.2f} / L')
        self._impact.setStyleSheet(f'font-size:20px; font-weight:700; color:{color}; border:none;')
        chunk_color = up_color if delta >= 0 else down_color
        self._bar.setStyleSheet(
            'QProgressBar { background:#F0F0F0; border-radius:2px; border:none; }'
            f'QProgressBar::chunk {{ border-radius:2px; background:{chunk_color}; }}'
        )
        pct = int(min(abs(delta) / max(max_abs, 0.01) * 100, 100))
        self._bar.setValue(pct)


class SimulationPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            'background:#FFFFFF; border:1px solid #EAEAEA; border-radius:10px;'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            'background:#FFFFFF; border:none; border-bottom:1px solid #EAEAEA;'
            'border-top-left-radius:10px; border-top-right-radius:10px;'
        )
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 10, 18, 10)
        icon = QLabel('🔮')
        icon.setStyleSheet('font-size:15px; border:none;')
        title = QLabel('If-Then Simulation')
        title.setStyleSheet('font-size:13px; font-weight:700; color:#111111; border:none;')
        badge = QLabel('Scenario impact on gas price')
        badge.setStyleSheet(
            'font-size:10px; font-weight:600; color:#888888;'
            'background:#F5F5F5; padding:3px 9px; border-radius:10px; border:none;'
        )
        hh.addWidget(icon)
        hh.addWidget(title)
        hh.addStretch()
        hh.addWidget(badge)
        layout.addWidget(header)

        # Three scenario columns
        cols = QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(0)

        self._oil_col = _SimScenario(
            'Oil prices rise +5%',
            'Higher crude input cost flows directly into refinery output pricing.'
        )
        self._usd_col = _SimScenario(
            'USD strengthens +2% vs PHP',
            'Dollar-denominated imports become more expensive in peso terms.'
        )
        self._dem_col = _SimScenario(
            'Demand index drops 10 pts',
            'Reduced consumption eases upward pressure, softening the price.'
        )

        for i, col in enumerate([self._oil_col, self._usd_col, self._dem_col]):
            if i > 0:
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.VLine)
                sep.setStyleSheet('color:#F5F5F5; border:none; border-left:1px solid #F5F5F5;')
                cols.addWidget(sep)
            cols.addWidget(col)

        layout.addLayout(cols)

    def refresh(self, scenarios: dict):
        max_abs = max(abs(v) for v in scenarios.values()) or 1.0
        self._oil_col.set_delta(scenarios['oil_shock'],   max_abs, '#E07A4A', '#4AAE90')
        self._usd_col.set_delta(scenarios['usd_shock'],   max_abs, '#E07A4A', '#4AAE90')
        self._dem_col.set_delta(scenarios['demand_drop'], max_abs, '#E07A4A', '#4AAE90')


# ── Right panel (gauge + drivers + advisory) ──────────────────────────────────

class _RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(270)
        self.setStyleSheet('background:#FFFFFF; border-left:1px solid #EAEAEA;')
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet('QScrollArea { border:none; } QScrollBar { width:0px; }')
        outer.addWidget(scroll)

        inner = QWidget()
        inner.setStyleSheet('background:#FFFFFF;')
        self._layout = QVBoxLayout(inner)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        scroll.setWidget(inner)

        self._build_gauge_section()
        self._build_drivers_section()
        self._build_advisory_section()
        self._layout.addStretch()

    # ── Gauge + bands ─────────────────────────────────────────────────────────
    def _build_gauge_section(self):
        sec = QFrame()
        sec.setStyleSheet('background:#FFFFFF; border-bottom:1px solid #EAEAEA;')
        lyt = QVBoxLayout(sec)
        lyt.setContentsMargins(18, 16, 18, 14)
        lyt.setSpacing(8)
        lyt.addWidget(_section_title('Economic Pressure Index'))

        self._gauge = PressureGauge(size=100)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(self._gauge)
        row.addStretch()
        lyt.addLayout(row)

        bands_grid = QHBoxLayout()
        bands_grid.setSpacing(4)
        self._band_frames: dict[str, QFrame] = {}
        self._band_labels: dict[str, QLabel] = {}
        configs = [
            ('Stable',   '0–30',   '#EBF4FF', '#4A90E2'),
            ('Rising',   '31–60',  '#FFF8EE', '#E0A84A'),
            ('High',     '61–80',  '#FFF3EE', '#E07A4A'),
            ('Critical', '81–100', '#FFEFEE', '#E05040'),
        ]
        for name, rng, bg, color in configs:
            f = QFrame()
            f.setStyleSheet(f'background:{bg}; border-radius:6px; border:1px solid #EAEAEA;')
            fl = QVBoxLayout(f)
            fl.setContentsMargins(6, 5, 6, 5)
            fl.setSpacing(1)
            nl = QLabel(name)
            nl.setStyleSheet(f'font-size:9px; font-weight:700; color:{color}; border:none;')
            rl = QLabel(rng)
            rl.setStyleSheet(f'font-size:9px; color:{color}; border:none;')
            fl.addWidget(nl)
            fl.addWidget(rl)
            bands_grid.addWidget(f)
            self._band_frames[name] = f
            self._band_labels[name] = nl

        self._now_label = QLabel('← NOW')
        lyt.addLayout(bands_grid)
        self._layout.addWidget(sec)

    # ── Drivers ───────────────────────────────────────────────────────────────
    def _build_drivers_section(self):
        sec = QFrame()
        sec.setStyleSheet('background:#FFFFFF; border-bottom:1px solid #EAEAEA;')
        lyt = QVBoxLayout(sec)
        lyt.setContentsMargins(18, 14, 18, 14)
        lyt.setSpacing(6)
        lyt.addWidget(_section_title('Key Drivers'))

        self._driver_rows: list[tuple[QLabel, QLabel, QLabel]] = []
        for _ in range(3):
            row = QHBoxLayout()
            row.setSpacing(8)
            icon_lbl = QLabel()
            icon_lbl.setStyleSheet('font-size:14px; border:none;')
            icon_lbl.setFixedWidth(20)
            text_col = QVBoxLayout()
            text_col.setSpacing(1)
            name_lbl = QLabel()
            name_lbl.setStyleSheet('font-size:12px; font-weight:600; color:#111111; border:none;')
            val_lbl = QLabel()
            val_lbl.setStyleSheet('font-size:10px; color:#888888; border:none;')
            text_col.addWidget(name_lbl)
            text_col.addWidget(val_lbl)
            status_lbl = QLabel()
            status_lbl.setStyleSheet('font-size:11px; font-weight:700; border:none;')
            row.addWidget(icon_lbl)
            row.addLayout(text_col)
            row.addStretch()
            row.addWidget(status_lbl)
            self._driver_rows.append((icon_lbl, name_lbl, val_lbl, status_lbl, row))
            lyt.addLayout(row)

        self._risk_badge = QLabel()
        self._risk_badge.setStyleSheet(
            'font-size:10px; font-weight:700; padding:4px 10px; border-radius:10px; border:none;'
        )
        lyt.addWidget(self._risk_badge)
        self._summary = QLabel()
        self._summary.setStyleSheet('font-size:11px; color:#555555; border:none;')
        self._summary.setWordWrap(True)
        lyt.addWidget(self._summary)
        self._layout.addWidget(sec)

    # ── Advisory ──────────────────────────────────────────────────────────────
    def _build_advisory_section(self):
        self._advisory_card = QFrame()
        self._advisory_card.setStyleSheet(
            'background:qlineargradient(x1:0,y1:0,x2:1,y2:1,'
            'stop:0 #EBF4FF, stop:1 #F0F7FF);'
            'border:1px solid #C8DEF5; border-radius:10px; margin:14px 18px 0px 18px;'
        )
        lyt = QVBoxLayout(self._advisory_card)
        lyt.setContentsMargins(14, 12, 14, 12)
        lyt.setSpacing(6)

        hdr = QHBoxLayout()
        self._adv_icon = QLabel('💡')
        self._adv_icon.setStyleSheet('font-size:14px; border:none;')
        adv_title = QLabel('Advisory Output')
        adv_title.setStyleSheet(
            'font-size:12px; font-weight:700; color:#4A90E2; border:none;'
        )
        hdr.addWidget(self._adv_icon)
        hdr.addWidget(adv_title)
        hdr.addStretch()
        lyt.addLayout(hdr)

        for attr, label in [
            ('_adv_increase', 'Expected change'),
            ('_adv_timing',   'Price adjustment'),
            ('_adv_window',   'Risk window'),
        ]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet('font-size:11px; color:#666666; border:none;')
            val = QLabel('—')
            val.setStyleSheet('font-size:11px; font-weight:700; color:#111111; border:none;')
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(val)
            lyt.addLayout(row)
            setattr(self, attr, val)

        self._action_btn = QLabel()
        self._action_btn.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._action_btn.setStyleSheet(
            'background:#4A90E2; color:#FFFFFF; border-radius:6px; border:none;'
            'font-size:11px; font-weight:600; padding:8px 12px; margin-top:4px;'
        )
        lyt.addWidget(self._action_btn)
        self._layout.addWidget(self._advisory_card)

    # ── Public refresh ────────────────────────────────────────────────────────
    def refresh(self, result: dict):
        self._gauge.set_value(result['pressure_index'])
        band = result['pressure_band']

        for name, frame in self._band_frames.items():
            active = name == band
            color = {'Stable':'#4A90E2','Rising':'#E0A84A','High':'#E07A4A','Critical':'#E05040'}[name]
            bg    = {'Stable':'#EBF4FF','Rising':'#FFF8EE','High':'#FFF3EE','Critical':'#FFEFEE'}[name]
            border = f'border:1.5px solid {color};' if active else 'border:1px solid #EAEAEA;'
            frame.setStyleSheet(f'background:{bg}; border-radius:6px; {border}')

        expl = result['explanation']
        for i, d in enumerate(expl['drivers']):
            icon_lbl, name_lbl, val_lbl, status_lbl, _ = self._driver_rows[i]
            icon_lbl.setText(d['icon'])
            name_lbl.setText(d['name'])
            val_lbl.setText(d['value'])
            status_lbl.setText(d['status'])
            status_lbl.setStyleSheet(
                f'font-size:11px; font-weight:700; color:{d["color"]}; border:none;'
            )

        self._risk_badge.setText(expl['risk_badge'])
        c = expl['risk_color']
        bg = {
            '#E07A4A': '#FFF3EE', '#E0A84A': '#FFF8EE', '#4A90E2': '#EBF4FF'
        }.get(c, '#F5F5F5')
        self._risk_badge.setStyleSheet(
            f'font-size:10px; font-weight:700; padding:4px 10px; border-radius:10px;'
            f'background:{bg}; color:{c}; border:1px solid {c};'
        )
        self._summary.setText(expl['summary'])

        self._adv_icon.setText(expl['advisory_icon'])
        self._adv_increase.setText(expl['expected_increase'])
        idx = result['pressure_index']
        self._adv_timing.setText('~48–72 hours' if idx > 60 else ('~1 week' if idx > 30 else 'Stable'))
        self._adv_window.setText(
            'High (next 7 days)' if idx > 60 else ('Medium' if idx > 30 else 'Low')
        )
        self._action_btn.setText(f'{expl["advisory_icon"]}  Suggested: {expl["advisory"]}')


# ── Dashboard page ────────────────────────────────────────────────────────────

class DashboardPage(QWidget):
    recalculate_requested = pyqtSignal()
    oil_shock_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet('background:#FAFAFA;')
        self._build()

    def _build(self):
        main = QHBoxLayout(self)
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
        pg_sub = QLabel('Philippines · Synthetic data · 120 data points · Trained on startup')
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

        # Chart
        self._chart = PriceChart()
        clyt.addWidget(self._chart)

        # Mini cards
        mini_row = QHBoxLayout()
        mini_row.setSpacing(10)
        self._price_card  = _MiniCard('Predicted Price')
        self._trend_card  = _MiniCard('Trend Direction')
        self._index_card  = _MiniCard('Pressure Index')
        for card in (self._price_card, self._trend_card, self._index_card):
            mini_row.addWidget(card)
        clyt.addLayout(mini_row)

        # Simulation panel
        self._sim_panel = SimulationPanel()
        clyt.addWidget(self._sim_panel)

        # ── Right panel ───────────────────────────────────────────────────────
        self._right = _RightPanel()

        main.addWidget(center, stretch=1)
        main.addWidget(self._right)

    def refresh(self, result: dict):
        # Chart
        self._chart.update_data(
            dates=result['df']['date'].tolist(),
            actuals=result['df']['gas_price'].values,
            train_means=result['train_means'],
            train_stds=result['train_stds'],
            predicted_price=result['predicted_price'],
            pred_std=result['pred_std'],
        )

        # Mini cards
        pp = result['predicted_price']
        cp = result['current_price']
        diff_pct = (pp - cp) / max(cp, 1) * 100
        sign = '+' if diff_pct >= 0 else ''
        self._price_card.update(
            f'₱{pp:.2f}', f'▲ {sign}{diff_pct:.1f}% vs current',
            '#4A90E2', '#FFF3EE', '#E07A4A'
        )

        trend = result['trend']
        trend_color = '#E07A4A' if trend == 'Rising' else ('#4AAE90' if trend == 'Falling' else '#888888')
        self._trend_card.update(
            f'{trend} {"▲" if trend=="Rising" else ("▼" if trend=="Falling" else "→")}',
            f'{result["confidence"]:.0f}% confidence',
            trend_color, '#EBF4FF', '#4A90E2'
        )

        idx = result['pressure_index']
        band = result['pressure_band']
        idx_color = pressure_band_color(idx)
        self._index_card.update(
            f'{idx:.0f} / 100', f'{band} Zone',
            idx_color,
            '#FFEFEE' if idx > 60 else '#FFF8EE',
            '#E05040' if idx > 80 else '#E07A4A'
        )

        # Simulation panel
        self._sim_panel.refresh(result['scenarios'])

        # Right panel
        self._right.refresh(result)
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/dashboard.py
git commit -m "feat: add dashboard page — chart, mini-cards, simulation, right panel"
```

---

## Task 11: Main Window (`ui/main_window.py`)

**Files:**
- Create: `ph_economic_ai/ui/main_window.py`

- [ ] **Step 1: Implement `main_window.py`**

`ph_economic_ai/ui/main_window.py`:
```python
import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                              QStackedWidget, QLabel, QVBoxLayout)
from PyQt6.QtCore import Qt

from ph_economic_ai.ui.sidebar import SidebarWidget
from ph_economic_ai.ui.dashboard import DashboardPage
from ph_economic_ai.ui.pressure import PressureGaugePage
from ph_economic_ai.ui.agent_graph import AgentGraphPage

from ph_economic_ai.utils.preprocessing import build_features, compute_index, pressure_band
from ph_economic_ai.utils.explanation import generate as generate_explanation
from ph_economic_ai import model as ml


class _SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lyt = QVBoxLayout(self)
        lyt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel('Settings — coming soon')
        lbl.setStyleSheet('font-size:16px; color:#AAAAAA;')
        lyt.addWidget(lbl)


class MainWindow(QMainWindow):
    def __init__(self, df, regressor, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._oil_shock_active = False

        X, y, _, self._df_feat = build_features(df)
        self._X = X
        self._last_features = np.array([
            self._df_feat.iloc[-1]['oil_price'],
            self._df_feat.iloc[-1]['usd_php'],
            self._df_feat.iloc[-1]['demand_index'],
            self._df_feat.iloc[-1]['gas_price'],
        ])

        self.setWindowTitle('Philippine Economic Pressure AI')
        self.setMinimumSize(1100, 680)
        self.setStyleSheet('background:#FFFFFF;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = SidebarWidget()
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

        # Initial render
        self._refresh()

    def _on_page_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)

    def _on_recalculate(self):
        self._oil_shock_active = False
        self._last_features = np.array([
            self._df_feat.iloc[-1]['oil_price'],
            self._df_feat.iloc[-1]['usd_php'],
            self._df_feat.iloc[-1]['demand_index'],
            self._df_feat.iloc[-1]['gas_price'],
        ])
        self._refresh()

    def _on_oil_shock(self):
        self._oil_shock_active = True
        features = self._last_features.copy()
        features[0] *= 1.10  # oil price +10%
        self._last_features = features
        self._refresh()

    def _build_result(self) -> dict:
        predicted_price, confidence, pred_std = ml.predict(
            self._regressor, self._last_features
        )
        current_price = float(self._df_feat.iloc[-1]['gas_price'])

        if predicted_price > current_price + 0.5:
            trend = 'Rising'
        elif predicted_price < current_price - 0.5:
            trend = 'Falling'
        else:
            trend = 'Stable'

        pressure_index, oil_delta, usd_delta, demand_norm = compute_index(
            float(self._last_features[0]),
            float(self._last_features[1]),
            float(self._last_features[2]),
            self._df,
        )
        band = pressure_band(pressure_index)

        explanation = generate_explanation(
            oil_delta, usd_delta, demand_norm,
            pressure_index, current_price, predicted_price,
        )

        scenarios = ml.simulate_scenarios(
            self._regressor, self._last_features, predicted_price
        )

        train_means, train_stds = ml.get_training_predictions(
            self._regressor, self._X
        )

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
        }

    def _refresh(self):
        result = self._build_result()
        self._dashboard.refresh(result)
        self._pressure_page.refresh(result)
```

- [ ] **Step 2: Commit**

```bash
git add ph_economic_ai/ui/main_window.py
git commit -m "feat: add MainWindow — wires sidebar, pages, and data pipeline"
```

---

## Task 12: Entry Point + Smoke Test (`main.py`)

**Files:**
- Create: `ph_economic_ai/main.py`

- [ ] **Step 1: Implement `main.py`**

`ph_economic_ai/main.py`:
```python
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # ── Startup pipeline ──────────────────────────────────────────────────────
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    regressor = ml.train(X, y)

    # ── Launch ────────────────────────────────────────────────────────────────
    window = MainWindow(df=df, regressor=regressor)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Run the full test suite one final time**

```bash
.venv\Scripts\pytest ph_economic_ai/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3: Smoke test — launch the app**

```bash
.venv\Scripts\python ph_economic_ai/main.py
```

Verify:
- Window opens (~1 second startup)
- Price chart renders with actual (gray) and predicted (blue) lines + confidence band
- Three mini-cards show price, trend, pressure values
- If-Then Simulation panel shows three columns with impact values
- Right panel shows gauge, band cards, drivers, advisory output
- Clicking "Pressure Index" in sidebar switches to gauge page
- Clicking "Agent Network" shows the 5-node graph
- "↺ Recalculate" resets to original data
- "⚡ Oil Shock +10%" raises predicted price and updates advisory

- [ ] **Step 4: Commit**

```bash
git add ph_economic_ai/main.py
git commit -m "feat: add entry point — Philippine Economic Pressure AI app complete"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by task |
|---|---|
| Gasoline price prediction | Tasks 4, 11 |
| Economic Pressure Index 0–100 | Tasks 3, 8, 10 |
| Price prediction graphs | Task 7, 10 |
| AI explanation panel | Tasks 5, 10 |
| Agent network visualization | Task 9 |
| Fully offline after startup | Tasks 2, 12 (no network calls) |
| Recalculate / Oil Shock buttons | Tasks 10, 11 |
| If-Then simulation panel | Tasks 4, 10 |
| Left sidebar nav | Task 6 |
| Pressure Index page with bands | Task 8 |
| Confidence band on chart | Tasks 7, 10 |
| Advisory output card | Task 10 |
| `result` dict schema | Task 11 |
| `PressureGauge` reuse (dashboard + page) | Tasks 8, 10 |
| Settings page placeholder | Task 11 |

**Placeholder scan:** No TBD, TODO, or "similar to" references. All code blocks contain complete implementation.

**Type consistency:** `predict()` returns `(float, float, float)` — used identically in Tasks 4 and 11. `simulate_scenarios()` returns `{'oil_shock', 'usd_shock', 'demand_drop'}` — consumed by `SimulationPanel.refresh()` in Task 10 using same keys. `result['df']` is the post-feature-build DataFrame with `date` column — consumed by `PriceChart.update_data()` with `result['df']['date'].tolist()`.

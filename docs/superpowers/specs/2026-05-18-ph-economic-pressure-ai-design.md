# Philippine Economic Pressure AI — Design Spec

**Date:** 2026-05-18  
**Status:** Approved  
**Location:** `ph_economic_ai/` (new subfolder in project root, sibling to AFP, Project_Maria, etc.)

---

## 1. Overview

A native PyQt6 desktop application that predicts Philippine gasoline prices, computes an Economic Pressure Index (0–100), and presents AI-generated economic analysis — all running fully offline after startup.

**Identity:** AI Economic Advisor (not a financial terminal). Advisory tone, insight-first, scannable panels.

**Entry point:** `python ph_economic_ai/main.py`

---

## 2. File Structure

```
ph_economic_ai/
├── main.py                  # QApplication entry point, constructs MainWindow
├── data.py                  # Synthetic dataset generation
├── model.py                 # ML training, prediction, confidence scoring
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # QMainWindow: sidebar + QStackedWidget pages
│   ├── dashboard.py         # Dashboard page (chart + mini-cards + sim panel + right panel)
│   ├── charts.py            # Matplotlib FigureCanvas wrapper
│   ├── pressure.py          # Pressure Index page + circular gauge widget
│   ├── agent_graph.py       # QGraphicsView agent network page
│   └── sidebar.py           # Left nav sidebar widget
└── utils/
    ├── __init__.py
    ├── explanation.py        # Rule-based AI explanation + advisory text
    └── preprocessing.py      # Feature scaling, delta computation, pressure index
```

---

## 3. Data Layer (`data.py`)

### Purpose
Generate a 120-row synthetic DataFrame representing monthly Philippine economic indicators from 2024-01 to 2033-12. No external files required.

### Schema

| Column         | Type    | Range       | Generation method                        |
|----------------|---------|-------------|------------------------------------------|
| `date`         | str     | 2024-01 … 2033-12 | Monthly periods                   |
| `oil_price`    | float   | 75–105 USD/bbl | Random walk with slight upward drift  |
| `usd_php`      | float   | 54–62       | Random walk correlated with oil          |
| `demand_index` | float   | 55–90       | Seasonal sine wave + Gaussian noise      |
| `gas_price`    | float   | 62–82 PHP/L | Derived: `f(oil, usd, demand) + noise`   |

### Contract
- Returns a `pd.DataFrame` with these five columns, 120 rows, no nulls.
- The last row represents the current period. Prediction target is the next (unseen) period.
- A `seed` parameter (default `42`) makes generation reproducible.

---

## 4. ML Model (`model.py`)

### Algorithm
`sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)`

xgboost is not installed in the project venv — scikit-learn 1.7.2 is used throughout.

### Features
```
[oil_price, usd_php, demand_index, prev_gas_price]
```
`prev_gas_price` is the lagged `gas_price` column (shift by 1).

### Training
- 80/20 train/test split (no shuffle — preserve time ordering).
- Trained once on startup using the synthetic dataset.
- No model persistence to disk (retrained each run; dataset is small).

### Outputs

| Output            | Type  | How computed                                                  |
|-------------------|-------|---------------------------------------------------------------|
| `predicted_price` | float | `model.predict(last_row_features)`                            |
| `confidence`      | float | `100 - (std of individual tree predictions / mean * 100)`, clipped 0–100 |
| `trend`           | str   | `"Rising"` if predicted > current, `"Stable"` if within ±0.5%, else `"Falling"` |

### Simulation predictions
For If-Then scenarios, the model is called with perturbed feature values — no retraining:
- Oil +5%: multiply `oil_price` in last row by 1.05, call `model.predict()`
- USD +2%: multiply `usd_php` by 1.02
- Demand −10pts: subtract 10 from `demand_index`

Delta vs. baseline prediction = scenario impact (shown as ±₱X.XX/L).

---

## 5. Economic Pressure Index (`utils/preprocessing.py`)

### Formula

```python
oil_delta   = (current_oil   - df.oil_price.mean())   / df.oil_price.std()
usd_delta   = (current_usd   - df.usd_php.mean())     / df.usd_php.std()
demand_norm = current_demand / 100.0

raw = (oil_delta * 0.50) + (usd_delta * 0.30) + (demand_norm * 0.20)
index = clip(normalize(raw, observed_min, observed_max, 0, 100), 0, 100)
```

### Bands

| Range   | Label    | UI color        |
|---------|----------|-----------------|
| 0–30    | Stable   | Blue (#4A90E2)  |
| 31–60   | Rising   | Orange (#E0A84A)|
| 61–80   | High     | Orange-red (#E07A4A) |
| 81–100  | Critical | Red (#E05040)   |

---

## 6. UI Design System

### Theme
- Background: `#FFFFFF` / `#FAFAFA` (panels)
- Primary text: `#111111`
- Secondary text: `#555555`
- Accent: `#4A90E2` (blue)
- Warning: `#E07A4A` (orange)
- Danger: `#E05040` (red)
- Border: `#EAEAEA`

### Layout
Three-column fixed layout:

```
┌─────────────┬──────────────────────────────┬──────────────┐
│  Sidebar    │        Center                │  Right Panel │
│  190px      │        flex-1                │  270px       │
│             │                              │              │
│  PH EconAI  │  [Header + buttons]          │  [Gauge]     │
│  ─────────  │  [Chart]                     │  [Bands]     │
│  Dashboard  │  [Mini cards ×3]             │  [Drivers]   │
│  Pressure   │  [If-Then Simulation]        │  [Advisory]  │
│  Agents     │                              │              │
│  Settings   │                              │              │
└─────────────┴──────────────────────────────┴──────────────┘
```

### Component Rules
- **Buttons:** white bg, 1px `#4A90E2` border, 8px radius, hover = `#EBF4FF` tint
- **Cards:** white bg, 1px `#EAEAEA` border, 10px radius
- **Active nav item:** `#EBF4FF` bg, left border `3px solid #4A90E2`
- **Typography:** system sans-serif (`-apple-system, Segoe UI`), bold titles only

---

## 7. UI Components

### `sidebar.py` — `SidebarWidget(QWidget)`
- Logo area: "PH ECONAI / Economic Advisor"
- Nav items: Dashboard, Pressure Index, Agent Network, Settings
- Emits `page_changed(int)` signal on click
- Footer: status pill (Trained · Offline)

### `charts.py` — `PriceChart(FigureCanvas)`
- Embeds matplotlib in Qt via `FigureCanvasQTAgg`
- Actual line: gray `#999999`, 1.8pt
- Predicted line: blue `#4A90E2`, 2.2pt
- Confidence band: shaded `#C8DEF5` polygon around predicted line (±1 std dev of tree predictions)
- Dashed vertical divider at forecast start
- X-axis: month labels; Y-axis: ₱ price labels
- Method `update_data(df, predicted, confidence_band)` redraws without recreating widget

### `dashboard.py` — `DashboardPage(QWidget)`
Three sub-regions composed horizontally:
1. **Center column** (flex): header row + chart + 3 mini-cards + simulation panel
2. **Right panel** (270px): pressure gauge + bands + key drivers + advisory output

Buttons:
- "↺ Recalculate" → emits `recalculate_requested`
- "⚡ Oil Shock +10%" → emits `oil_shock_requested`

### `pressure.py` — `PressureGaugePage(QWidget)` + `PressureGauge(QWidget)`
- Full Pressure Index page (sidebar nav: "Pressure Index")
- `PressureGauge` uses `paintEvent` + `QPainter` to draw a conic arc
  - Arc color transitions: blue → yellow → orange → red based on value
  - Center shows numeric value + "/100" label
- Page also shows the four band cards and a history list of last 5 index values

### `agent_graph.py` — `AgentGraphPage(QWidget)`
Uses `QGraphicsView` + `QGraphicsScene`.

Five nodes:
```
Oil Market ──┐
USD/PHP ──────→ ML Model → Prediction Output
Demand ───────┘
                   ↓
             Economic Index
```

Node style: white `QGraphicsRectItem` (120×50px, 8px radius), `#4A90E2` pen 1.5px, label centered.  
Edge style: `QGraphicsLineItem` gray `#CCCCCC`, with small arrowhead polygon at target end.  
Layout is static (fixed coordinates), no physics simulation.

### `main_window.py` — `MainWindow(QMainWindow)`
- Constructs sidebar + `QStackedWidget` with 4 pages
- Connects `sidebar.page_changed` → `stack.setCurrentIndex`
- Connects dashboard signals → `_on_recalculate()` and `_on_oil_shock()`
- `_on_recalculate()`: calls `model.predict()`, `preprocessing.compute_index()`, `explanation.generate()`, pushes results to dashboard via `dashboard.refresh(result)`
- `_on_oil_shock()`: multiplies current oil by 1.10, calls same pipeline

### `result` dict schema (passed to `dashboard.refresh`)
```python
{
    "predicted_price": float,       # ₱/L next period
    "current_price": float,         # ₱/L current period
    "trend": str,                   # "Rising" | "Stable" | "Falling"
    "confidence": float,            # 0–100
    "pressure_index": float,        # 0–100
    "pressure_band": str,           # "Stable" | "Rising" | "High" | "Critical"
    "oil_delta": float,             # z-score
    "usd_delta": float,             # z-score
    "demand_norm": float,           # 0–1
    "explanation": dict,            # keys: drivers, risk_badge, summary, advisory
    "scenarios": {                  # If-Then simulation
        "oil_shock": float,         # ±₱/L vs baseline
        "usd_shock": float,
        "demand_drop": float,
    },
    "confidence_band": tuple,       # (lower_prices, upper_prices) arrays for chart shading
}
```

### Note on `PressureGauge` reuse
`PressureGauge(QWidget)` is defined in `pressure.py` and instantiated in two places: the dashboard right panel (compact, 90px) and the full Pressure Index page (larger, 160px). Pass `size` as a constructor arg.

---

## 8. Rule-Based AI Explanation (`utils/explanation.py`)

### Key Drivers section
```python
if oil_delta > 0.5:    → "Crude Oil: +X% this period · Weight 50% · ↑ High"
elif oil_delta > 0.0:  → "Crude Oil: +X% · ↑ Rising"
else:                  → "Crude Oil: stable · → Neutral"

if usd_delta > 0.3:    → "USD/PHP: Y (+Z%) · ↑ Rising"
else:                  → "USD/PHP: Y · → Stable"

# demand_norm compared to 0.7 threshold
if demand_norm > 0.7:  → "Demand Index: Z/100 · ↑ High"
else:                  → "Demand Index: Z/100 · → Neutral"
```

### Risk badge
```python
if index > 60:  → "⚠ High Pressure — Price rise likely"
if index > 30:  → "⚡ Rising Pressure — Monitor closely"
else:           → "✓ Stable — No immediate risk"
```

### Advisory output
```python
if index > 60:  → suggested_action = "Refuel within 48 hours"
elif index > 30:→ suggested_action = "Monitor prices this week"
else:           → suggested_action = "No action needed — prices stable"
```

### Summary line (1–2 sentences)
Concatenated from triggered conditions. Example:
> "Oil and currency pressures are combining. Import costs have risen, pushing prices toward the next adjustment."

---

## 9. If-Then Simulation Panel

Displayed in the center-column bottom, full width, 3-column grid.

| Scenario            | Perturbation           | Display color |
|---------------------|------------------------|---------------|
| Oil prices +5%      | `oil_price × 1.05`     | Orange (price up) |
| USD strengthens +2% | `usd_php × 1.02`       | Orange (price up) |
| Demand drops 10pts  | `demand_index − 10`    | Green (price down) |

Each column shows: condition text → arrow → `±₱X.XX / L` (large) → proportional bar → 1-line explanation.

Simulation values recomputed on every recalculate/oil-shock event.

---

## 10. Dependencies

All already installed in the project venv:

| Package      | Version | Used for                        |
|--------------|---------|---------------------------------|
| PyQt6        | 6.10.0  | UI framework                    |
| matplotlib   | 3.8.4   | Embedded price chart            |
| scikit-learn | 1.7.2   | RandomForestRegressor           |
| numpy        | 1.26.4  | Data generation, array ops      |
| pandas       | 1.5.3   | DataFrame management            |

No pip installs required.

---

## 11. Startup Sequence

```
1. QApplication created
2. Data generated (data.generate_dataset())
3. Features preprocessed (preprocessing.build_features(df))
4. Model trained (model.train(X, y))
5. Initial prediction run (model.predict(last_row))
6. Pressure index computed (preprocessing.compute_index(last_row, df))
7. Explanation generated (explanation.generate(deltas, index))
8. If-Then scenarios computed (model.predict × 3 perturbed inputs)
9. MainWindow shown with all results populated
```

Steps 2–8 run synchronously before the window opens (dataset is small; expected <1 second).

---

## 12. Out of Scope

- Real-time data fetching (no network calls)
- Model persistence to disk
- CSV data loading (synthetic only, as decided)
- SQLite history storage
- Settings page functionality (nav item present, page shows placeholder)
- Packaging / installer

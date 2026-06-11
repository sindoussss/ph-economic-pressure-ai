# Chart Polish (SP2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the gas forecast chart the editorial "C" treatment (recent actuals + forecast + hero 90% calibrated band + soft grid) and turn the 3-sector card into honest magnitude bars — using only data already in the report.

**Architecture:** A pure `bar` magnitude added to `sector_forecast_rows` (tested seam); the 3-sector card renders bar widgets; the gas chart + feature-importances chart get editorial matplotlib styling inside the existing `try/except`. No new data plumbing, numbers unchanged.

**Tech Stack:** Python 3.10, PyQt6, matplotlib, pytest (offscreen Qt).

**Spec:** `docs/superpowers/specs/2026-06-11-chart-polish-design.md`.

**Confirmed anchors:**
- `ui/sector_forecast.py`: `sector_forecast_rows(gas, food, elec)` returns rows `{key,label,value,value_str,direction}`; `_SECTORS = [('gas',...),('food',...),('elec',...)]`.
- `ui/stage4_report.py::set_sector_forecasts` (~line 163): builds `_sector_holder_layout`; the row loop is `for r in sector_forecast_rows(gas, food, elec): lbl = QLabel(f"{r['label']}:  {arrows[r['direction']]}  {r['value_str']}") ...`. `arrows`/`colors` dicts defined just above. `QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel` are imported in this file.
- `ui/stage4_report._build_right` (~line 492): the gas chart block `if forecast_prices is not None and len(forecast_prices) > 0:` builds a `Figure`/`FigureCanvasQTAgg`, uses `_hs.conformal_halfwidth(_report)`, `cv_rmse`, `forecast_prices`, `month_labels`; feature-importances block follows (~line 524). Both inside `try/except`. `df` is a param of `_build_right` (has `gas_price`).
- Tests: `tests/test_stage4_sector.py` has `test_set_sector_forecasts_renders_card`; `tests/test_stage4_swarm.py` exercises `populate_swarm` (builds `_build_right` → the chart). Offscreen Qt via the modules' shims.

**Conventions:** Tests in `ph_economic_ai/tests/`, offscreen Qt. **Git hygiene:** commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):** `git checkout master && git pull && git checkout -b feature/chart-polish`

---

## Task 1: Honest `bar` magnitude in `sector_forecast`

**Files:** Modify `ph_economic_ai/ui/sector_forecast.py`; Test `ph_economic_ai/tests/test_sector_bar.py`

- [ ] **Step 1: Failing test** — create `ph_economic_ai/tests/test_sector_bar.py`:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai.ui.sector_forecast import sector_forecast_rows


def _by_key(**kw):
    return {r['key']: r for r in sector_forecast_rows(**kw)}


def test_bar_fraction_per_sector():
    r = _by_key(gas=-1.8, food=-2.6, elec=0.18)
    assert abs(r['gas']['bar'] - 0.36) < 1e-9     # 1.8 / 5.0
    assert abs(r['food']['bar'] - 0.52) < 1e-9    # 2.6 / 5.0
    assert abs(r['elec']['bar'] - 0.09) < 1e-9    # 0.18 / 2.0


def test_bar_clamps_and_none():
    r = _by_key(gas=100.0, food=None)
    assert r['gas']['bar'] == 1.0                  # clamped
    assert r['food']['bar'] == 0.0 and r['food']['direction'] == 'na'
    # existing fields unchanged
    assert r['gas']['value_str'] == '+100.00 ₱/L'
```

- [ ] **Step 2: Run → fails** (`python -m pytest ph_economic_ai/tests/test_sector_bar.py -v`) — `KeyError: 'bar'`.

- [ ] **Step 3: Implement** — in `ph_economic_ai/ui/sector_forecast.py`:
(a) add after `_SECTORS`:
```python
_BAR_SCALE = {'gas': 5.0, 'food': 5.0, 'elec': 2.0}   # per-sector "full bar" move (display only)
```
(b) in the `sector_forecast_rows` loop, add a `bar` key to the appended dict:
```python
            'direction': _direction(v),
            'bar': 0.0 if v is None else min(abs(v) / _BAR_SCALE[key], 1.0),
```

- [ ] **Step 4: Run → passes.** **Step 5: Commit** (`sector_forecast.py` + test) — `feat(ui): honest per-sector bar magnitude for the forecast card`.

---

## Task 2: Render the 3-sector card as magnitude bars

**Files:** Modify `ph_economic_ai/ui/stage4_report.py`; Test `ph_economic_ai/tests/test_stage4_sector.py` (append)

- [ ] **Step 1: Append the failing test**
```python
def test_sector_card_renders_bars(app):
    from PyQt6.QtWidgets import QFrame, QLabel
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    p = Stage4ReportPanel()
    p.set_sector_forecasts(-1.8, -2.6, 0.18)
    texts = ' || '.join(l.text() for l in p.findChildren(QLabel))
    assert '1.80' in texts and '2.60' in texts and '0.1800' in texts
    assert len(p._sector_holder.findChildren(QFrame)) >= 3   # a bar track per sector
```
(Reuse the `app` fixture already in `test_stage4_sector.py`.)

- [ ] **Step 2: Run → fails** (no `QFrame` bars yet): `python -m pytest ph_economic_ai/tests/test_stage4_sector.py::test_sector_card_renders_bars -v`.

- [ ] **Step 3: Implement** — in `set_sector_forecasts`, replace the row loop:
```python
            for r in sector_forecast_rows(gas, food, elec):
                lbl = QLabel(f"{r['label']}:  {arrows[r['direction']]}  {r['value_str']}")
                lbl.setStyleSheet(f"font-size:12px;font-weight:600;color:{colors[r['direction']]};")
                self._sector_holder_layout.addWidget(lbl)
```
with:
```python
            for r in sector_forecast_rows(gas, food, elec):
                color = colors[r['direction']]
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.setSpacing(8)
                name = QLabel(f"{arrows[r['direction']]}  {r['label']}")
                name.setFixedWidth(118)
                name.setStyleSheet(f'font-size:11px;font-weight:600;color:{color};')
                rl.addWidget(name)
                track = QFrame()
                track.setFixedSize(120, 8)
                track.setStyleSheet('background:#EEF0F4;border-radius:4px;')
                fill = QFrame(track)
                w = max(2, int(120 * r['bar'])) if r['bar'] > 0 else 0
                fill.setGeometry(0, 0, w, 8)
                fill.setStyleSheet(f'background:{color};border-radius:4px;')
                rl.addWidget(track)
                val = QLabel(r['value_str'])
                val.setStyleSheet(f'font-size:11px;font-weight:600;color:{color};')
                rl.addWidget(val)
                rl.addStretch()
                self._sector_holder_layout.addWidget(row)
```
(Keep the `NEXT-MONTH SECTOR FORECAST` title + `exploratory — not validated` sub-caption + the surrounding `try/except` unchanged.)

- [ ] **Step 4: Run → passes**: `python -m pytest ph_economic_ai/tests/test_stage4_sector.py -v` (existing + new).

- [ ] **Step 5: Commit** (`stage4_report.py` + `test_stage4_sector.py`) — `feat(ui): 3-sector forecast card as honest magnitude bars`.

---

## Task 3: Editorial restyle of the gas forecast + feature-importances charts

**Files:** Modify `ph_economic_ai/ui/stage4_report.py`. Verification: existing build tests.

- [ ] **Step 1: Replace the gas forecast chart block** (the `if forecast_prices is not None and len(forecast_prices) > 0:` block in `_build_right`) with:
```python
        if forecast_prices is not None and len(forecast_prices) > 0:
            try:
                import numpy as _np
                now = datetime.now()
                fp = _np.asarray(forecast_prices, dtype=float)
                n = len(fp)
                month_labels = [
                    calendar.month_abbr[(now.month - 1 + i) % 12 + 1]
                    for i in range(1, n + 1)
                ]
                hist = (df['gas_price'].dropna().tail(6).tolist()
                        if 'gas_price' in getattr(df, 'columns', []) else [])
                kh = len(hist)
                fig = Figure(figsize=(5, 2.4), facecolor='#FBFBFA')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#FBFBFA')
                fx = list(range(kh, kh + n))
                if kh:
                    ax.plot(range(kh), hist, color='#9AA1AC', linewidth=1.6)
                    ax.plot([kh - 1, kh], [hist[-1], fp[0]], color='#1C1E26', linewidth=2)
                    ax.axvline(kh - 0.5, color='#D1D5DB', linewidth=1.0, linestyle='--')
                ax.plot(fx, fp, color='#1C1E26', linewidth=2)
                _band = _hs.conformal_halfwidth(_report) or cv_rmse
                ax.fill_between(fx, fp - _band, fp + _band, alpha=0.28,
                                color='#1C1E26', linewidth=0)
                ax.axvspan((kh - 0.5) if kh else 0, kh + n - 1, color='#1C1E26', alpha=0.03)
                for _sp in ('top', 'right'):
                    ax.spines[_sp].set_visible(False)
                for _sp in ('left', 'bottom'):
                    ax.spines[_sp].set_color('#E5E7EB')
                ax.grid(axis='y', color='#EEEEEE', linewidth=0.6)
                ax.set_axisbelow(True)
                ax.set_xticks(fx)
                ax.set_xticklabels(month_labels, fontsize=7, color='#9AA1AC')
                ax.tick_params(axis='y', labelsize=7, colors='#9AA1AC')
                fig.tight_layout(pad=1.0)
                canvas = FigureCanvasQTAgg(fig)
                canvas.setFixedHeight(210)
                cl.addWidget(canvas)
                _bcap = self._muted('90% calibrated interval (conformal)'
                                    if _hs.conformal_halfwidth(_report) is not None
                                    else '±cross-val RMSE (uncalibrated)')
                _bcap.setWordWrap(True)
                cl.addWidget(_bcap)
            except Exception:
                pass
```

- [ ] **Step 2: Restyle the feature-importances chart to match** — in the feature-importances `try` block, after `fi_ax.barh(...)` (and before its canvas is created), add editorial axes styling:
```python
                fi_fig.set_facecolor('#FBFBFA')
                fi_ax.set_facecolor('#FBFBFA')
                for _sp in ('top', 'right'):
                    fi_ax.spines[_sp].set_visible(False)
                for _sp in ('left', 'bottom'):
                    fi_ax.spines[_sp].set_color('#E5E7EB')
                fi_ax.tick_params(labelsize=7, colors='#6B7280')
                fi_ax.grid(axis='x', color='#EEEEEE', linewidth=0.6)
                fi_ax.set_axisbelow(True)
```
(Place these lines among the existing `fi_ax` setup; keep the existing bar colour `#1C1E26` and the rest of the block intact. If the variable names differ, read the block and adapt — the change is styling only.)

- [ ] **Step 3: Build-without-crash verification**

Run: `python -m pytest ph_economic_ai/tests/test_stage4_swarm.py ph_economic_ai/tests/test_stage4_sector.py -q`
Expected: pass (the chart restyle is inside the existing `try/except`; `populate_swarm` builds `_build_right` → the chart). If a test fails, the styling code raised outside the guard — fix to stay inside `try`.

- [ ] **Step 4: Commit** (`stage4_report.py`) — `feat(ui): editorial gas forecast chart (recent actuals + hero band + soft grid)`.

---

## Final verification
- [ ] `python -m pytest ph_economic_ai/tests/ -q` → all pass.
- [ ] Manual (GUI): run a sim → the 3-sector card shows magnitude bars; the report Outputs gas chart shows recent gray actuals → dark forecast with a dashed divider, a prominent 90% band, soft gridlines, trimmed spines; the feature-importances chart matches.

---

## Self-Review (completed by plan author)
**Spec coverage:** §3.1 `bar` field (per-sector scale, clamp, None→0) → Task 1; §3.2 sector card bar rendering → Task 2; §3.3 gas chart (actuals + hero band + zone + grid + spines) and feature-importances restyle → Task 3. §5 testing (bar fractions; card renders bars; chart build-without-crash) → Tasks 1–3. §2 non-negotiables: bar is display-only/per-sector (Task 1 scale dict), chart restyle styling-only + caption/labels kept (Task 3 keeps `_bcap` + the "exploratory" title in `set_sector_forecasts` untouched).
**Placeholder scan:** none — complete code; Task 3 Step 2 notes "adapt if variable names differ" but gives the exact styling lines.
**Type consistency:** `sector_forecast_rows` rows gain `bar` (float 0–1) used in Task 2's `int(120 * r['bar'])`. `_BAR_SCALE` keys (`gas/food/elec`) match `_SECTORS` keys. Task 2 uses `colors`/`arrows` dicts already defined above the loop and `QWidget/QHBoxLayout/QFrame/QLabel` already imported. Task 3 uses `fp = np.asarray(forecast_prices)` so `fp - _band` is safe whether the input was a list or array; `df['gas_price']` guarded by the `columns` check; everything inside the existing `try/except`.
````

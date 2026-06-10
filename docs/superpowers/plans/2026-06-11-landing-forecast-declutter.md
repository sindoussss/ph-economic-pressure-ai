# Landing Forecast Declutter + Confidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the landing's latest forecast a date + agreement%, and de-duplicate the history (show only prior runs as a "FUEL TRACK RECORD"), so the forecast area reads clean instead of cramped/redundant.

**Architecture:** All changes are in `ui/landing.py`: make the latest heading dynamic (date + agreement from the latest run), populate the fuel history from `runs[1:4]` (prior runs only), relabel "confidence" → "agreement", and factor a small `_fmt_date` helper.

**Tech Stack:** Python 3.10, PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-06-11-landing-forecast-declutter-design.md`.

**Prereqs (on `master`):** in `ui/landing.py`:
- `_build_recent_strip`: line `latest_head = QLabel('LATEST FORECAST  ·  exploratory')`, then `latest_head.setStyleSheet(...)`, then `cl.addWidget(latest_head)`; later `head = QLabel('RECENT FUEL FORECASTS')`. Layouts `self._latest_row` and `self._runs_row` already exist.
- `_refresh_recent_runs`: fetches `runs = self._store.get_recent_runs(limit=4)`; populates `self._latest_row` from `runs[0]`; then clears `self._runs_row`, shows "No simulations on record yet." if `not runs`, else `for r in runs:` builds `_build_run_tile(r)`.
- `_build_run_tile(run)`: formats date via `datetime.fromisoformat(ts).strftime('%b %d')` (fallback `ts[:10]`); builds `f'{conf}% confidence'` in `sub_parts`. Run dict has `run_id`, `timestamp`, `final_estimate`, `confidence_pct`.
- `refresh_recent()` is the public refresh wrapper (calls `_refresh_recent_runs`). `TEXT_3`, `UP`, `DOWN`, `datetime`, `QLabel`, `QWidget`, `QVBoxLayout`, `QHBoxLayout` are imported.
- `LandingPanel(store=None, parent=None)`. Test harness: `ph_economic_ai/tests/test_landing_latest.py` (offscreen Qt, `_FakeStore`, `app` fixture).

**Conventions:**
- Tests in `ph_economic_ai/tests/`. **Git hygiene:** staging clean; commit ONLY listed paths; NEVER `git add -A`/`.`; `git status --short` first; do NOT stage `accuracy_report.json`.

**Task 0 (branch):**
```bash
git checkout master && git pull && git checkout -b feature/landing-forecast-declutter
```

---

## Task 1: Dynamic latest heading + de-duplicated track record

**Files:**
- Modify: `ph_economic_ai/ui/landing.py`
- Modify: `ph_economic_ai/tests/test_landing_latest.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `ph_economic_ai/tests/test_landing_latest.py` (the file already has the `app` fixture, `_FakeStore`, and `QLabel` import):

```python
def _run(rid, ts, gas, conf, food=None, elec=None):
    return {'run_id': rid, 'timestamp': ts, 'final_estimate': gas,
            'confidence_pct': conf, 'food_estimate': food,
            'electricity_estimate': elec, 'actual_price_change': None}


def test_latest_heading_shows_date_and_agreement(app):
    from ph_economic_ai.ui.landing import LandingPanel
    runs = [_run(6, '2026-06-10T00:00:00+00:00', -1.8, 72, -2.6, 0.18),
            _run(5, '2026-06-10T00:00:00+00:00', -1.5, 50),
            _run(4, '2026-06-09T00:00:00+00:00', -2.4, 54)]
    panel = LandingPanel(store=_FakeStore(runs))
    panel.refresh_recent()
    texts = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'LATEST FORECAST' in texts
    assert '72% agreement' in texts          # latest run's agreement on the latest block
    assert 'Jun 10' in texts                  # latest run's date


def test_track_record_excludes_latest_run(app):
    from ph_economic_ai.ui.landing import LandingPanel
    runs = [_run(6, '2026-06-10T00:00:00+00:00', -1.8, 72, -2.6, 0.18),
            _run(5, '2026-06-10T00:00:00+00:00', -1.5, 50),
            _run(4, '2026-06-09T00:00:00+00:00', -2.4, 54)]
    panel = LandingPanel(store=_FakeStore(runs))
    panel.refresh_recent()
    labels = [l.text() for l in panel.findChildren(QLabel)]
    text = ' || '.join(labels)
    assert 'FUEL TRACK RECORD' in text
    assert any(t.startswith('#5') for t in labels)   # a prior run is in the record
    assert not any(t.startswith('#6') for t in labels)  # the latest is NOT duplicated
    assert 'agreement' in text and 'confidence' not in text  # relabelled


def test_single_run_track_record_placeholder(app):
    from ph_economic_ai.ui.landing import LandingPanel
    panel = LandingPanel(store=_FakeStore([_run(6, '2026-06-10T00:00:00+00:00', -1.8, 72, -2.6, 0.18)]))
    panel.refresh_recent()  # must not raise
    text = ' || '.join(l.text() for l in panel.findChildren(QLabel))
    assert 'LATEST FORECAST' in text            # latest still shown
    assert 'No simulations on record yet.' in text  # no prior runs -> placeholder
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_landing_latest.py -k "heading or track_record or single_run" -v`
Expected: FAIL — `'72% agreement'` / `'FUEL TRACK RECORD'` not found (and `'#6'` still present in the record).

- [ ] **Step 3: Make the latest heading a member + rename the fuel heading**

In `_build_recent_strip`, change:
```python
        latest_head = QLabel('LATEST FORECAST  ·  exploratory')
        latest_head.setStyleSheet(
```
to:
```python
        self._latest_head = QLabel('LATEST FORECAST  ·  exploratory')
        self._latest_head.setStyleSheet(
```
and change `cl.addWidget(latest_head)` to `cl.addWidget(self._latest_head)`.
Then change `head = QLabel('RECENT FUEL FORECASTS')` to `head = QLabel('FUEL TRACK RECORD')`.

- [ ] **Step 4: Add the `_fmt_date` helper**

Add this method to `LandingPanel` (e.g. just before `_build_run_tile`):
```python
    @staticmethod
    def _fmt_date(ts: str) -> str:
        ts = ts or ''
        try:
            return datetime.fromisoformat(ts).strftime('%b %d')
        except Exception:
            return ts[:10]
```

- [ ] **Step 5: Update `_refresh_recent_runs` (heading text + prior-runs-only history)**

Replace the body from the latest-row population through the run-tile loop. Specifically:

(a) Right after `self._latest_row.addStretch()` (end of the latest-row block), set the heading text:
```python
        if runs and runs[0].get('confidence_pct') is not None:
            self._latest_head.setText(
                f"LATEST FORECAST  ·  {self._fmt_date(runs[0].get('timestamp'))}"
                f"  ·  {runs[0]['confidence_pct']}% agreement  ·  exploratory")
        elif runs:
            self._latest_head.setText(
                f"LATEST FORECAST  ·  {self._fmt_date(runs[0].get('timestamp'))}  ·  exploratory")
        else:
            self._latest_head.setText('LATEST FORECAST  ·  exploratory')
```

(b) In the fuel-history section, replace:
```python
        if not runs:
            empty = QLabel('No simulations on record yet.')
            empty.setStyleSheet(f'color:{TEXT_3};font-size:12px;')
            self._runs_row.addWidget(empty)
            self._runs_row.addStretch()
            return

        for r in runs:
            self._runs_row.addWidget(self._build_run_tile(r))
        self._runs_row.addStretch()
```
with (prior runs only — `runs[1:4]`):
```python
        prior = runs[1:4]
        if not prior:
            empty = QLabel('No simulations on record yet.')
            empty.setStyleSheet(f'color:{TEXT_3};font-size:12px;')
            self._runs_row.addWidget(empty)
            self._runs_row.addStretch()
            return

        for r in prior:
            self._runs_row.addWidget(self._build_run_tile(r))
        self._runs_row.addStretch()
```

- [ ] **Step 6: Relabel the per-tile "confidence" → "agreement"**

In `_build_run_tile`, change:
```python
            sub_parts.append(f'{conf}% confidence')
```
to:
```python
            sub_parts.append(f'{conf}% agreement')
```
(Optional consistency: `_build_run_tile` may also use `self._fmt_date(ts)` in place of its inline try/except — not required.)

- [ ] **Step 7: Run to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_landing_latest.py -v`
Expected: PASS (existing + 3 new).

- [ ] **Step 8: Window smoke**

Run: `python -m pytest ph_economic_ai/tests/test_main_window.py -q`
Expected: all pass (no new failures).

- [ ] **Step 9: Commit**

```bash
git add ph_economic_ai/ui/landing.py ph_economic_ai/tests/test_landing_latest.py
git commit -m "feat(ui): landing latest forecast shows date + agreement; history = prior runs only"
```

---

## Final verification

- [ ] **Full suite**

Run: `python -m pytest ph_economic_ai/tests/ -q`
Expected: all pass (adds passing tests).

- [ ] **Manual visual check (optional, GUI session)**

Open Home after a run: the latest block reads "LATEST FORECAST · <date> · <N>% agreement · exploratory" with Gas/Food/Electricity; below it "FUEL TRACK RECORD" lists the *prior* runs only (the latest no longer duplicated); tiles say "agreement", not "confidence".

---

## Self-Review (completed by plan author)

**Spec coverage:** §4.1 dynamic `self._latest_head` + rename to "FUEL TRACK RECORD" → Steps 3, 5a. §4.2 heading text (date + agreement) + `runs[1:4]` prior-only + placeholder → Steps 4, 5. §4.3 "confidence" → "agreement" → Step 6. §6 edge cases (no runs / one run / None conf/ts) → Steps 4, 5. §7 testing (heading date+agreement; track record excludes latest + relabelled; single-run placeholder) → Step 1.

**Placeholder scan:** none — every step has complete code/exact strings.

**Type consistency:** `self._latest_head` created in `_build_recent_strip` (Step 3) before use in `_refresh_recent_runs` (Step 5a). `_fmt_date(ts)` (Step 4) used in Step 5a (and optionally Step 6). `runs[1:4]` slice keyed on the same run dicts; `confidence_pct`/`timestamp` accessed via `.get`. The test asserts the latest `run_id` (`#6`) is absent from the run-tile labels and a prior (`#5`) is present, directly verifying de-duplication; `'confidence' not in text` verifies the relabel. `_build_run_tile`'s `#{rid}  ·  {d_str}` head is what `t.startswith('#5')` matches.
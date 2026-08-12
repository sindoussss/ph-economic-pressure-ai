# Food Sub-Category App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Forum's food debate reports a signed read for six PSA sub-categories (rice, meat, fish, dairy & eggs, vegetables, sugar) alongside its existing blended estimate, in the same debate, with no new LLM calls. The Monitor page's Food card shows the breakdown; compact cards elsewhere keep one number, honestly relabeled as a basket average.

**Architecture:** `SectorReading` gains a `subcategories` field. The food judge's closing prompt gains six worked-example lines. A new, self-contained `_extract_category_percents` parses them, reusing the existing `_MAX_REALISTIC_FOOD_PCT` ceiling. `pressure_monitor.py`'s Food card renders the breakdown; `sector_forecast.py`'s shared label feeds both Report's stat card and Landing's tiles from one place.

**Tech Stack:** Python, PyQt6, pytest (offscreen Qt platform for tests — do not force `QT_QPA_PLATFORM=offscreen` when visually verifying by hand; see this session's own finding that it renders all text as tofu boxes on this machine).

## Global Constraints

- No new LLM-call multiplication: the six category lines are additional output from the *existing* food judge call, not a new call per category.
- A category with no parseable line is **absent** from the result — never `0.0`, never copied from another category or the blended estimate.
- Every signed value renders with `{v:+.1f}%` (or equivalent explicit sign format) — never a literal `+` concatenated with the value. This is the exact mistake fixed twice already this session (`RSK-053`, the BSP banner; the general rule every signed-number surface in this codebase follows).
- `SectorReading`'s new field must be `kw_only=True`, matching `estimates`' own existing convention — not field-ordering alone, which is a weaker guarantee.
- `economy_overview.py`'s "Food Index (derived)" card is **out of scope** — confirmed by reading its own code comment ("Food and electricity are deterministic pass-through transforms of the gas price, not independent predictions"), it is a different, gas-derived quantity, not the Forum's food estimate. Relabeling it "(basket avg)" would be incorrect. Do not touch it in this plan.

---

### Task 1: `SectorReading.subcategories` field

**Files:**
- Modify: `ph_economic_ai/engine/pressure_brief.py`
- Test: `ph_economic_ai/tests/test_trust_on_screen.py`

**Interfaces:**
- Produces: `SectorReading.subcategories: dict[str, float]` (kw_only, default `{}`) — consumed by Task 3 (forum.py, sets it) and Task 4 (pressure_monitor.py, reads it).

- [ ] **Step 1: Write the failing test**

Add to `ph_economic_ai/tests/test_trust_on_screen.py`, immediately after `test_the_estimates_field_cannot_be_hit_positionally` (around line 227):

```python
def test_the_subcategories_field_cannot_be_hit_positionally():
    """Same guard as the estimates field above, for subcategories -- kw_only=True
    is what actually protects a dataclass built positionally elsewhere in this
    codebase, not field-ordering alone."""
    from ph_economic_ai.engine.pressure_brief import SectorReading
    with pytest.raises(TypeError):
        SectorReading('gas', 'rising', 1.0, '₱/L', 100, 0, [], [], [1.0], {'rice': 0.1})
    r = SectorReading('gas', 'rising', 1.0, '₱/L', 100, 90, ['d'], ['s'], estimates=[1.0])
    assert r.subcategories == {}
    r2 = SectorReading('gas', 'rising', 1.0, '₱/L', 100, 90, ['d'], ['s'],
                       estimates=[1.0], subcategories={'rice': 0.1})
    assert r2.subcategories == {'rice': 0.1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_trust_on_screen.py::test_the_subcategories_field_cannot_be_hit_positionally -v`
Expected: FAIL with `TypeError: SectorReading.__init__() got an unexpected keyword argument 'subcategories'` (the second assertion in the test body; the first `pytest.raises(TypeError)` block will actually pass by accident since 10 positional args already fails today — the real signal is the later `AttributeError`/`TypeError` on `subcategories=`)

- [ ] **Step 3: Add the field**

In `ph_economic_ai/engine/pressure_brief.py`, in the `SectorReading` dataclass, immediately after the `estimates` field (line 43):

```python
    estimates: list[float] = field(default_factory=list, kw_only=True)
    # Per-PSA-sub-category signed reads (rice, meat, fish, dairy_eggs,
    # vegetables, sugar), when the food judge produced them -- keys present
    # only for categories with an actual parsed value; a category with no
    # parseable line is absent here, never 0.0. Empty for gas/electricity,
    # which have no sub-categories. kw_only for the same reason `estimates`
    # is: a positionally-built call site must not silently reassign this.
    subcategories: dict[str, float] = field(default_factory=dict, kw_only=True)
```

`to_dict()` needs no change — it already uses `asdict(self)`, which picks up any dataclass field automatically.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_trust_on_screen.py -v`
Expected: All tests PASS, including the new one and every pre-existing test in the file (this file has broad coverage of `SectorReading`/`honesty.py` — a regression here would show up immediately).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/pressure_brief.py ph_economic_ai/tests/test_trust_on_screen.py
git commit -m "feat(engine): SectorReading gains a subcategories field

kw_only=True, matching the existing estimates field's own convention.
Empty by default; to_dict() picks it up automatically via asdict()."
```

---

### Task 2: `_extract_category_percents` in `debate.py`

**Files:**
- Modify: `ph_economic_ai/engine/debate.py`
- Test: `ph_economic_ai/tests/test_estimate_extraction.py`

**Interfaces:**
- Consumes: `_TOLERANCE_BAND_RE`, `_MAX_REALISTIC_FOOD_PCT` (both existing, unmodified — reused, not duplicated).
- Produces: `_CATEGORY_LABELS: dict[str, str]` (category key → prompt label, e.g. `'dairy_eggs': 'DAIRY_EGGS'`); `_extract_category_percents(text: str) -> dict[str, float]` — used by Task 3.

- [ ] **Step 1: Write the failing tests**

Add to `ph_economic_ai/tests/test_estimate_extraction.py`, after `test_no_estimate_anywhere_returns_none` (end of file):

```python
def test_category_percents_parses_all_six():
    from ph_economic_ai.engine.debate import _extract_category_percents
    text = (
        'RICE: +0.2%\nMEAT: -0.3%\nFISH: +0.8%\n'
        'DAIRY_EGGS: +0.0%\nVEGETABLES: -0.1%\nSUGAR: +0.0%\n'
    )
    result = _extract_category_percents(text)
    assert result == {
        'rice': 0.2, 'meat': -0.3, 'fish': 0.8,
        'dairy_eggs': 0.0, 'vegetables': -0.1, 'sugar': 0.0,
    }


def test_category_percents_missing_category_is_absent_not_zero():
    from ph_economic_ai.engine.debate import _extract_category_percents
    text = 'RICE: +0.2%\nMEAT: -0.3%\n'  # only two of six
    result = _extract_category_percents(text)
    assert result == {'rice': 0.2, 'meat': -0.3}
    assert 'fish' not in result
    assert 'sugar' not in result


def test_category_percents_rejects_implausible_value():
    from ph_economic_ai.engine.debate import _extract_category_percents, _MAX_REALISTIC_FOOD_PCT
    text = f'RICE: +{_MAX_REALISTIC_FOOD_PCT + 5:.1f}%\nMEAT: -0.3%\n'
    result = _extract_category_percents(text)
    assert 'rice' not in result  # implausible, dropped
    assert result['meat'] == -0.3


def test_category_percents_takes_the_last_line_per_category():
    from ph_economic_ai.engine.debate import _extract_category_percents
    text = 'RICE: +0.5%\nOn reflection, RICE: +0.2%\n'
    result = _extract_category_percents(text)
    assert result['rice'] == pytest.approx(0.2)


def test_category_percents_empty_text_returns_empty_dict():
    from ph_economic_ai.engine.debate import _extract_category_percents
    assert _extract_category_percents('The outlook is broadly stable.') == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_estimate_extraction.py::test_category_percents_parses_all_six -v`
Expected: FAIL with `ImportError: cannot import name '_extract_category_percents'`

- [ ] **Step 3: Write the implementation**

In `ph_economic_ai/engine/debate.py`, immediately after `_extract_percent` (which ends around line 626), add:

```python
#: Food sub-category prompt labels, in the order they should appear in the
#: judge's closing instruction. Six PSA CPI sub-categories confirmed live
#: against openstat.psa.gov.ph during this feature's brainstorming (see
#: docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md).
_CATEGORY_LABELS = {
    'rice': 'RICE', 'meat': 'MEAT', 'fish': 'FISH',
    'dairy_eggs': 'DAIRY_EGGS', 'vegetables': 'VEGETABLES', 'sugar': 'SUGAR',
}


def _extract_category_percents(text: str) -> dict[str, float]:
    """One signed-percent line per food sub-category (RICE:, MEAT:, ...).

    Deliberately a separate, self-contained parser rather than a refactor of
    `_last_estimate_match` (which every other extractor in this file depends
    on and is heavily tested) -- this reuses its two safe-to-share pieces,
    `_TOLERANCE_BAND_RE` and `_MAX_REALISTIC_FOOD_PCT`, without touching the
    well-tested ESTIMATE-line parser itself.

    A category whose line is missing, unparseable, or out of bound is simply
    absent from the returned dict -- never present as 0.0 or copied from
    another category's value. Takes the LAST match per category, same reason
    every other extractor in this file does: agents restate and revise.
    """
    cleaned = _TOLERANCE_BAND_RE.sub(' ', text)
    result: dict[str, float] = {}
    for category, label in _CATEGORY_LABELS.items():
        hits = re.findall(
            rf'{label}\s*:\s*\**\s*([+\-])?\s*(\d+\.?\d*)\s*%',
            cleaned, flags=re.IGNORECASE,
        )
        if not hits:
            continue
        sign, raw = hits[-1]
        value = (-1 if sign == '-' else 1) * float(raw)
        if abs(value) <= _MAX_REALISTIC_FOOD_PCT:
            result[category] = value
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_estimate_extraction.py -v`
Expected: All tests PASS (5 new tests plus every pre-existing test in the file).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/debate.py ph_economic_ai/tests/test_estimate_extraction.py
git commit -m "feat(debate): _extract_category_percents for food sub-categories

Self-contained parser reusing _TOLERANCE_BAND_RE and _MAX_REALISTIC_FOOD_PCT
without touching the well-tested _last_estimate_match/_extract_percent path.
A missing or implausible category is absent from the result, never 0.0."
```

---

### Task 3: Wire category parsing into the Forum's food judge

**Files:**
- Modify: `ph_economic_ai/engine/forum.py`
- Test: `ph_economic_ai/tests/test_forum.py`

**Interfaces:**
- Consumes: `debate._CATEGORY_LABELS`, `debate._extract_category_percents` (Task 2); `SectorReading.subcategories` (Task 1).
- Produces: `Forum._judge_sector` now returns a 3-tuple `(estimate, statement, subcategories)` instead of 2 — the one existing call site (`_run_sector`) and the one existing return-type assumption in tests must be updated in the same commit.

- [ ] **Step 1: Write the failing tests**

Add to `ph_economic_ai/tests/test_forum.py`, immediately after `test_estimate_line_is_an_instruction_not_a_template` (around line 495):

```python
def test_food_category_lines_are_instructions_not_templates():
    """Same guard as the ESTIMATE line above, for the six new food
    sub-category lines: each must carry a worked example and must not leave
    a copyable 'X.X' placeholder outside its own worked example."""
    from ph_economic_ai.engine.forum import _FOOD_CATEGORY_LINES
    from ph_economic_ai.engine.debate import _CATEGORY_LABELS, _extract_category_percents
    assert set(_FOOD_CATEGORY_LINES) == set(_CATEGORY_LABELS)
    for category, line in _FOOD_CATEGORY_LINES.items():
        assert 'worked example' in line
        assert 'your own number' in line
        import re
        for m in re.findall(rf'"({_CATEGORY_LABELS[category]}:[^"]+)"', line):
            assert category in _extract_category_percents(m), f'{category}: {m!r}'


def test_judge_sector_returns_subcategories_for_food(monkeypatch):
    """The judge's synthesis, for food, must feed _extract_category_percents
    and thread the result through -- not just the blended estimate."""
    import ph_economic_ai.engine.forum as forum_mod
    from ph_economic_ai.engine.auto_assemble import SectorContext

    def fake_complete(msgs, **kw):
        return ('Prices are broadly steady with a modest rice uptick.\n'
                'RICE: +0.3%\nMEAT: +0.0%\nFISH: -0.2%\n'
                'DAIRY_EGGS: +0.0%\nVEGETABLES: +0.1%\nSUGAR: +0.0%\n'
                'ESTIMATE: +0.1%')

    monkeypatch.setattr(forum_mod.llm, 'complete', fake_complete)
    f = forum_mod.Forum(rag=None, contexts=[], as_of='2026-08-12', window='this_week')
    f._rag = type('R', (), {'query': lambda self, *a, **kw: []})()
    ctx = SectorContext(sector='food', unit='%', verdict_note='exploratory',
                        anchor=None, social_counts={})
    estimate, statement, subcategories = f._judge_sector(ctx, finals=[])
    assert estimate == pytest.approx(0.1)
    assert subcategories == {'rice': 0.3, 'meat': 0.0, 'fish': -0.2,
                             'dairy_eggs': 0.0, 'vegetables': 0.1, 'sugar': 0.0}


def test_judge_sector_returns_empty_subcategories_for_gas(monkeypatch):
    """Gas and electricity have no PSA sub-categories -- the judge must not
    try to parse category lines that were never asked for."""
    import ph_economic_ai.engine.forum as forum_mod
    from ph_economic_ai.engine.auto_assemble import SectorContext

    monkeypatch.setattr(forum_mod.llm, 'complete',
                        lambda msgs, **kw: 'Steady. ESTIMATE: +0.10/L')
    f = forum_mod.Forum(rag=None, contexts=[], as_of='2026-08-12', window='this_week')
    f._rag = type('R', (), {'query': lambda self, *a, **kw: []})()
    ctx = SectorContext(sector='gas', unit='PHP/L', verdict_note='exploratory',
                        anchor=None, social_counts={})
    estimate, statement, subcategories = f._judge_sector(ctx, finals=[])
    assert subcategories == {}
```

**Note for the implementer:** check `SectorContext`'s actual field names in `ph_economic_ai/engine/auto_assemble.py` before running this step — the test above assumes `sector`, `unit`, `verdict_note`, `anchor`, `social_counts` based on `forum.py`'s own usage (`ctx.sector`, `ctx.unit`, `ctx.verdict_note`, `ctx.anchor`, `ctx.social_counts` all appear in `forum.py`'s existing code), but confirm the exact constructor signature directly (`grep -n "class SectorContext" -A 15 ph_economic_ai/engine/auto_assemble.py`) and adjust the test's `SectorContext(...)` call if any field is missing or named differently — do not guess a field into existence.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest ph_economic_ai/tests/test_forum.py::test_food_category_lines_are_instructions_not_templates ph_economic_ai/tests/test_forum.py::test_judge_sector_returns_subcategories_for_food ph_economic_ai/tests/test_forum.py::test_judge_sector_returns_empty_subcategories_for_gas -v`
Expected: FAIL — first with `ImportError: cannot import name '_FOOD_CATEGORY_LINES'`, others with `ValueError: too many values to unpack` (the current `_judge_sector` returns a 2-tuple).

- [ ] **Step 3: Add `_FOOD_CATEGORY_LINES` and wire it into the judge prompt and parsing**

In `ph_economic_ai/engine/forum.py`, add the import and the new dict near the top (after the existing `from ph_economic_ai.engine.debate import (...)` block, which already imports several debate.py names):

```python
from ph_economic_ai.engine.debate import _CATEGORY_LABELS, _extract_category_percents
```

Add after `_EST_LINE = ESTIMATE_LINE` (around line 132):

```python
#: Six worked-example lines the food judge must append after its ESTIMATE
#: line, one per PSA sub-category. Same instructions-not-template pattern
#: _EST_LINE already established (RSK-012's lesson): a small model copies a
#: bare template verbatim, so every line needs its own worked example.
_FOOD_CATEGORY_LINES = {
    'rice': ('RICE: <the percent month-on-month CHANGE you expect for rice specifically, '
             'signed> (worked example: "RICE: +0.2%" or "RICE: -0.1%". Write your own '
             'number; never write X.X.)'),
    'meat': ('MEAT: <the percent month-on-month CHANGE you expect for meat specifically, '
             'signed> (worked example: "MEAT: +0.3%" or "MEAT: -0.2%". Write your own '
             'number; never write X.X.)'),
    'fish': ('FISH: <the percent month-on-month CHANGE you expect for fish and seafood '
             'specifically, signed> (worked example: "FISH: +0.8%" or "FISH: -0.4%". '
             'Write your own number; never write X.X.)'),
    'dairy_eggs': ('DAIRY_EGGS: <the percent month-on-month CHANGE you expect for milk, '
                  'dairy and eggs specifically, signed> (worked example: '
                  '"DAIRY_EGGS: +0.1%" or "DAIRY_EGGS: -0.1%". Write your own number; '
                  'never write X.X.)'),
    'vegetables': ('VEGETABLES: <the percent month-on-month CHANGE you expect for '
                   'vegetables specifically, signed> (worked example: '
                   '"VEGETABLES: +0.5%" or "VEGETABLES: -0.3%". Write your own number; '
                   'never write X.X.)'),
    'sugar': ('SUGAR: <the percent month-on-month CHANGE you expect for sugar and '
             'confectionery specifically, signed> (worked example: "SUGAR: +0.1%" or '
             '"SUGAR: -0.1%". Write your own number; never write X.X.)'),
}
```

In `_judge_sector` (around line 524-592), change the prompt-building and return:

```python
    def _judge_sector(self, ctx: SectorContext, finals: list[AgentResponse]):
        """... (existing docstring unchanged) ..."""
        # ... existing transcript-building code unchanged, up to msgs = [...] ...

        category_lines = (
            '\n' + '\n'.join(_FOOD_CATEGORY_LINES.values())
            if ctx.sector == 'food' else ''
        )
        msgs = [
            {'role': 'system', 'content': _JUDGE_SYSTEM},
            {'role': 'user', 'content': (
                f"Sector: {ctx.sector} (report in {ctx.unit}). "
                f"This pressure lands at the next scheduled change on "
                f"{_next_change_label(ctx.sector)}. That date is a published schedule, "
                f"not something you are predicting. Read the pressure building into it; "
                f"do not restate a change already in effect.\n"
                f"Benchmark note: {ctx.verdict_note}\n\n"
                f"Analyst statements:\n{transcript}\n\n"
                + (f"Retrieved evidence (for CHECKING the analysts, not for adding "
                   f"new drivers):\n{judge_evidence}\n\n" if judge_evidence else "")
                + "Weigh the analysts, resolve their disagreement, and give the single "
                "present read. End with:\n" + _EST_LINE[ctx.sector] + category_lines)},
        ]
        try:
            text = llm.complete(msgs, tier=self._deep, max_tokens=280 + (120 if ctx.sector == 'food' else 0),
                                seed=llm.derive_seed(self._as_of, ctx.sector, 'judge'))
        except Exception:
            return None, '', {}
        _, statement = _parse_think(text)
        accepted, _ = _extract_guarded(ctx.sector, statement)
        subcategories = _extract_category_percents(statement) if ctx.sector == 'food' else {}
        return accepted, statement.strip(), subcategories
```

Update the one call site, `_run_sector` (around line 730-731):

```python
        finals = _latest_per_agent(history)
        judged, verdict, subcategories = self._judge_sector(ctx, finals)
        self._emit('judge', {'sector': ctx.sector, 'text': verdict,
                             'estimate': judged, 'unit': ctx.unit})
        return self._aggregate(ctx, history, judged=judged, cited=cited,
                               subcategories=subcategories)
```

Update `_aggregate`'s signature and its `SectorReading(...)` construction (around line 736-794):

```python
    def _aggregate(self, ctx: SectorContext, history: list[AgentResponse],
                   judged: Optional[float] = None,
                   cited: Optional[set] = None,
                   subcategories: Optional[dict] = None) -> SectorReading:
        # ... existing body unchanged, down to the return ...
        return SectorReading(
            sector=ctx.sector, direction=_direction(ctx.sector, avg),
            estimate=(round(avg, 2) if avg is not None else None),
            unit=ctx.unit, confidence=confidence,
            direction_agreement=direction_agreement,
            estimates=[round(float(e), 2) for e in ests],
            drivers=drivers, sources=sources,
            subcategories=subcategories or {})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest ph_economic_ai/tests/test_forum.py -v`
Expected: All tests PASS, including the 3 new ones. **Watch specifically** for any other existing test that unpacks `_judge_sector`'s return as a 2-tuple (e.g. `estimate, verdict = self._judge_sector(...)` in a test's own inline reimplementation) — if any exist, they will fail with `ValueError: too many values to unpack` and must be updated to unpack 3 values, matching how `_run_sector` itself was just updated.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/engine/forum.py ph_economic_ai/tests/test_forum.py
git commit -m "feat(forum): food judge reports six PSA sub-category reads

Same debate, same call budget -- the food judge's existing closing
instruction gains six worked-example lines, parsed by
debate._extract_category_percents and threaded through SectorReading.
Gas/electricity always get an empty subcategories dict; only food's judge
prompt includes the category lines at all."
```

---

### Task 4: Monitor's Food card shows the breakdown

**Files:**
- Modify: `ph_economic_ai/ui/pressure_monitor.py`
- Test: `ph_economic_ai/tests/test_monitor.py`

**Interfaces:**
- Consumes: `SectorReading.subcategories` (Task 1).

- [ ] **Step 1: Write the failing test**

First, check `test_monitor.py`'s existing test style for constructing a `PressureMonitorPanel`/calling `_sector_card` directly (read the file's first ~40 lines and any existing test calling `_sector_card` or building a `SectorReading` before writing this step, to match its exact fixture pattern rather than guessing one). Then add:

```python
def test_food_card_shows_subcategory_breakdown():
    """Rice/meat/fish/dairy&eggs/vegetables/sugar each get their own signed
    caption; a missing category reads as unavailable, never 0.0%; no value
    ever shows a literal '+' concatenated with a negative number (the exact
    RSK-053 shape, checked explicitly here because a new signed-percentage
    line is exactly where it would recur)."""
    from ph_economic_ai.engine.pressure_brief import SectorReading
    # Construct the panel the same way this file's other _sector_card tests do
    # -- see the existing pattern confirmed in Step 1 above.
    r = SectorReading(
        sector='food', direction='rising', estimate=0.4, unit='%', confidence=64,
        estimates=[0.3, 0.4, 0.5],
        subcategories={'rice': 0.1, 'meat': -0.3, 'fish': 0.8,
                       'vegetables': 0.5},  # dairy_eggs, sugar deliberately absent
    )
    card = panel._sector_card(r)
    from PyQt6.QtWidgets import QLabel
    texts = ' || '.join(w.text() for w in card.findChildren(QLabel))
    assert 'Rice +0.1%' in texts
    assert 'Meat -0.3%' in texts
    assert 'Fish +0.8%' in texts
    assert 'Vegetables +0.5%' in texts
    assert 'Dairy' in texts and '—' in texts  # missing category shows unavailable
    assert '+-' not in texts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_monitor.py::test_food_card_shows_subcategory_breakdown -v`
Expected: FAIL — `AssertionError` (breakdown text absent from the card; `_sector_card` doesn't render it yet).

- [ ] **Step 3: Add the breakdown row**

In `ph_economic_ai/ui/pressure_monitor.py`, add near the module's other color constants (`_DIR_COLOR`, `_SECTOR_COLOR`, around line 31-32):

```python
_CATEGORY_DISPLAY_LABELS = {
    'rice': 'Rice', 'meat': 'Meat', 'fish': 'Fish', 'dairy_eggs': 'Dairy & Eggs',
    'vegetables': 'Vegetables', 'sugar': 'Sugar',
}
```

In `_sector_card`, immediately before `return card` (currently line 843):

```python
        if r.sector == 'food' and getattr(r, 'subcategories', None) is not None:
            parts = []
            for category, label in _CATEGORY_DISPLAY_LABELS.items():
                value = r.subcategories.get(category)
                text = f'{label} {value:+.1f}%' if value is not None else f'{label} —'
                parts.append(text)
            breakdown = QLabel('  ·  '.join(parts))
            breakdown.setStyleSheet(f'color:{_T2};font-size:11px;margin-top:6px;')
            breakdown.setWordWrap(True)
            lay.addWidget(breakdown)

        return card
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_monitor.py -v`
Expected: All tests PASS, including the new one and every pre-existing test in the file (gas/electricity cards must render exactly as before — the new block is gated on `r.sector == 'food'`).

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/pressure_monitor.py ph_economic_ai/tests/test_monitor.py
git commit -m "feat(ui): Monitor's Food card shows the six-category breakdown

One caption row under the existing driver bullets: 'Rice +0.1%  ·  Meat
-0.3%  ·  ...', em-dash for a category with no parseable read this cycle.
Every value uses {v:+.1f}% -- never a literal '+' concatenated with the
value, the exact RSK-053 shape."
```

---

### Task 5: Relabel compact cards as a basket average

**Files:**
- Modify: `ph_economic_ai/ui/sector_forecast.py`
- Test: `ph_economic_ai/tests/test_sector_forecast.py`

**Interfaces:**
- Produces: updated `_SECTORS` label for `'food'` — consumed by both `stage4_report.py`'s sector-forecast card and `landing.py`'s recent-forecast tiles, confirmed to be the single shared source both read from (`sector_forecast_rows()`), so this one change reaches both without touching either file directly.

- [ ] **Step 1: Write the failing test**

Read `ph_economic_ai/tests/test_sector_forecast.py` first to confirm its exact existing assertions on the `'food'` row's label (so Step 1's new/updated test matches the file's real current content rather than an assumed one), then add or update:

```python
def test_food_row_is_labelled_as_a_basket_average():
    """FOOD (basket avg) — the compact card shows one blended number, but
    since it's now broken into six categories on the Monitor detail view,
    this label must not read as if it speaks for every category."""
    from ph_economic_ai.ui.sector_forecast import sector_forecast_rows
    rows = sector_forecast_rows(gas=1.0, food=0.4, elec=0.01)
    food_row = next(r for r in rows if r['key'] == 'food')
    assert 'basket avg' in food_row['label'].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest ph_economic_ai/tests/test_sector_forecast.py::test_food_row_is_labelled_as_a_basket_average -v`
Expected: FAIL — `AssertionError` (current label is `'Food'`, no "basket avg").

- [ ] **Step 3: Update the label**

In `ph_economic_ai/ui/sector_forecast.py`, change the `_SECTORS` tuple (currently lines 11-15):

```python
_SECTORS = [
    ('gas',  'Gas / fuel',  '{:+.2f} ₱/L'),
    ('food', 'Food (basket avg)', '{:+.2f} %'),
    ('elec', 'Electricity', '{:+.4f} ₱/kWh'),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest ph_economic_ai/tests/test_sector_forecast.py -v`
Expected: All tests PASS. **Check specifically** for any pre-existing test asserting the literal string `'Food'` (e.g. `assert row['label'] == 'Food'` or a card's rendered text containing exactly `'FOOD'` with no suffix) — those must be updated to expect `'Food (basket avg)'` / `'FOOD (BASKET AVG)'` in the same commit, not left failing.

Also run the two consumers directly, since they read this same list:
```bash
python -m pytest ph_economic_ai/tests/test_stage4_swarm.py ph_economic_ai/tests/test_stage4_classic_honesty.py ph_economic_ai/tests/test_landing_latest.py -v
```
Expected: All PASS. If any asserts the literal `'FOOD'` label text, update it the same way.

- [ ] **Step 5: Commit**

```bash
git add ph_economic_ai/ui/sector_forecast.py ph_economic_ai/tests/test_sector_forecast.py
git commit -m "fix(ui): label the compact Food card as a basket average

sector_forecast_rows() feeds both Report's stat card and Landing's tiles
from one place -- one label change reaches both. economy_overview.py's
separate 'Food Index (derived)' card is a different, gas-derived quantity
and is deliberately untouched (confirmed by its own code comment)."
```

---

## Final verification

- [ ] Run the complete affected-area suite:

```bash
python -m pytest ph_economic_ai/tests/test_trust_on_screen.py ph_economic_ai/tests/test_estimate_extraction.py ph_economic_ai/tests/test_forum.py ph_economic_ai/tests/test_monitor.py ph_economic_ai/tests/test_monitor_sourcing.py ph_economic_ai/tests/test_sector_forecast.py ph_economic_ai/tests/test_stage4_swarm.py ph_economic_ai/tests/test_stage4_classic_honesty.py ph_economic_ai/tests/test_landing_latest.py ph_economic_ai/tests/test_main_window.py -v
```
Expected: All PASS.

- [ ] Run the full suite: `python -m pytest ph_economic_ai/tests -q --no-header`
Expected: All PASS, count higher than the pre-plan baseline by exactly the number of new tests added across all five tasks.

- [ ] Visually verify by hand (per this session's own finding — do not force `QT_QPA_PLATFORM=offscreen`, it renders text as tofu boxes on this machine): build a `SimMainWindow` with a populated store, populate the Monitor page with a `PressureBrief` whose food `SectorReading` has a mixed-sign, partially-missing `subcategories` dict, and confirm the breakdown row reads correctly — same method this session used to find `RSK-053` and `RSK-054`.

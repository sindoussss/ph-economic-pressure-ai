# ph_economic_ai — Food Sub-Category Forecast

**Date:** 2026-08-12
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** New feature, not a continuation of an existing spec. Triggered by the owner sharing six Manila Public Information Office "Bantay Presyo" market bulletins (itemized wet-market prices for rice, fish, poultry, beef, pork, vegetables, sugar, eggs across six markets) and pointing out that the app's single blended "Food" number can't tell a household "fish is expensive this week because of rain, but rice is fine" — a real, common divergence a basket-wide percentage erases.

---

## 1. Problem & Goal

Every food-facing surface in the app today — the Overview card, the Monitor card, the Report screen's sector-forecast card, the benchmark's `food_cpi` target — represents "food" as one blended number. PSA's own CPI, which this project already validates against, is itself a *weighted average* of categories that move independently and sometimes in opposite directions (the owner's example: rain disrupts fishing and raises fish prices without necessarily touching rice or meat). A single number hides exactly the information a reader would act on.

**Goal:** break "food" into PSA CPI's own sub-categories — validated in the benchmark the same way every other target in this project is, and reasoned about explicitly (not just displayed) by the Forum's existing food debate — while keeping the compact cards that were built for one number honest about what they show rather than rebuilding every screen.

**Explicitly not a goal:** anchoring any category to a real peso price (e.g. "rice: ₱52/kg → +0.7% → ≈₱52.36/kg"). Investigated during brainstorming: the PSA API this project already automates (`openstat.psa.gov.ph/.../PI/`) only carries *indices* (CPI, PPI, RPI, WPI) for every category, never an absolute peso level, for any of the six sub-categories including rice. The one rice peso figure that exists in the codebase today (`_RICE_PRICE_PHP_KG = 52.0`, `engine/debate.py`) has no source and no date attached — it's a prompt-scale constant, not a tracked figure. A real per-category peso anchor needs a real, verified peso-price source first; see §7, Open Question.

---

## 2. Scope

### In scope

- **`benchmark/psa_cpi.py`** — refactor the three existing near-identical `fetch_X_cpi()` functions (transport, food, electricity) into thin wrappers over one parameterized `fetch_cpi_subcategory(coicop_prefix, out_csv, column_name, source_label)`. Pure refactor: the three existing functions keep their names and produce byte-identical committed CSVs. Add six new thin wrappers for the sub-categories below, each with its own committed CSV + provenance sidecar, via the same PX-Web mechanism.

- **Six new PSA CPI sub-category targets**, confirmed live against `openstat.psa.gov.ph`'s own "Commodity Description" dimension during brainstorming (not assumed):

  | Category | COICOP code |
  |---|---|
  | Rice | `01.1.1.12` |
  | Meat | `01.1.2` |
  | Fish and other seafood | `01.1.3` |
  | Milk, dairy products & eggs | `01.1.4` |
  | Vegetables, tubers & pulses | `01.1.7` |
  | Sugar, confectionery & desserts | `01.1.8` |

  Deliberately excludes Oils & Fats (`01.1.5`), Fruits & Nuts (`01.1.6`), and Ready-made/other (`01.1.9`) — none map to the owner's stated concern or the market bulletins, and the fetch mechanism is identical for any COICOP code if these are wanted later.

  **Known, permanent limitation, stated up front rather than discovered later:** PSA's CPI does not split poultry from beef/pork — `01.1.2 Meat` is one category. "Chicken vs. other meat" is not representable from this data source at any COICOP depth this API exposes.

- **`benchmark/food_subcategory_nowcast.py`** (new) — one nowcast per category, reusing `nowcast.run_mom_nowcast` / `mom_verdict` as-is (already target-agnostic) rather than six near-copies of `food_nowcast.py`. Same predictor frame as the existing food nowcast (oil, FX, global rice/wheat/corn/soybean futures, prev-month own value) — no bespoke per-category predictors in v1.

- **Statistical rigor** — identical to every other target in this benchmark: mean-corrected `BASELINE_POOL`, DM significance test, `audit.verdict_from_panel`-style exclusion of baselines from winning. Every category runs through `selection.run_selection_holdout` (`RSK-004`'s protocol) before any "predictable" claim is made anywhere. Per `DEC-010`, the six frames get pre-registered in a committed doc *before* the actual backtest runs.

- **`engine/debate.py`** — `_extract_category_percents(text) -> dict[str, Optional[float]]`, parsed the same way `_extract_percent` already is (anchored to its own line, worked example in the prompt per `RSK-012`'s lesson, reuses the existing `_MAX_REALISTIC_FOOD_PCT` ceiling rather than a new one).

- **`engine/forum.py`** — the food judge's closing instruction gains six required lines (`RICE:`, `MEAT:`, `FISH:`, `DAIRY_EGGS:`, `VEGETABLES:`, `SUGAR:`, each `+X.X%`/`-X.X%`) alongside the existing blended `ESTIMATE:` line. A category the judge doesn't address, or that fails to parse, comes back `None` — never defaulted to `0` or copied from the blended number.

- **`engine/pressure_brief.py`** — `SectorReading` gains one new field, **appended after `estimates`**: `subcategories: dict[str, Optional[float]] = field(default_factory=dict)`. Appended, not inserted, because `test_the_estimates_field_cannot_be_hit_positionally` exists specifically to guard against a field added mid-dataclass silently reassigning positional arguments at existing call sites.

- **`ui/pressure_monitor.py`** — the Food sector card (`_sector_card`) gains a breakdown row under the existing driver bullets, e.g. `Rice +0.1%  ·  Meat -0.3%  ·  Fish +0.8%  ·  Dairy & Eggs flat  ·  Vegetables +0.5%  ·  Sugar —` (em-dash for "no read this cycle"). Every value formatted `{v:+.1f}%` — never a literal `+` concatenated with the value, the exact mistake fixed elsewhere on this same screen (`RSK-053`, the BSP banner's per-sector ppt breakdown) earlier this session, named explicitly here because a new signed-percentage line is exactly where it would recur.

- **Compact cards** (`ui/economy_overview.py`'s Food card, `ui/stage4_report.py`'s sector-forecast stat card, Landing's recent-forecast tiles) — label only, no layout rework. `FOOD` / `FOOD INDEX (DERIVED)` becomes `FOOD (basket avg)` or equivalent, so the one remaining number doesn't read as if it speaks for every category.

### Out of scope

- Any peso-anchored price for any category (§1, explicitly not a goal; §7 tracks it as an open question).
- Bespoke per-category predictors in the benchmark (reuses the existing food predictor set).
- New navigation from the compact cards into the Monitor detail view.
- Any other sector (gas, electricity) — this spec is food-only.
- Oils & Fats, Fruits & Nuts, Ready-made/other sub-categories.

### Non-negotiable

- The `fetch_X_cpi()` refactor produces byte-identical CSVs for the three existing series, confirmed by hash before merging — a refactor that changes committed data by accident is the exact "verify in the worktree before pushing" failure mode this project's own history (`RSK-007` follow-up) already found and fixed once for a different function.
- No category's forecast is displayed as "predictable" anywhere in the app or docs unless it has passed the selection-holdout protocol, pre-registered before the run.
- A category with no parseable judge output is never displayed as `0%` or as a silent copy of another category's or the blended number's value.

---

## 3. Components

### 3.1 `benchmark/psa_cpi.py`

```python
def fetch_cpi_subcategory(coicop_prefix: str, out_csv: Path, column_name: str,
                          source_label: str) -> None:
    """Fetch one PSA OpenSTAT COICOP series (backcast + current tables spliced
    on the overlap) and freeze it to CSV with a provenance sidecar. Shared by
    every fetch_X_cpi() wrapper below -- extracted so a seventh series is a
    four-line wrapper, not a fourth copy of the same ~25-line function."""

def fetch_transport_cpi(out_csv=TRANSPORT_CSV) -> None:
    fetch_cpi_subcategory('07', out_csv, 'transport_cpi', '... COICOP 07 Transport')

def fetch_food_cpi(out_csv=FOOD_CSV) -> None:
    fetch_cpi_subcategory('01', out_csv, 'food_cpi', '... COICOP 01 Food and non-alcoholic beverages')

def fetch_electricity_cpi(out_csv=ELECTRICITY_CSV) -> None:
    fetch_cpi_subcategory('04.5.1', out_csv, 'electricity_cpi', '... COICOP 04.5.1 Electricity')

def fetch_rice_cpi(out_csv=RICE_CSV) -> None:
    fetch_cpi_subcategory('01.1.1.12', out_csv, 'rice_cpi', '... COICOP 01.1.1.12 Rice')

# ... fetch_meat_cpi ('01.1.2'), fetch_fish_cpi ('01.1.3'),
#     fetch_dairy_eggs_cpi ('01.1.4'), fetch_vegetables_cpi ('01.1.7'),
#     fetch_sugar_cpi ('01.1.8'), same shape.

def load_rice_cpi(csv_path=RICE_CSV) -> pd.Series: ...
def load_rice_mom(csv_path=RICE_CSV) -> pd.Series: ...
# ... one load_X_cpi / load_X_mom pair per new category, matching the
#     existing load_food_cpi / load_food_mom pattern exactly.
```

### 3.2 `benchmark/food_subcategory_nowcast.py` (new)

```python
CATEGORIES = ['rice', 'meat', 'fish', 'dairy_eggs', 'vegetables', 'sugar']

def _build_subcategory_frame(category: str, features: pd.DataFrame) -> pd.DataFrame:
    """Same shape as food_nowcast._build_food_frame: category's own CPI MoM as
    target, existing food predictor frame (oil, FX, global commodity futures),
    calendar_lag for prev_mom -- not a row lag, per the existing comment on
    every nowcast frame builder in this codebase."""

def run_subcategory_nowcast(category: str, min_train=24, features=None) -> dict:
    """Delegates to nowcast.run_mom_nowcast on the category's own frame.
    Returns the same shape run_mom_nowcast always returns (verdict,
    best_method, best_naive, best_skill_vs_naive, dm_p, n, calibration,
    rmse_by_method) -- no new verdict shape to learn."""
```

### 3.3 `engine/debate.py`

```python
_CATEGORY_LINES = {
    'rice': 'RICE', 'meat': 'MEAT', 'fish': 'FISH',
    'dairy_eggs': 'DAIRY_EGGS', 'vegetables': 'VEGETABLES', 'sugar': 'SUGAR',
}

def _extract_category_percents(text: str) -> dict[str, Optional[float]]:
    """One _last_estimate_match-style anchored parse per category line,
    each bounded by the existing _MAX_REALISTIC_FOOD_PCT (reused, not
    duplicated -- RSK-052's lesson). A category whose line is missing or
    unparseable is absent from the returned dict, never present as 0.0."""
```

### 3.4 `engine/forum.py`

- `_JUDGE_SYSTEM`'s food-sector closing instruction gains the six category lines, each with a worked example inline (`RSK-012`'s pattern: instructions, never a bare copyable template).
- `_judge_sector` calls `_extract_category_percents` alongside the existing single-estimate extraction and threads the result into the `SectorReading` it builds.

### 3.5 `engine/pressure_brief.py`

```python
@dataclass
class SectorReading:
    sector: str
    direction: str
    estimate: Optional[float]
    unit: str
    confidence: int
    direction_agreement: int = 0
    drivers: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    estimates: list[float] = field(default_factory=list)
    subcategories: dict[str, Optional[float]] = field(default_factory=dict)  # NEW, appended last
```

### 3.6 `ui/pressure_monitor.py`

- `_sector_card(r)`: for `r.sector == 'food'` and `r.subcategories` non-empty, add one caption row below the existing driver bullets rendering each category as `{label} {value:+.1f}%` or `{label} —` for a missing read, joined with `  ·  `.

### 3.7 `ui/economy_overview.py`, `ui/stage4_report.py`, Landing tiles

- Label text change only: `FOOD` → `FOOD (basket avg)` (or the closest equivalent given each card's existing label length constraints). No new data plumbing, no new fields consumed.

---

## 4. Testing

- `test_psa_cpi.py`: the refactored `fetch_transport_cpi`/`fetch_food_cpi`/`fetch_electricity_cpi` produce byte-identical CSVs to what's currently committed (hash comparison, not just "row count looks right"). One test per new `fetch_X_cpi`/`load_X_cpi`/`load_X_mom` triple, mirroring the existing food/electricity/transport tests exactly.
- `test_food_subcategory_nowcast.py` (new): one test per category mirroring `test_food_nowcast.py`'s structure (frame has contemporaneous drivers and lagged target, no same-month CPI-derived feature leak, insufficient-data path).
- `test_estimate_extraction.py`: `_extract_category_percents` tested the same way `_extract_percent` already is — worked examples, a missing category, an out-of-bound value rejected via the shared ceiling.
- `test_forum.py`: the food judge prompt contains all six category lines with worked examples (mirrors the existing "every prompt enumerates all its output lines" completeness test from `ADR-012`'s follow-up). A judge response missing one category line produces a `SectorReading.subcategories` dict without that key, not a `0.0` entry.
- A new test asserting `SectorReading`'s field order — constructing it positionally with the pre-existing field count must still raise `TypeError` if `subcategories` is omitted, and must succeed with it appended last — extending `test_the_estimates_field_cannot_be_hit_positionally`'s existing coverage rather than trusting append-only by convention alone.
- `test_pressure_monitor.py` (or wherever `_sector_card` is covered): mixed-sign regression test on the breakdown row (assert `'+-'` never appears, assert a missing category renders as `—` not `0.0%`), same shape as this session's `test_set_alert_sector_breakdown_has_one_sign_not_two`.
- Full suite green. Pre-registration doc for the six backtests written and committed before `food_subcategory_nowcast.py` is run for real (`DEC-010`).

---

## 5. Deliverables (definition of done)

1. `psa_cpi.py` refactored to one shared fetcher; three existing series byte-identical; six new series fetchable and committed with provenance sidecars.
2. `food_subcategory_nowcast.py` backtests all six categories through the full existing statistical pipeline (mean-baseline, DM test, selection-holdout), pre-registered before running.
3. The Forum's food judge reports all six categories per run; `SectorReading.subcategories` carries the result through to the UI layer.
4. Monitor's Food card shows the six-category breakdown; Overview/Report/Landing's compact cards are relabeled `(basket avg)` with no structural change.
5. Full suite green; the open peso-anchor question (§7) logged, not silently dropped.

---

## 6. Why it matters

Closes the actual gap the owner pointed at: a single "Food +0.4%" number cannot tell a reader that fish is expensive this week for a reason that doesn't touch rice, when the underlying PSA data the app already validates against is itself an average of categories that move independently. Matches the discipline every other sector in this project already follows — nothing is called "predictable" without a real backtest and a holdout re-test behind it — rather than adding an LLM-reasoned number with no statistical floor under it. Chicken-vs-meat and any peso-anchored price are named as explicit non-goals rather than silently absent, so the gap between what was asked for and what ships is visible rather than discovered later.

---

## 7. Open Question (not blocking this spec)

**Is there a real, structured, citable peso-price source for any of these six categories?** Investigated during brainstorming: the PSA OpenSTAT API this project already automates only publishes indices (CPI/PPI/RPI/WPI) — no absolute peso level for any category. PSA very likely publishes actual retail rice prices through a separate product (a "Palay, Rice and Corn" price bulletin), but whether it's reachable as a clean structured feed (like the CPI PX-Web API) or only as periodic PDF releases is unconfirmed. For meat/fish/vegetables/dairy & eggs/sugar, no candidate source has been identified at all. This is the same shape of problem `Q-ENG-009` (regional fuel retail prices) already turned out to be harder than assumed — the source believed to exist didn't publish what was needed. Recommended next step, separate from this spec: a dedicated research pass per category before any peso-anchor feature is designed, not a guess or a second unlabeled hardcoded constant.

# ph_economic_ai — Peso-Anchored Food Sub-Category Forecast

**Date:** 2026-08-13
**Status:** Approved (design), pending implementation plan
**Owner:** Sindous
**Program context:** Direct follow-on to the food sub-category forecast feature (`docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md`), specifically its §7 open question: is there a real peso-price source, so the app could say "rice was ₱120/kg on July 1, forecast +0.7%" instead of a bare percentage? That question was answered by a dedicated research pass (`docs/superpowers/specs/2026-08-12-food-subcategory-forecast-design.md` §7's 2026-08-13 update, [PR #20](https://github.com/sindoussss/ph-economic-pressure-ai/pull/20)): a real, live, structured peso-price source exists for four of the six categories. This spec designs the feature that source makes possible.

---

## 1. Problem & Goal

The food sub-category feature (shipped [PR #16](https://github.com/sindoussss/ph-economic-pressure-ai/pull/16)–[#18](https://github.com/sindoussss/ph-economic-pressure-ai/pull/18)) shows each category's forecast as a bare signed percentage — `Rice +0.3%`. A percentage alone doesn't answer the question a household actually has: *is that a big deal on what I'm already paying?* A real peso anchor, confirmed live for four categories, lets the app answer that directly.

**Goal:** for rice, meat, fish, and vegetables, show the real current peso price alongside the existing percentage forecast, and compute a projected price from the two — `₱52.36/kg now → ≈₱52.52/kg`. For dairy & eggs and sugar, change nothing; they keep showing percentage only, exactly as today.

**Explicitly not a goal:** inventing peso data where none was confirmed to exist. Dairy & eggs has partial coverage (eggs, not milk) and sugar has none at all — both stay out of scope for this feature rather than showing a misleadingly-labeled or partial number. See §2.

---

## 2. Scope

### In scope

- **Four categories get a peso anchor + projection:** Rice (Regular Milled Rice), Meat (Fresh Pork, Kasim), Fish (Galunggong / Round Scad), Vegetables (Tomato). One specific PSA commodity item per category, chosen for being the most commonly referenced in Philippine price-watch contexts — not an average across the many items PSA tracks per category (Fish alone has 57 species; a single representative item is legible, an average across all of them would not correspond to anything a shopper actually buys).

- **New module, `ph_economic_ai/engine/peso_anchor.py`** — fetches the four items from PSA OpenSTAT's confirmed-live `2M/2018NEW/` table family (the same folder the 2026-08-13 research pass confirmed current through July 2026; explicitly **not** `2M/RP` or `2M/NRP`, both confirmed frozen at 2021 during that research). Reuses the fetch-mechanism shape `psa_cpi.py`'s `_fetch_px_table` already establishes (GET metadata, POST query body, tabular `json` format), adapted for this table family's own variable names (`Commodity`, not `Commodity Description`; no COICOP-prefix matching needed since each fetch targets one named item, not a division).

- **Local JSON cache**, `ph_economic_ai/cache/bantay_cache.json` — already gitignored (a pre-existing, unused entry in `.gitignore`; no code referenced it before this feature). Cache key is per-category; each entry carries the fetched price, the PSA `as_of` month, and the date it was fetched. A request for "today's price" checks the cache first; only fetches from PSA if today's cache entry is missing.

- **Staleness bound, named explicitly because `RSK-041` already taught this project what happens without one:** a cache entry older than **60 days** is treated as unavailable, not silently shown. PSA's own normal reporting lag is about one month; 60 days catches "fetches have been failing repeatedly" without flagging PSA's routine lag as a problem.

- **Projection math:** `projected = anchor_price × (1 + subcategory_pct / 100)`. Pure arithmetic on two numbers that already exist independently — the anchor from PSA, the percentage from the Forum's existing food-judge debate (`SectorReading.subcategories`, already shipped). No new LLM reasoning, no new call.

- **UI: Monitor's Food card gains a peso strip**, a new block *below* the existing per-category percentage line, not replacing it. Four small blocks (rice/meat/fish/vegetables), each showing `anchor → projected`, e.g. `₱52.36 → ₱52.52`. A category missing either its peso price (fetch/cache failure) or its debate percentage (judge produced no read that cycle) shows `—` in that slot, the same convention the existing percentage line already uses — never a zero, never one piece standing in for the other. This layout keeps the existing compact line untouched (still all six categories) and gives the four peso numbers their own room, confirmed against two alternatives via the visual companion during brainstorming.

- **A dedicated caption** under the new strip, separate from the existing "not components of the figure above" note: *"PSA retail price (as of [month]) × this debate's forecast — exploratory projection, not a validated prediction."* This is the honesty guardrail this spec's own brainstorming settled on: the projected number is a stronger claim than a bare percentage (it reads as more precise even though it carries identical uncertainty), so it gets the same kind of caveat the gas sector's own peso-denominated exploratory numbers already carry — not a novel invention, an application of an existing convention to a new number.

### Out of scope

- Dairy & eggs and sugar. Confirmed during research: eggs alone (not milk) has peso data, and sugar has none. Neither gets a peso anchor in this feature — both continue showing percentage only, unchanged from the current shipped behavior. A future pass could revisit eggs specifically (labeled honestly as eggs, not the broader "dairy & eggs" bucket) as its own scoped addition, but that's not this spec.
- Any category-item other than the four named above. PSA's tables carry many items per category (rice alone has 4 variants); this feature shows exactly one representative item per category, not a picker, not an average.
- Any change to the underlying percentage forecast itself — `debate.py`, `forum.py`, and the food judge's prompt are all untouched by this feature. The peso anchor is purely additive display logic, reusing a percentage that already exists.
- Historical peso-price charts, trend lines, or multi-month views. This feature shows the current anchor and one projected point, nothing else.

### Non-negotiable

- The four fetchers must resolve against the confirmed-live `2M/2018NEW/` folder specifically. Fetching from `2M/RP` or `2M/NRP` by mistake would silently anchor the app to 2021-era prices with no error — exactly the trap the research pass found and named.
- A category missing either half of the pair (peso price or debate percentage) never gets a fabricated stand-in value. Absent means absent, the same rule the percentage-only line already enforces.
- The projection caption is present whenever the peso strip renders anything. No peso-denominated number appears without its caveat attached.

---

## 3. Components

### 3.1 `ph_economic_ai/engine/peso_anchor.py` (new)

```python
CATEGORY_ITEMS = {
    'rice':       {'table': '0042M4ARN01.px', 'commodity': 'RICE, REGULAR-MILLED, 1 KG'},
    'meat':       {'table': '0042M4ARN09.px', 'commodity': 'FRESH PORK, KASIM, 1 KG'},
    'fish':       {'table': '0042M4ARN11.px', 'commodity': 'FRESH FISH, ROUND SCAD, GALUNGGONG, MEDIUM, 1 KG'},
    'vegetables': {'table': '0042M4ARN05.px', 'commodity': 'TOMATO, 1 KG'},
}

CACHE_PATH = Path(__file__).parent.parent / 'cache' / 'bantay_cache.json'
STALE_AFTER_DAYS = 60

def get_anchor(category: str, cache_path: Path = CACHE_PATH) -> Optional[dict]:
    """Today's cached price for `category`, fetching live from PSA if the
    cache doesn't have today's entry yet. Returns None (never a stale or
    fabricated value) if: category isn't one of the four in scope, the fetch
    fails and no usable cache exists, or the only cached entry is older than
    STALE_AFTER_DAYS. Returns {'price': float, 'as_of': 'YYYY-MM',
    'fetched_on': 'YYYY-MM-DD'} on success."""

def project(anchor_price: float, pct_change: float) -> float:
    """anchor_price * (1 + pct_change / 100). Pure arithmetic, no I/O --
    factored out for direct unit testing independent of the fetch/cache path."""

def _fetch_live(category: str) -> Optional[dict]:
    """One PSA OpenSTAT PX-Web call for the category's specific commodity item,
    under 2M/2018NEW/ -- the confirmed-live folder, never 2M/RP or 2M/NRP.
    Geolocation is always the national figure ('Philippines'/value id '0'),
    matching every other PSA fetch already in this codebase (CPI, the six
    food sub-category series) -- not any of the 118 regional/provincial
    breakdowns this table family also carries. The commodity ID is resolved
    at request time by matching `CATEGORY_ITEMS[category]['commodity']`
    against the table's own `valueTexts` (label text, not a numeric index),
    the same resolution shape `psa_cpi.py::_resolve_commodity_id` already
    uses for CPI -- PX-Web value IDs aren't stable/guessable across tables,
    only the label text is."""
```

### 3.2 `ph_economic_ai/ui/pressure_monitor.py`

- `_sector_card(r)`: after the existing per-category percentage breakdown (and its "not components of the figure above" caption), for food only, add the new peso strip: four small blocks (rice/meat/fish/vegetables in that fixed order), each calling `peso_anchor.get_anchor(category)` and, if both that and `r.subcategories.get(category)` are present, `peso_anchor.project(...)` to compute the projected price. A category missing either input renders `—` in its block rather than being omitted from the layout (all four slots always present, for a stable, predictable card shape). Followed immediately by the dedicated caption naming the anchor's `as_of` month.

---

## 4. Testing

- `peso_anchor.py`: `project()` tested directly (positive percentage raises the price, negative lowers it — the sign has to actually flip the direction, not just format correctly, since this is real arithmetic not a formatted string). Fetch tested with the network mocked (never a real network call in a test — this project's own history, most recently `RSK-056`'s investigation, is full of what happens when tests touch live services). Cache read/write tested directly against a temp file. The 60-day staleness cutoff tested at the boundary (59 days: used; 61 days: treated as absent) and confirmed it doesn't fire on PSA's own normal ~30-day lag.
- `_sector_card`: the peso strip renders all four categories when both the anchor and the percentage are present for the same commodity. A category missing either half renders `—` in that slot specifically — the test asserts this by omitting first the anchor, then the percentage, independently, rather than only testing the both-present and both-absent cases. The caption is present whenever any slot has real content. Gas and electricity cards remain byte-for-byte unaffected — this feature's UI changes are all inside the existing `if r.sector == 'food'` gate.
- Full suite green before merge, matching every other change to this file this project has made.

---

## 5. Deliverables (definition of done)

1. `peso_anchor.py` fetches and caches the four confirmed items from the live PSA table family, with a working 60-day staleness bound.
2. Monitor's Food card shows the peso strip for rice/meat/fish/vegetables, with a projected price computed from the existing debate percentage, and its own caveat caption.
3. Dairy & eggs and sugar are provably unchanged — no new code path touches their rendering at all.
4. Full suite green.

---

## 6. Why it matters

Closes the gap the owner's original ask pointed at directly: "is it possible to get the old price... then the forecast... so..." — the app can now answer that for four of the six categories, with real numbers from a confirmed, live, structured source rather than a guess or a hardcoded constant. Matches this project's own hard-won caution about specificity implying confidence it hasn't earned (the entire `RSK-031` sign-bug family, and more directly, the null result the food sub-category backtest already returned for all six categories): the projected price gets the same "exploratory, not validated" treatment every other unvalidated number in this app already carries, applied consistently rather than invented fresh for this one feature.

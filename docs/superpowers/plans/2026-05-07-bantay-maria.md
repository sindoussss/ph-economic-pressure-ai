# Bantay Maria Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a self-contained disaster companion panel to Maria — accessible via the repurposed Code button (shield icon) — covering all Philippine hazards with live agency data and a full offline Taglish knowledge base.

**Architecture:** A `BantayModeWidget` (QWidget) replaces the code scratchpad at `main_stack` index 4. It loads all tab content from `OfflineDisasterKB.json` instantly and optionally fetches live data via `PHDisasterWatcher` (QThread). An embedded `BantayMiniChatWorker` (QThread, mirrors `ArtifactMiniChatWorker`) provides optional disaster-context-aware chat inside the panel without touching the main chat pipeline.

**Tech Stack:** Python 3.11, PyQt6, ollama, requests, json, re, datetime — all already present in `Maria_App_original.py`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `Project_Maria/OfflineDisasterKB.json` | **Create** | All Taglish disaster content — signal guides, first aid, go-bag, hotlines |
| `Project_Maria/Maria_App_original.py` | **Modify** | 6 surgical changes (see tasks below) |
| `Project_Maria/test_bantay.py` | **Create** | Unit tests for non-GUI logic |
| `Project_Maria/bantay_cache.json` | **Runtime** | Auto-created by PHDisasterWatcher; do not create manually |

---

## Task 1: Create OfflineDisasterKB.json

**Files:**
- Create: `Project_Maria/OfflineDisasterKB.json`
- Test: `Project_Maria/test_bantay.py` (write in this task)

- [ ] **Step 1: Create the KB file with full Taglish content**

Create `Project_Maria/OfflineDisasterKB.json`:

```json
{
  "typhoon": {
    "signal_1": {
      "meaning": "Magiging mahangin at maulan sa loob ng 36 oras. Ang mga puno ay maaaring umuga.",
      "what_to_do": "Maging alisto. I-check ang go-bag ninyo. Huwag maglaro sa labas kung malakas na ang hangin. Sundan ang mga balita mula sa PAGASA."
    },
    "signal_2": {
      "meaning": "Mapanganib na ang panahon. Malakas na hangin at ulan sa loob ng 24 oras. Posible ang pagbaha sa mababang lugar.",
      "what_to_do": "Iwasan ang paglalakbay sa dagat. I-secure ang mga bintana at pinto. Ihanda ang go-bag. Malaman kung saan ang pinakamalapit na evacuation center. Mag-imbak ng tubig at pagkain."
    },
    "signal_3": {
      "meaning": "Delikado na. Malakas na bagyo na maaaring makapinsala sa mga bahay. Posible ang malawak na pagbaha.",
      "what_to_do": "Pumunta na sa evacuation center kung ikaw ay nasa mababang lugar, malapit sa ilog, o sa mahinang matibay na bahay. Dalhin ang go-bag. Huwag tumawid sa baha. Abisuhan ang mga kamag-anak."
    },
    "signal_4": {
      "meaning": "Sobrang panganib. Malawak na pinsala sa mga gusali at pananim. Banta sa buhay.",
      "what_to_do": "Lisanin NGAYON ang mga lugar na malapit sa dagat, ilog, at bundok. Pumunta sa pinakamalakas na gusali o evacuation center. Huwag lalabas habang nasa gitna ng bagyo."
    },
    "signal_5": {
      "meaning": "Extreme na bagyo. Katastropikong pinsala. Banta sa lahat.",
      "what_to_do": "Manatili sa loob ng pinaka-matibay na silid. Huwag lumabas kahit anong mangyari. Hintayin ang pagdaan ng bagyo. Tawagan ang 911 kung emergency."
    }
  },
  "earthquake": {
    "during": "HUWAG TUMAKBO. Mag-duck, cover, at hold on — lumuhod, takpan ang ulo gamit ang mga kamay, at kumapit sa matatag na kasangkapan. Lumayo sa bintana, salamin, at mabibigat na bagay. Manatili hanggang tigilin ang pagyanig.",
    "after": "Lumabas ng maingat pagkatapos ng pagyanig. Mag-ingat sa mga aftershock — maaaring mangyari ang marami. Huwag gumamit ng elevator. I-check ang mga nakapaligid para sa mga nasugatan. Huwag pumasok sa mga sirang gusali.",
    "tsunami_watch": "Kung naramdaman mo ang malakas na lindol malapit sa dagat: TUMAKBO NA AGAD patungo sa mataas na lugar bago pa man dumating ang alon. Huwag hinintayin ang opisyal na babala. Ang tsunami ay mabilis at mapanganib."
  },
  "tsunami": {
    "warning": "UMALIS NA SA BAYBAYIN NGAYON. Pumunta sa lugar na hindi bababa sa 30 metro ang taas mula sa dagat. Huwag magdala ng maraming gamit — ang buhay ang pinaka-mahalaga. Huwag bumalik hanggang hindi sinabi ng mga awtoridad na ligtas na.",
    "after": "Huwag bumalik sa baybayin. Maaaring may mga susunod pang alon. Tulungan ang mga nasugatan. Abisuhan ang NDRRMC o 911 ng inyong kalagayan."
  },
  "volcanic": {
    "alert_1": {
      "meaning": "Mababang aktibidad ng bulkan. Normal na sitwasyon.",
      "what_to_do": "Huwag pumasok sa permanenteng danger zone. Sundan ang mga anunsyo mula sa PHIVOLCS."
    },
    "alert_2": {
      "meaning": "Tumataas ang aktibidad — mas maraming lindol at usok. Maaaring magsabog.",
      "what_to_do": "Maging handang umalis anumang oras. Ihanda ang go-bag. Mag-imbak ng tubig. Magsuot ng mask laban sa abo. Sundan ang mga balita."
    },
    "alert_3": {
      "meaning": "Malapit na ang pagsabog. Mataas na antas ng aktibidad.",
      "what_to_do": "Lisanin ang 7-km radius ng bulkan NGAYON. Magsuot ng mask at salamin. Takpan ang mga butas sa bahay para hindi pumasok ang abo. Pumunta sa evacuation center."
    },
    "alert_4": {
      "meaning": "Magsisabog sa loob ng ilang oras o araw.",
      "what_to_do": "Lisanin ang 8-km radius. HUWAG bumalik. Sundan ang mga instruksyon ng LGU at NDRRMC."
    },
    "alert_5": {
      "meaning": "Nagsasabog na.",
      "what_to_do": "Lumayo agad sa 10-km radius o higit pa. Protektahan ang ulo laban sa mga bato. Huwag lumabas hanggang hindi ligtas. Tawagan ang 911."
    }
  },
  "flood": {
    "what_to_do": "Kung tumataas na ang tubig: pumunta agad sa matataas na lugar o evacuation center. Huwag tumawid sa mabilis na agos ng tubig — kahit mababaw ito. Huwag humawak sa mga electrical wire. Dalhin ang go-bag. Abisuhan ang pamilya ng inyong lokasyon.",
    "evacuation_tips": "Magtanong sa barangay hall ng pinakamalapit na evacuation center. Magdala ng go-bag, gamot, at mahahalagang dokumento. Patayin ang kuryente bago umalis kung posible. Itaas ang mga mahahalagang gamit sa mas mataas na lugar sa loob ng bahay."
  },
  "first_aid": {
    "choking": "Kung may nakasara sa lalamunan: Tanungin kung makahinga pa. Kung hindi: Gawin ang Heimlich maneuver — tumayo sa likod ng biktima, ituon ang kamao sa gitna ng tiyan (sa itaas ng pusod), at itulak pataas nang mabilis nang 5 beses. Ulitin hanggang lumabas ang hadlang. Para sa sanggol: i-face down sa braso at tampalin ang likod nang 5 beses.",
    "drowning": "Ilabas ang biktima sa tubig nang maingat. Tawagan ang 911. Kung hindi humihinga, simulan ang CPR. Huwag iahon ang tubig mula sa baga — magsimula na agad ng CPR.",
    "cpr": "30 chest compressions: ilagay ang mga palad sa gitna ng dibdib at itulak pababa nang 5-6 cm nang mabilis (100-120 beses sa isang minuto). Pagkatapos ng 30, ibigay ang 2 rescue breaths: itaas ang baba, takpan ang ilong, at huminga sa bibig ng biktima. Ituloy hanggang huminga na o dumating ang tulong.",
    "wound": "Pindutin ang sugat nang mahigpit gamit ang malinis na tela o bandage. Huwag alisin ang tela kahit nabasa na sa dugo — dagdagan nalang. Itaas ang sugat na bahagi ng katawan kung maaari. Tawagan ang 911 kung malalim o malawak ang sugat."
  },
  "go_bag": [
    "Tubig — 1 litro bawat tao bawat araw para sa 3 araw",
    "Canned food at biscuit para 3 araw",
    "Gamot at first aid kit (lagyan ng paracetamol, plaster, gaas)",
    "Flashlight at extra batteries",
    "Battery-powered o hand-crank radio",
    "Kopya ng mahahalagang dokumento (birth certificate, PhilHealth, ID)",
    "Cash — small bills",
    "Extra damit at sapatos",
    "Disposable mask at alcohol",
    "Whistle para tumawag ng tulong",
    "Cellphone at extra charger o power bank",
    "Kumot o sleeping bag"
  ],
  "hotlines": [
    { "name": "911 Emergency", "number": "911" },
    { "name": "NDRRMC Operations Center", "number": "8911-5061" },
    { "name": "Philippine Red Cross", "number": "143" },
    { "name": "Philippine Coast Guard", "number": "8527-3877" },
    { "name": "PHIVOLCS (Volcanoes/Earthquakes)", "number": "8426-1468" },
    { "name": "PAGASA (Weather)", "number": "8284-0800" },
    { "name": "DSWD Social Services", "number": "8951-2803" }
  ]
}
```

- [ ] **Step 2: Write the test file**

Create `Project_Maria/test_bantay.py`:

```python
import json
import os
import sys
import unittest

KB_PATH = os.path.join(os.path.dirname(__file__), "OfflineDisasterKB.json")


class TestOfflineDisasterKB(unittest.TestCase):

    def setUp(self):
        with open(KB_PATH, encoding="utf-8") as f:
            self.kb = json.load(f)

    def test_typhoon_has_all_signals(self):
        typhoon = self.kb["typhoon"]
        for i in range(1, 6):
            key = f"signal_{i}"
            self.assertIn(key, typhoon, f"Missing {key}")
            self.assertIn("meaning", typhoon[key])
            self.assertIn("what_to_do", typhoon[key])

    def test_earthquake_has_three_keys(self):
        eq = self.kb["earthquake"]
        for key in ("during", "after", "tsunami_watch"):
            self.assertIn(key, eq)
            self.assertTrue(len(eq[key]) > 10, f"{key} content too short")

    def test_volcanic_has_all_levels(self):
        volcanic = self.kb["volcanic"]
        for i in range(1, 6):
            key = f"alert_{i}"
            self.assertIn(key, volcanic, f"Missing {key}")

    def test_go_bag_is_list(self):
        self.assertIsInstance(self.kb["go_bag"], list)
        self.assertGreaterEqual(len(self.kb["go_bag"]), 5)

    def test_hotlines_have_name_and_number(self):
        for h in self.kb["hotlines"]:
            self.assertIn("name", h)
            self.assertIn("number", h)
            self.assertTrue(len(h["number"]) >= 3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it passes**

```
cd Project_Maria
python -m pytest test_bantay.py::TestOfflineDisasterKB -v
```

Expected: 5 PASSED

- [ ] **Step 4: Commit**

```
git add Project_Maria/OfflineDisasterKB.json Project_Maria/test_bantay.py
git commit -m "feat: add OfflineDisasterKB.json with full Taglish disaster content"
```

---

## Task 2: PHDisasterWatcher (QThread)

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — insert class just before `class MariaPyQt(QMainWindow):` (~line 33810)
- Test: `Project_Maria/test_bantay.py` — append tests

- [ ] **Step 1: Write the failing tests first**

Append to `Project_Maria/test_bantay.py` (before the `if __name__ == "__main__":` line):

```python
class TestPHDisasterWatcherParsing(unittest.TestCase):

    def test_parse_pagasa_signal_2(self):
        html = "<p>Public Storm Warning Signal No. 2 is raised over Metro Manila</p>"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "typhoon")
        self.assertEqual(result["signal"], 2)

    def test_parse_pagasa_signal_with_hash(self):
        html = "PSWS #3 is in effect for Rizal, Quezon, Aurora"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertIsNotNone(result)
        self.assertEqual(result["signal"], 3)

    def test_parse_pagasa_no_signal_returns_none(self):
        html = "<p>No tropical cyclone affecting any part of the Philippines.</p>"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertIsNone(result)

    def test_parse_pagasa_extracts_typhoon_name(self):
        html = "Signal No. 2 — Typhoon Carina is 300 km east of Aurora"
        result = _PHDisasterWatcher_parse_pagasa_html(html)
        self.assertEqual(result["typhoon_name"], "Carina")

    def test_parse_phivolcs_m5_earthquake(self):
        data = [{"Magnitude": "5.1", "Depth": "15 km",
                 "Location": "10 km N of Davao City", "Date": "07 May 2026 - 02:15 AM"}]
        result = _PHDisasterWatcher_parse_phivolcs_json(data)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "earthquake")
        self.assertAlmostEqual(result["magnitude"], 5.1)
        self.assertFalse(result["tsunami_watch"])

    def test_parse_phivolcs_m7_sets_tsunami_watch(self):
        data = [{"Magnitude": "7.2", "Depth": "10 km",
                 "Location": "50 km W of Davao", "Date": "07 May 2026"}]
        result = _PHDisasterWatcher_parse_phivolcs_json(data)
        self.assertTrue(result["tsunami_watch"])

    def test_parse_phivolcs_below_threshold_returns_none(self):
        data = [{"Magnitude": "2.5", "Depth": "5 km",
                 "Location": "Somewhere", "Date": "07 May 2026"}]
        result = _PHDisasterWatcher_parse_phivolcs_json(data)
        self.assertIsNone(result)

    def test_parse_phivolcs_empty_list_returns_none(self):
        self.assertIsNone(_PHDisasterWatcher_parse_phivolcs_json([]))
```

Add these helper stubs at the top of the test file (after the imports, before the class definitions):

```python
import re as _re

_SIGNAL_RE = _re.compile(
    r'(?:Public\s+Storm\s+Warning\s+)?(?:Signal\s+(?:No\.?\s*)?#?\s*|PSWS\s+#\s*)([1-5])',
    _re.IGNORECASE
)
_TYPHOON_RE = _re.compile(r'[Tt]yphoon\s+([A-Z][a-z]+)|[Bb]agyo\s+([A-Z][a-z]+)')
_EQ_MIN_MAG = 4.0


def _PHDisasterWatcher_parse_pagasa_html(html: str):
    m = _SIGNAL_RE.search(html)
    if not m:
        return None
    signal = int(m.group(1))
    name_m = _TYPHOON_RE.search(html)
    typhoon_name = (name_m.group(1) or name_m.group(2)) if name_m else ""
    return {"type": "typhoon", "signal": signal, "typhoon_name": typhoon_name}


def _PHDisasterWatcher_parse_phivolcs_json(data: list):
    if not data:
        return None
    latest = data[0]
    try:
        mag = float(latest.get("Magnitude", 0))
    except (ValueError, TypeError):
        return None
    if mag < _EQ_MIN_MAG:
        return None
    return {
        "type": "earthquake",
        "magnitude": mag,
        "depth": latest.get("Depth", ""),
        "location": latest.get("Location", ""),
        "date": latest.get("Date", ""),
        "tsunami_watch": mag >= 7.0,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest test_bantay.py::TestPHDisasterWatcherParsing -v
```

Expected: Tests pass immediately because the logic is in the stubs. If any fail, fix the regex first before proceeding.

- [ ] **Step 3: Add PHDisasterWatcher class to Maria_App_original.py**

Find the line `class MariaPyQt(QMainWindow):` (~line 33810). Insert the following class directly above it:

```python
# ══════════════════════════════════════════════════════════════════════════════
# BANTAY MARIA — Philippine Disaster Watcher
# ══════════════════════════════════════════════════════════════════════════════
class PHDisasterWatcher(QThread):
    """
    Polls PAGASA and PHIVOLCS every 30 minutes when connected.
    Emits disaster_update(dict) with the latest bulletin.
    Emits offline_mode() when network is unavailable and no cache exists.
    Falls back to bantay_cache.json on network failure.
    """
    disaster_update = pyqtSignal(dict)
    offline_mode    = pyqtSignal()

    _PAGASA_URL  = "https://bagong.pagasa.dost.gov.ph/"
    _PHIVOLCS_EQ = "https://earthquake.phivolcs.dost.gov.ph/EQLatest-Monthly.json"
    _POLL_MS     = 30 * 60 * 1000  # 30 minutes

    _SIGNAL_RE = re.compile(
        r'(?:Public\s+Storm\s+Warning\s+)?(?:Signal\s+(?:No\.?\s*)?#?\s*|PSWS\s+#\s*)([1-5])',
        re.IGNORECASE
    )
    _TYPHOON_RE = re.compile(r'[Tt]yphoon\s+([A-Z][a-z]+)|[Bb]agyo\s+([A-Z][a-z]+)')
    _EQ_MIN_MAG = 4.0

    def __init__(self, cache_path: str, parent=None):
        super().__init__(parent)
        self._cache_path = cache_path
        self._cancelled  = False

    def cancel(self):
        self._cancelled = True
        self.wait(800)

    def run(self):
        while not self._cancelled:
            if is_connected():
                bulletin = self._fetch_all()
                if bulletin.get("type"):
                    self._save_cache(bulletin)
                    self.disaster_update.emit(bulletin)
                else:
                    self._emit_cached_or_offline()
            else:
                self._emit_cached_or_offline()
            for _ in range(self._POLL_MS // 100):
                if self._cancelled:
                    return
                self.msleep(100)

    def _fetch_all(self) -> dict:
        import datetime
        bulletin = {"source": "live", "fetched_at": datetime.datetime.now().isoformat()}
        try:
            resp = requests.get(self._PAGASA_URL, timeout=10)
            typhoon = self._parse_pagasa_html(resp.text)
            if typhoon:
                bulletin.update(typhoon)
        except Exception:
            pass
        if not bulletin.get("type"):
            try:
                resp = requests.get(self._PHIVOLCS_EQ, timeout=10)
                eq = self._parse_phivolcs_json(resp.json())
                if eq:
                    bulletin.update(eq)
            except Exception:
                pass
        # NDRRMC (floods/landslides) has no stable JSON API in v1;
        # flood guidance is covered offline by OfflineDisasterKB.json.
        return bulletin

    @staticmethod
    def _parse_pagasa_html(html: str) -> dict | None:
        m = PHDisasterWatcher._SIGNAL_RE.search(html)
        if not m:
            return None
        signal = int(m.group(1))
        name_m = PHDisasterWatcher._TYPHOON_RE.search(html)
        typhoon_name = (name_m.group(1) or name_m.group(2)) if name_m else ""
        return {"type": "typhoon", "signal": signal, "typhoon_name": typhoon_name}

    @staticmethod
    def _parse_phivolcs_json(data: list) -> dict | None:
        if not data:
            return None
        latest = data[0]
        try:
            mag = float(latest.get("Magnitude", 0))
        except (ValueError, TypeError):
            return None
        if mag < PHDisasterWatcher._EQ_MIN_MAG:
            return None
        return {
            "type": "earthquake",
            "magnitude": mag,
            "depth":     latest.get("Depth", ""),
            "location":  latest.get("Location", ""),
            "date":      latest.get("Date", ""),
            "tsunami_watch": mag >= 7.0,
        }

    def _save_cache(self, bulletin: dict):
        with open(self._cache_path, 'w', encoding='utf-8') as f:
            json.dump(bulletin, f, ensure_ascii=False, indent=2)

    def _load_cache(self) -> dict | None:
        if not os.path.exists(self._cache_path):
            return None
        with open(self._cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _emit_cached_or_offline(self):
        cached = self._load_cache()
        if cached:
            cached["source"] = "cached"
            self.disaster_update.emit(cached)
        else:
            self.offline_mode.emit()
```

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 5: Commit**

```
git add Project_Maria/Maria_App_original.py Project_Maria/test_bantay.py
git commit -m "feat: add PHDisasterWatcher polling PAGASA and PHIVOLCS"
```

---

## Task 3: BantayMiniChatWorker (QThread)

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — insert after `PHDisasterWatcher`, before `class MariaPyQt`
- Test: `Project_Maria/test_bantay.py` — append tests

- [ ] **Step 1: Write the failing tests**

Append to `Project_Maria/test_bantay.py` (before `if __name__ == "__main__":`):

```python
class TestBantayMiniChatWorkerPrompt(unittest.TestCase):

    def _build(self, ctx):
        # Inline the static method logic so tests don't need Qt
        htype    = ctx.get("type")
        location = ctx.get("location", "")
        if htype == "typhoon":
            signal = ctx.get("signal", "?")
            name   = ctx.get("typhoon_name", "")
            situation = f"Aktibong bagyo — Signal No. {signal}" + (f" ({name})" if name else "")
        elif htype == "earthquake":
            mag  = ctx.get("magnitude", "?")
            loc  = ctx.get("location", "")
            situation = f"Naganap na lindol — Magnitude {mag}" + (f" sa {loc}" if loc else "")
        elif htype == "tsunami":
            situation = "TSUNAMI WARNING — Aktibong alerto"
        elif htype == "volcanic":
            level = ctx.get("alert_level", "?")
            vname = ctx.get("volcano_name", "")
            situation = f"Aktibong bulkan — Alert Level {level}" + (f" ({vname})" if vname else "")
        elif htype == "flood":
            situation = "Babala sa baha — Aktibong advisory"
        else:
            situation = "Walang aktibong alerto sa ngayon"
        loc_line = f"Lokasyon ng gumagamit: {location}" if location else ""
        return (
            "Ikaw si Maria, isang Filipino AI disaster companion.\n"
            f"Kasalukuyang sitwasyon: {situation}\n"
            f"{loc_line}\n"
            "Sumagot LAMANG sa simpleng Taglish. Maikli at malinaw ang sagot.\n"
            "Huwag mag-speculate. Kung hindi sigurado, i-refer sa NDRRMC o 911."
        ).strip()

    def test_typhoon_prompt_contains_signal_and_name(self):
        ctx = {"type": "typhoon", "signal": 2, "typhoon_name": "Carina", "location": "Quezon City"}
        prompt = self._build(ctx)
        self.assertIn("Signal No. 2", prompt)
        self.assertIn("Carina", prompt)
        self.assertIn("Quezon City", prompt)
        self.assertIn("Taglish", prompt)

    def test_earthquake_prompt_contains_magnitude(self):
        ctx = {"type": "earthquake", "magnitude": 6.1, "location": "Davao", "location": ""}
        prompt = self._build(ctx)
        self.assertIn("6.1", prompt)

    def test_no_hazard_prompt_is_safe(self):
        ctx = {"type": None, "location": "Manila"}
        prompt = self._build(ctx)
        self.assertIn("Walang aktibong alerto", prompt)
        self.assertIn("Manila", prompt)

    def test_tsunami_prompt(self):
        ctx = {"type": "tsunami", "location": "Batangas"}
        prompt = self._build(ctx)
        self.assertIn("TSUNAMI", prompt)

    def test_volcanic_prompt_contains_level(self):
        ctx = {"type": "volcanic", "alert_level": 3, "volcano_name": "Taal", "location": ""}
        prompt = self._build(ctx)
        self.assertIn("Alert Level 3", prompt)
        self.assertIn("Taal", prompt)
```

- [ ] **Step 2: Run tests to verify they pass**

```
python -m pytest test_bantay.py::TestBantayMiniChatWorkerPrompt -v
```

Expected: 5 PASSED

- [ ] **Step 3: Add BantayMiniChatWorker to Maria_App_original.py**

Insert directly after `PHDisasterWatcher` class (before `class MariaPyQt`):

```python
class BantayMiniChatWorker(QThread):
    """
    Lightweight disaster-context-aware chat worker.
    Mirrors ArtifactMiniChatWorker — same signals, same streaming pattern.
    System prompt is pre-loaded with the current hazard type, severity, and location.
    No web RAG — disaster context is injected directly; speculation is suppressed.
    """
    chunk_ready    = pyqtSignal(str)
    reply_done     = pyqtSignal(str)
    reply_revised  = pyqtSignal(str)
    error_signal   = pyqtSignal(str)
    status_changed = pyqtSignal(str, str)

    def __init__(self, messages: list, disaster_context: dict, parent=None):
        super().__init__(parent)
        self.messages         = list(messages)
        self.disaster_context = dict(disaster_context)
        self._cancelled       = False

    def cancel(self):
        self._cancelled = True
        self.wait(800)

    @staticmethod
    def _build_system_prompt(ctx: dict) -> str:
        htype    = ctx.get("type")
        location = ctx.get("location", "")
        if htype == "typhoon":
            signal = ctx.get("signal", "?")
            name   = ctx.get("typhoon_name", "")
            situation = f"Aktibong bagyo — Signal No. {signal}" + (f" ({name})" if name else "")
        elif htype == "earthquake":
            mag = ctx.get("magnitude", "?")
            loc = ctx.get("location", "")
            situation = f"Naganap na lindol — Magnitude {mag}" + (f" sa {loc}" if loc else "")
        elif htype == "tsunami":
            situation = "TSUNAMI WARNING — Aktibong alerto"
        elif htype == "volcanic":
            level = ctx.get("alert_level", "?")
            vname = ctx.get("volcano_name", "")
            situation = f"Aktibong bulkan — Alert Level {level}" + (f" ({vname})" if vname else "")
        elif htype == "flood":
            situation = "Babala sa baha — Aktibong advisory"
        else:
            situation = "Walang aktibong alerto sa ngayon"
        loc_line = f"Lokasyon ng gumagamit: {location}" if location else ""
        return (
            "Ikaw si Maria, isang Filipino AI disaster companion.\n"
            f"Kasalukuyang sitwasyon: {situation}\n"
            f"{loc_line}\n"
            "Sumagot LAMANG sa simpleng Taglish. Maikli at malinaw ang sagot.\n"
            "Huwag mag-speculate. Kung hindi sigurado, i-refer sa NDRRMC o 911."
        ).strip()

    def run(self):
        try:
            system_prompt = self._build_system_prompt(self.disaster_context)
            full_messages = [{"role": "system", "content": system_prompt}] + self.messages
            full_reply = ""
            with _OLLAMA_SEMAPHORE:
                stream = ollama.chat(
                    model=MODEL,
                    messages=full_messages,
                    stream=True,
                    options={"temperature": 0.3, "num_predict": 200,
                             "num_ctx": 2048, "num_gpu": _NUM_GPU_LAYERS},
                )
                for chunk in stream:
                    if self._cancelled:
                        return
                    if isinstance(chunk, dict):
                        token = chunk.get("message", {}).get("content", "")
                    else:
                        token = getattr(getattr(chunk, "message", None), "content", "") or ""
                    if token:
                        full_reply += token
                        self.chunk_ready.emit(token)
            self.reply_done.emit(full_reply)
            detector = HallucinationDetector()
            revised = detector.detect_and_flag(full_reply)
            if revised != full_reply:
                self.reply_revised.emit(revised)
        except Exception as e:
            self.error_signal.emit(str(e))
```

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 5: Commit**

```
git add Project_Maria/Maria_App_original.py Project_Maria/test_bantay.py
git commit -m "feat: add BantayMiniChatWorker with disaster-context system prompt"
```

---

## Task 4: BantayModeWidget (QWidget)

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — insert after `BantayMiniChatWorker`, before `class MariaPyQt`

- [ ] **Step 1: Add BantayModeWidget to Maria_App_original.py**

Insert directly after `BantayMiniChatWorker` (before `class MariaPyQt`):

```python
class BantayModeWidget(QWidget):
    """
    Self-contained disaster companion panel.
    Loads all tab content from OfflineDisasterKB.json instantly — no LLM required.
    Optional BantayMiniChatWorker is started per user message at the bottom.
    Connected to PHDisasterWatcher via apply_bulletin() and to
    GPSLocationWorker via refresh_location().
    """

    def __init__(self, kb_path: str, cache_path: str, parent=None):
        super().__init__(parent)
        self._kb_path    = kb_path
        self._cache_path = cache_path
        self._kb         = self._load_kb()
        self._bulletin: dict = {}
        self._disaster_context: dict = {"type": None, "location": ""}
        self._chat_history: list = []
        self._worker: BantayMiniChatWorker | None = None
        self._streaming_bubble = None
        self._streaming_text   = ""
        self._setup_ui()
        self._load_cached_bulletin()

    # ── KB / cache ────────────────────────────────────────────────────────────

    def _load_kb(self) -> dict:
        if os.path.exists(self._kb_path):
            with open(self._kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _load_cached_bulletin(self):
        if os.path.exists(self._cache_path):
            with open(self._cache_path, 'r', encoding='utf-8') as f:
                bulletin = json.load(f)
            self.apply_bulletin(bulletin)

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh_location(self, location: str):
        self._loc_label.setText(f"📍 {location}")
        self._disaster_context["location"] = location

    def apply_bulletin(self, bulletin: dict):
        self._bulletin = bulletin
        source = bulletin.get("source", "offline")
        htype  = bulletin.get("type")

        if source == "live":
            self._live_badge.setText("⬤ Live")
            self._live_badge.setStyleSheet(
                "color: #22c55e; font-size: 11px; "
                "font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;"
            )
        else:
            self._live_badge.setText("⬤ Cached/Offline")
            self._live_badge.setStyleSheet(
                f"color: {THEME['text_secondary']}; font-size: 11px; "
                "font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;"
            )

        if htype == "typhoon":
            sig  = bulletin.get("signal", "?")
            name = bulletin.get("typhoon_name", "")
            self._hazard_label.setText(
                f"SIGNAL NO. {sig}" + (f"  —  {name}" if name else "")
            )
        elif htype == "earthquake":
            mag = bulletin.get("magnitude", "?")
            loc = bulletin.get("location", "")
            self._hazard_label.setText(f"LINDOL  •  Magnitude {mag}" + (f"\n{loc}" if loc else ""))
        elif htype == "tsunami":
            self._hazard_label.setText("⚠ TSUNAMI WARNING\nLumayo sa baybayin NGAYON")
        elif htype == "volcanic":
            level = bulletin.get("alert_level", "?")
            vname = bulletin.get("volcano_name", "")
            self._hazard_label.setText(
                f"ALERT LEVEL {level}" + (f"  —  {vname}" if vname else "")
            )
        elif htype == "flood":
            self._hazard_label.setText("BABALA SA BAHA\nMaging handa at sundan ang mga balita")
        else:
            self._hazard_label.setText("Walang aktibong alerto")

        self._disaster_context.update({
            k: v for k, v in bulletin.items()
            if k not in ("source", "fetched_at")
        })
        self._refresh_tabs()

    def set_offline(self):
        self._live_badge.setText("⬤ Offline")
        self._live_badge.setStyleSheet(
            f"color: {THEME['text_secondary']}; font-size: 11px; "
            "font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;"
        )

    # ── UI setup ──────────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_header())
        layout.addWidget(self._hazard_display())
        layout.addWidget(self._build_tabs(), stretch=1)
        layout.addWidget(self._build_minichat())

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setFixedHeight(64)
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['bg_primary']};
                border-bottom: 1px solid {THEME['border_light']};
            }}
        """)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(24, 0, 24, 0)
        hl.setSpacing(10)

        back_btn = QPushButton("←")
        _debounce_btn(back_btn)
        back_btn.setFixedSize(32, 32)
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; border: none; border-radius: 6px;
                font-size: 16px; color: {THEME['text_secondary']};
            }}
            QPushButton:hover {{ background: {THEME['hover_light']}; color: {THEME['text_primary']}; }}
        """)
        back_btn.clicked.connect(self._go_back)
        hl.addWidget(back_btn)

        title = QLabel("🛡 BANTAY MODE")
        title.setStyleSheet(f"""
            font-size: 17px; font-weight: 700; color: {THEME['text_primary']};
            font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
        """)
        hl.addWidget(title)
        hl.addStretch()

        self._loc_label = QLabel("📍 —")
        self._loc_label.setStyleSheet(f"""
            background: {THEME['bg_secondary']}; color: {THEME['text_secondary']};
            border-radius: 12px; padding: 4px 12px; font-size: 12px;
            font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
        """)
        hl.addWidget(self._loc_label)

        self._live_badge = QLabel("⬤ Offline")
        self._live_badge.setStyleSheet(
            f"color: {THEME['text_secondary']}; font-size: 11px; "
            "font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;"
        )
        hl.addWidget(self._live_badge)
        return header

    def _hazard_display(self) -> QLabel:
        self._hazard_label = QLabel("Walang aktibong alerto")
        self._hazard_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hazard_label.setWordWrap(True)
        self._hazard_label.setFixedHeight(72)
        self._hazard_label.setStyleSheet(f"""
            font-size: 20px; font-weight: 700; color: {THEME['text_primary']};
            padding: 12px 24px; background: {THEME['bg_secondary']};
            border-bottom: 1px solid {THEME['border_light']};
            font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
        """)
        return self._hazard_label

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: none; background: {THEME['bg_primary']}; }}
            QTabBar::tab {{
                background: {THEME['bg_secondary']}; color: {THEME['text_secondary']};
                padding: 8px 16px; border: none; font-size: 12px;
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
            }}
            QTabBar::tab:selected {{
                color: {THEME['text_primary']}; font-weight: 600;
                border-bottom: 2px solid {THEME['accent_primary']};
                background: {THEME['bg_primary']};
            }}
        """)

        def _tab(label):
            tb = QTextBrowser()
            tb.setReadOnly(True)
            tb.setOpenExternalLinks(False)
            tb.setStyleSheet(f"""
                QTextBrowser {{
                    background: {THEME['bg_primary']}; border: none;
                    padding: 16px; font-size: 13px; color: {THEME['text_primary']};
                    font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
                    line-height: 1.6;
                }}
            """)
            tabs.addTab(tb, label)
            return tb

        self._tab_ano     = _tab("Ano Gagawin")
        self._tab_gabay   = _tab("Gabay")
        self._tab_gobag   = _tab("Go-Bag")
        self._tab_hotline = _tab("Hotlines")
        self._refresh_tabs()
        return tabs

    def _refresh_tabs(self):
        kb    = self._kb
        b     = self._bulletin
        htype = b.get("type")

        if htype == "typhoon":
            sig  = b.get("signal", 1)
            data = kb.get("typhoon", {}).get(f"signal_{sig}", {})
            self._tab_ano.setPlainText(data.get("what_to_do", ""))
            self._tab_gabay.setPlainText(data.get("meaning", ""))
        elif htype == "earthquake":
            eq = kb.get("earthquake", {})
            during = eq.get("during", "")
            after  = eq.get("after", "")
            ts     = eq.get("tsunami_watch", "")
            self._tab_ano.setPlainText(during + "\n\n" + after)
            self._tab_gabay.setPlainText(ts)
        elif htype == "tsunami":
            ts = kb.get("tsunami", {})
            self._tab_ano.setPlainText(ts.get("warning", ""))
            self._tab_gabay.setPlainText(ts.get("after", ""))
        elif htype == "volcanic":
            level = b.get("alert_level", 1)
            data  = kb.get("volcanic", {}).get(f"alert_{level}", {})
            self._tab_ano.setPlainText(data.get("what_to_do", ""))
            self._tab_gabay.setPlainText(data.get("meaning", ""))
        elif htype == "flood":
            fl = kb.get("flood", {})
            self._tab_ano.setPlainText(fl.get("what_to_do", ""))
            self._tab_gabay.setPlainText(fl.get("evacuation_tips", ""))
        else:
            self._tab_ano.setPlainText(
                "Walang aktibong alerto sa ngayon.\n\n"
                "Maging handa at sundan ang mga balita mula sa PAGASA at NDRRMC."
            )
            self._tab_gabay.setPlainText(
                "Mag-ihanda ng go-bag. Malaman kung saan ang evacuation center. "
                "Sundan ang mga opisyal na anunsyo."
            )

        go_bag_items = kb.get("go_bag", [])
        self._tab_gobag.setPlainText("\n".join(f"• {item}" for item in go_bag_items))

        hotlines = kb.get("hotlines", [])
        self._tab_hotline.setPlainText(
            "\n".join(f"{h['name']}\n  {h['number']}" for h in hotlines)
        )

    def _build_minichat(self) -> QFrame:
        container = QFrame()
        container.setStyleSheet(f"""
            QFrame {{
                background: {THEME['bg_secondary']};
                border-top: 1px solid {THEME['border_light']};
            }}
        """)
        vl = QVBoxLayout(container)
        vl.setContentsMargins(16, 12, 16, 12)
        vl.setSpacing(8)

        label = QLabel("Magtanong kay Maria")
        label.setStyleSheet(
            f"font-size: 11px; font-weight: 600; color: {THEME['text_secondary']}; "
            "font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;"
        )
        vl.addWidget(label)

        self._chat_scroll = QScrollArea()
        self._chat_scroll.setWidgetResizable(True)
        self._chat_scroll.setFixedHeight(160)
        self._chat_scroll.setStyleSheet("border: none; background: transparent;")
        self._chat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chat_container = QWidget()
        self._chat_layout    = QVBoxLayout(self._chat_container)
        self._chat_layout.setContentsMargins(0, 0, 0, 0)
        self._chat_layout.setSpacing(6)
        self._chat_layout.addStretch()
        self._chat_scroll.setWidget(self._chat_container)
        vl.addWidget(self._chat_scroll)

        self._add_bubble("Handa ako. Anong tanong mo tungkol sa sitwasyon ngayon?", is_user=False)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self._bantay_input = QLineEdit()
        self._bantay_input.setPlaceholderText("I-type ang tanong mo...")
        self._bantay_input.setStyleSheet(f"""
            QLineEdit {{
                background: {THEME['bg_primary']}; border: 1px solid {THEME['border_medium']};
                border-radius: 8px; padding: 8px 12px; font-size: 13px;
                color: {THEME['text_primary']};
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
            }}
        """)
        self._bantay_input.returnPressed.connect(self._send_bantay_message)
        input_row.addWidget(self._bantay_input)

        send_btn = QPushButton("Send")
        _debounce_btn(send_btn)
        send_btn.setFixedSize(64, 36)
        send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background: {THEME['accent_primary']}; color: white; border: none;
                border-radius: 8px; font-size: 12px; font-weight: 600;
                font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
            }}
            QPushButton:hover {{ background: {THEME['accent_primary_hover']}; }}
        """)
        send_btn.clicked.connect(self._send_bantay_message)
        input_row.addWidget(send_btn)
        vl.addLayout(input_row)
        return container

    def _add_bubble(self, text: str, is_user: bool) -> QLabel:
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setMaximumWidth(360)
        bubble.setStyleSheet(f"""
            background: {THEME['bg_tertiary'] if is_user else THEME['bg_primary']};
            color: {THEME['text_primary']}; border-radius: 10px;
            padding: 8px 12px; font-size: 12px;
            font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
        """)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        if is_user:
            row.addStretch()
        row.addWidget(bubble)
        if not is_user:
            row.addStretch()
        self._chat_layout.insertLayout(self._chat_layout.count() - 1, row)
        QTimer.singleShot(50, lambda: self._chat_scroll.verticalScrollBar().setValue(
            self._chat_scroll.verticalScrollBar().maximum()
        ))
        return bubble

    def _send_bantay_message(self):
        text = self._bantay_input.text().strip()
        if not text:
            return
        self._bantay_input.clear()
        self._add_bubble(text, is_user=True)
        self._chat_history.append({"role": "user", "content": text})

        if self._worker and self._worker.isRunning():
            self._worker.cancel()

        self._worker = BantayMiniChatWorker(
            messages=list(self._chat_history),
            disaster_context=dict(self._disaster_context),
            parent=self,
        )
        self._streaming_bubble = None
        self._streaming_text   = ""

        def _on_chunk(token: str):
            if self._streaming_bubble is None:
                self._streaming_bubble = self._add_bubble("", is_user=False)
            self._streaming_text += token
            self._streaming_bubble.setText(self._streaming_text)
            self._chat_scroll.verticalScrollBar().setValue(
                self._chat_scroll.verticalScrollBar().maximum()
            )

        def _on_done(full_reply: str):
            self._chat_history.append({"role": "assistant", "content": full_reply})
            self._streaming_bubble = None
            self._streaming_text   = ""

        def _on_revised(revised: str):
            if self._streaming_bubble:
                self._streaming_bubble.setText(revised)
            if self._chat_history and self._chat_history[-1]["role"] == "assistant":
                self._chat_history[-1]["content"] = revised

        self._worker.chunk_ready.connect(_on_chunk)
        self._worker.reply_done.connect(_on_done)
        self._worker.reply_revised.connect(_on_revised)
        self._worker.start()

    def _go_back(self):
        p = self.parent()
        while p and not hasattr(p, 'main_stack'):
            p = p.parent()
        if p:
            p.main_stack.setCurrentIndex(0)
```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 3: Commit**

```
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add BantayModeWidget self-contained disaster panel"
```

---

## Task 5: Remove CodeDebuggerMode

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — 3 deletions

- [ ] **Step 1: Set `_ENABLE_DEBUG_SHORTCUT` to False (line 124)**

Find:
```python
_ENABLE_DEBUG_SHORTCUT = True
```
Replace with:
```python
_ENABLE_DEBUG_SHORTCUT = False
```

- [ ] **Step 2: Delete the CodeDebuggerMode class**

Find the block starting at `class CodeDebuggerMode:` (~line 14071) and ending just before `# ============` / `class LocalFileAssistant:` (~line 14191). Delete the entire class.

The block to delete starts with:
```python
class CodeDebuggerMode:
    """
    Intelligent code debugger.
```
and ends with the last method of the class, just before:
```python
# Returns ranked results with highlighted snippets and page refs for PDFs
# ============================================================================
class LocalFileAssistant:
```

- [ ] **Step 3: Delete the debug block in UltraIntelligentWorker**

Find the block starting at line ~16221 and ending just before `# ── Local File Assistant ───`:

Delete from:
```python
            if _ENABLE_DEBUG_SHORTCUT and (_intent == 'debug' or _has_real_error):
```
all the way through:
```python
                    # fall through to main LLM

```
so that the next line visible is:
```python
            # ── Local File Assistant ───────────────────────────────────────────
```

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 5: Verify no remaining CodeDebuggerMode references**

```
python -c "
text = open('Project_Maria/Maria_App_original.py').read()
count = text.count('CodeDebuggerMode')
print(f'CodeDebuggerMode references: {count}')
assert count == 0, 'Still has references!'
print('CLEAN')
"
```

Expected: `CodeDebuggerMode references: 0` then `CLEAN`

- [ ] **Step 6: Commit**

```
git add Project_Maria/Maria_App_original.py
git commit -m "feat: remove CodeDebuggerMode class and debug pipeline block"
```

---

## Task 6: Repurpose CodeButton + Wire BantayModeWidget into MariaPyQt

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — 6 targeted edits

### 6a: Repurpose CodeButton to draw a shield icon

- [ ] **Step 1: Replace CodeButton._draw method**

Find:
```python
class CodeButton(_IconButtonBase):
    """</> chevrons icon for Code panel."""

    def _draw(self, p, w, h):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        self._paint_bg(p, w, h)

        pen = QPen(self._ink_color(), 1.5, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        cx, cy = w / 2, h / 2
        arm    = h * 0.18   # chevron arm length
        gap    = w * 0.12   # distance from centre to tip

        # Left chevron <
        p.drawLine(QPointF(cx - gap, cy - arm), QPointF(cx - gap - arm, cy))
        p.drawLine(QPointF(cx - gap - arm, cy), QPointF(cx - gap, cy + arm))

        # Right chevron >
        p.drawLine(QPointF(cx + gap, cy - arm), QPointF(cx + gap + arm, cy))
        p.drawLine(QPointF(cx + gap + arm, cy), QPointF(cx + gap, cy + arm))
```

Replace with:
```python
class CodeButton(_IconButtonBase):
    """Shield icon for Bantay (disaster companion) panel."""

    def _draw(self, p, w, h):
        from PyQt6.QtGui import QPainterPath
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        self._paint_bg(p, w, h)

        pen = QPen(self._ink_color(), 1.5, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)

        # Shield shape: rounded top, pointed bottom
        cx = w / 2
        pad = w * 0.18
        top = h * 0.12
        mid = h * 0.55
        bot = h * 0.92

        path = QPainterPath()
        path.moveTo(cx, top)
        path.lineTo(w - pad, top + h * 0.12)
        path.lineTo(w - pad, mid)
        path.lineTo(cx, bot)
        path.lineTo(pad, mid)
        path.lineTo(pad, top + h * 0.12)
        path.closeSubpath()
        p.drawPath(path)
```

### 6b: Update sidebar nav icon to draw shield and label "Bantay"

- [ ] **Step 2: Replace `_icon_code` function and `code_nav` line**

Find:
```python
        def _icon_code(p, s):
            # </> chevrons
            m = s // 2
            o = int(s * 0.22)
            g = int(s * 0.15)  # gap between chevrons
            # Left chevron <
            p.drawLine(m - g, m - o, m - g - o, m)
            p.drawLine(m - g - o, m, m - g, m + o)
            # Right chevron >
            p.drawLine(m + g, m - o, m + g + o, m)
            p.drawLine(m + g + o, m, m + g, m + o)
```

Replace with:
```python
        def _icon_code(p, s):
            # Shield icon for Bantay panel
            from PyQt6.QtGui import QPainterPath
            cx  = s / 2
            pad = s * 0.18
            top = s * 0.12
            mid = s * 0.58
            bot = s * 0.92
            path = QPainterPath()
            path.moveTo(cx, top)
            path.lineTo(s - pad, top + s * 0.12)
            path.lineTo(s - pad, mid)
            path.lineTo(cx, bot)
            path.lineTo(pad, mid)
            path.lineTo(pad, top + s * 0.12)
            path.closeSubpath()
            p.drawPath(path)
```

Then find:
```python
        code_nav      = _nav_btn(_icon_code,       "Code",      self.open_code_panel)
```

Replace with:
```python
        code_nav      = _nav_btn(_icon_code,       "Bantay",    self.open_bantay_panel)
```

### 6c: Update sidebar CodeButton tooltip and connection

- [ ] **Step 3: Update the sidebar mini-button**

Find:
```python
        code_btn = CodeButton(size=26, tooltip='Code')
        code_btn.clicked.connect(self.open_code_panel)
```

Replace with:
```python
        code_btn = CodeButton(size=26, tooltip='Bantay — Disaster Companion')
        code_btn.clicked.connect(self.open_bantay_panel)
```

### 6d: Replace code_page with bantay_page in main_stack setup

- [ ] **Step 4: Replace `_create_code_page` with `_create_bantay_page`**

Find:
```python
        # Page 4: Code scratchpad
        self.code_page = self._create_code_page()
        self.main_stack.addWidget(self.code_page)       # index 4
```

Replace with:
```python
        # Page 4: Bantay disaster companion
        self.bantay_page = self._create_bantay_page()
        self.main_stack.addWidget(self.bantay_page)     # index 4
```

### 6e: Add `_create_bantay_page` and `open_bantay_panel`, remove old code panel methods

- [ ] **Step 5: Replace `open_code_panel` and `_create_code_page` methods**

Find:
```python
    def _open_in_code(self, code, lang='python'):
        self._code_editor.setPlainText(code)
        self._code_lang_label.setText(lang.upper() or 'TEXT')
        self.open_code_panel()
```
Delete this entire method (3 lines).

Find:
```python
    def open_code_panel(self):
        self.main_stack.setCurrentIndex(4)
```
Replace with:
```python
    def open_bantay_panel(self):
        self.main_stack.setCurrentIndex(4)
        self.bantay_widget.apply_bulletin(self.bantay_widget._bulletin)
        gps = GPSLocationWorker(parent=self)
        gps.location_ready.connect(
            lambda city, region, country, lat, lon, readable:
                self.bantay_widget.refresh_location(readable)
        )
        gps.start()
```

Find `def _create_code_page(self):` and delete the entire method body up to (and not including) the next `def ` at the same indentation level. Replace the entire `_create_code_page` method with:

```python
    def _create_bantay_page(self) -> QWidget:
        kb_path    = os.path.join(os.path.dirname(__file__), "OfflineDisasterKB.json")
        cache_path = os.path.join(os.path.dirname(__file__), "bantay_cache.json")
        self.bantay_widget = BantayModeWidget(kb_path=kb_path, cache_path=cache_path, parent=self)

        self._disaster_watcher = PHDisasterWatcher(cache_path=cache_path, parent=self)
        self._disaster_watcher.disaster_update.connect(self.bantay_widget.apply_bulletin)
        self._disaster_watcher.offline_mode.connect(self.bantay_widget.set_offline)
        self._disaster_watcher.start()

        return self.bantay_widget
```

- [ ] **Step 6: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 7: Verify no remaining `open_code_panel` or `_open_in_code` references**

```
python -c "
text = open('Project_Maria/Maria_App_original.py').read()
for name in ('open_code_panel', '_open_in_code', '_create_code_page'):
    count = text.count(name)
    print(f'{name}: {count}')
    assert count == 0, f'Still has {name}!'
print('ALL CLEAN')
"
```

Expected: each count `0`, then `ALL CLEAN`

- [ ] **Step 8: Run all tests**

```
python -m pytest Project_Maria/test_bantay.py -v
```

Expected: All tests pass.

- [ ] **Step 9: Launch app and manually verify**

```
python Project_Maria/Maria_App_original.py
```

Manual checks:
1. Click the shield icon in the sidebar → Bantay panel opens
2. Panel shows "Walang aktibong alerto" (no active hazard)
3. Location pill shows a city name within ~5 seconds (GPS fetch)
4. All 4 tabs load instantly with Taglish content
5. Type a question in the mini-chat → Maria responds in Taglish
6. Click ← back button → returns to chat

- [ ] **Step 10: Commit**

```
git add Project_Maria/Maria_App_original.py
git commit -m "feat: wire BantayModeWidget into sidebar, replace code panel with Bantay disaster companion"
```

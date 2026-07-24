# Bantay Maria — Disaster Companion Feature Design

**Date:** 2026-05-07
**Status:** Approved
**Target:** Maria_App_original.py

---

## Overview

Bantay Maria is a self-contained disaster companion panel added to Maria. It covers all major Philippine hazards — typhoons, earthquakes, tsunamis, volcanic eruptions, floods, and landslides — with live agency data when connected and a full offline knowledge base when not. It is accessed via a shield icon repurposed from the existing Code button in the left sidebar.

The panel has no dependency on Ollama being running. All tabs load instantly from a bundled JSON file. An optional embedded mini-chat at the bottom uses `BantayMiniChatWorker` (modeled after `ArtifactMiniChatWorker`) with the current disaster context pre-loaded into the system prompt.

---

## Goals

- Give ordinary Filipino households (nanay, lolo, kids) fast, readable Taglish disaster guidance
- Work fully offline when cell towers or internet are down
- Cover all 5 Philippine hazard types from a single panel
- Be demo-ready for a pitch to DICT / UP Diliman / local startups
- Be newsworthy across tech media, mainstream broadcast, and social media

---

## What Changes in the Existing App

| Existing | Change |
|---|---|
| `CodeButton` (sidebar, line ~26192) | Repurposed → `BantayButton`, shield icon, same position |
| `CodeDebuggerMode` class (line ~14071) | Removed entirely |
| `_INTENT_CODE` chat routing | Unchanged — code questions via main chat still work |
| `EmergencySystem` class | Unchanged — coexists with Bantay panel |

---

## New Components

### 1. `PHDisasterWatcher` (QThread)

Polls three Philippine disaster agencies every 30 minutes when connected. Uses the existing `is_connected()` TCP probe to decide whether to run or sleep.

**Data sources:**
- PAGASA — typhoon bulletins, signal numbers per province, storm track
- PHIVOLCS — earthquake magnitude/depth/epicenter, tsunami watch status, volcanic alert levels
- NDRRMC — flood and landslide advisories, all-hazard bulletins

**Behavior:**
- On new bulletin: emits `disaster_update` signal to `BantayModeWidget`
- On no connectivity: emits `offline_mode` signal; widget falls back to cached bulletin or offline KB
- Caches last fetched bulletin to disk as `Project_Maria/bantay_cache.json` so it survives app restarts

---

### 2. `OfflineDisasterKB.json` (bundled data file)

A ~500KB JSON file shipped with the app. Covers all hazard types in plain Taglish. Structure:

```json
{
  "typhoon": {
    "signal_1": { "meaning": "...", "what_to_do": "...", "go_bag": [...] },
    "signal_2": { ... },
    "signal_3": { ... },
    "signal_4": { ... },
    "signal_5": { ... }
  },
  "earthquake": {
    "during": "Drop, Cover, Hold on...",
    "after": "Lumayo sa gusali...",
    "tsunami_watch": "Pumunta agad sa mataas na lugar..."
  },
  "tsunami": {
    "warning": "Lumayo na NGAYON sa dalampasigan...",
    "after": "Huwag bumalik hanggang..."
  },
  "volcanic": {
    "alert_1": { "meaning": "...", "what_to_do": "..." },
    "alert_2": { ... },
    "alert_3": { ... },
    "alert_4": { ... },
    "alert_5": { ... }
  },
  "flood": {
    "advisory": "...", "what_to_do": "...", "evacuation_tips": "..."
  },
  "first_aid": {
    "choking": "...", "drowning": "...", "wound": "...", "cpr": "..."
  },
  "go_bag": ["tubig para 3 araw", "pagkain", "gamot", ...],
  "hotlines": [
    { "name": "911 Emergency", "number": "911" },
    { "name": "NDRRMC", "number": "8911-5061" },
    { "name": "Philippine Red Cross", "number": "143" },
    { "name": "Coast Guard", "number": "8527-3877" },
    { "name": "PHIVOLCS", "number": "8426-1468" }
  ]
}
```

---

### 3. `BantayModeWidget` (QWidget — self-contained panel)

Opens when the user clicks `BantayButton`. Replaces the right panel content. No LLM required for any tab.

**Header (always visible):**
- Shield icon + "BANTAY MODE" label
- Location pill — auto-filled from existing `GPSLocationWorker`; editable fallback text input
- Live/Offline badge — green dot (live) or grey dot (offline)

**Hazard display:**
- Shows active hazard type and severity (signal level, magnitude, alert level)
- Adapts label and color per hazard:
  - Typhoon → signal number + typhoon name
  - Earthquake → magnitude + depth + epicenter
  - Tsunami → wave ETA + coastal warning
  - Volcanic → alert level + volcano name + danger zone radius
  - Flood/Landslide → advisory level + affected areas
- Falls back to "Walang aktibong alerto" when no hazard is active

**Four tabs (load from `OfflineDisasterKB.json` instantly):**

| Tab | Typhoon content | Earthquake content | Volcanic content |
|---|---|---|---|
| Ano Gagawin | Signal level guide | Drop-Cover-Hold steps | Alert level guide |
| Specific | Storm surge / flood warning | Tsunami watch status | Danger zone radius / ash fall |
| Go-Bag | Standard checklist | Standard checklist | Add ash mask, goggles |
| Hotlines | 911, NDRRMC, Red Cross | 911, PHIVOLCS | 911, PHIVOLCS, LGU |

**Bottom section — Magtanong kay Maria:**
- Embedded mini-chat area (scrollable bubble view)
- Text input + mic button (hooks into existing STT)
- Powered by `BantayMiniChatWorker`
- Maria's opening line on panel open: *"Handa ako. Anong tanong mo tungkol sa sitwasyon ngayon?"*

---

### 4. `BantayMiniChatWorker` (QThread)

Mirrors `ArtifactMiniChatWorker` structure exactly (same signals: `chunk_ready`, `reply_done`, `reply_revised`, `error_signal`, `status_changed`).

**One key difference:** The system prompt is pre-loaded with the current disaster context:

```
You are Maria, a Filipino AI disaster companion.
Current situation: [hazard type] · [severity] · [user location]
Answer ONLY in plain Taglish. Keep answers short and actionable.
Do not speculate. If unsure, refer to NDRRMC or 911.
```

Uses existing `HallucinationDetector` and `SelfCritiqueLoopEngine` from the main pipeline. Does not use web RAG — disaster context is already injected; speculation is actively suppressed.

---

## Data Flow

```
[BantayButton tap]
        │
        ▼
BantayModeWidget opens
        │
        ├─► GPSLocationWorker → location pill
        │
        ├─► ConnectivityGate (existing is_connected() TCP probe)
        │       ├── Connected → PHDisasterWatcher fetches latest bulletin
        │       │               → displays live hazard + severity
        │       └── Offline  → loads OfflineDisasterKB.json
        │                       → displays cached bulletin or "No active alert"
        │
        └─► Four tabs render immediately from OfflineDisasterKB.json
                   │
                   ▼
        [Optional] User taps Magtanong kay Maria
                   │
                   ▼
        BantayMiniChatWorker
        System prompt = current hazard + severity + location
        → streams Taglish response into embedded bubble view
```

---

## Hazard Coverage Matrix

| Hazard | Agency | Live Data | Offline KB | Tabs Adapt |
|---|---|---|---|---|
| Typhoon | PAGASA | Signal 1–5, storm track | Signal guides, surge warning | Yes |
| Earthquake | PHIVOLCS | Magnitude, depth, epicenter | Drop-Cover-Hold, aftershock | Yes |
| Tsunami | PHIVOLCS + PTWC | Wave ETA, coastal zones | Evacuation NOW guide | Yes |
| Volcanic eruption | PHIVOLCS | Alert 1–5, danger zone | Ash fall, evacuation | Yes |
| Flood / Landslide | NDRRMC | Advisory level, areas | Rainfall guide, evacuation | Yes |

---

## What Is NOT in Scope

- Barangay-level evacuation center mapping (requires LGU data partnership)
- Push notifications when app is closed (requires background service, out of scope v1)
- Multi-language support beyond Taglish (future)
- Real-time storm track map rendering (future)

---

## Files Affected

| File | Change |
|---|---|
| `Maria_App_original.py` | Remove `CodeDebuggerMode`; repurpose `CodeButton` → `BantayButton`; add `PHDisasterWatcher`, `BantayModeWidget`, `BantayMiniChatWorker` |
| `Project_Maria/OfflineDisasterKB.json` | New file — bundled Taglish knowledge base |
| (no new Python files) | All new classes go into the single existing file |

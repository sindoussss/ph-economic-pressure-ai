# Voice Call Adaptive UI — Design Spec
**Date:** 2026-05-05  
**File:** `Project_Maria/Maria_App_original.py`  
**Scope:** `CallDialog` class and its nested widget classes

---

## Goal

Replace the static `"Recognising..."` placeholder with a fully adaptive UI that tells the user what is happening at every moment during a voice call — whether the connection is fast or slow, whether the mic is working, and when Maria has heard them.

---

## Components

### 1. `_SignalBarsWidget` (new class)

A 3-bar mobile-signal-style widget placed in the **top bar** of `CallDialog`, right-aligned after `status_label`.

**Geometry:** ~22×14px. Three bars of increasing height (short / medium / tall), spaced 2px apart. Painted with `QPainter` in `paintEvent`.

**Bar states:**
| `set_bars(n)` | Meaning | Latency trigger |
|---|---|---|
| 3 | Fast | rolling avg < 800ms |
| 2 | Medium | 800ms – 2000ms |
| 1 | Slow | > 2000ms |
| 0 | Network error | `RequestError` |

**Colors:**
- Filled bar: `QColor(THEME['text_secondary'])` — charcoal
- Empty bar: `QColor(THEME['border_light'])` — faded

**Initialization:** starts at 3/3 (assumes good connection until proven otherwise).

**Data source:** `CallDialog._latency_samples` — a plain `list` capped at 5 entries. After each `recognize_google()` call, elapsed time is appended and the average is computed to set bar count.

---

### 2. `_MicActivityBar` (new class)

A thin (240×6px) rounded bar placed **between `waveform` and `speaking_label`** in the body layout. Gives the user visible proof that the mic loop is alive without requiring raw PyAudio amplitude access.

**States (controlled via `set_state(state: str)`):**

| State | Visual | Trigger |
|---|---|---|
| `"listening"` | Slow left-to-right shimmer (breathing animation via `QTimer` at 40ms) | Mic loop active, no speech yet |
| `"sending"` | Fully solid charcoal | `recognize_google()` in flight |
| `"hidden"` | Height collapses to 0 | Maria is speaking (`_is_speaking=True`) |

**Silence timeout:** If 15 seconds pass in `"listening"` state with no successful transcript, bar turns **amber** (`QColor(200,140,50)`) and `speaking_label` shows `"Can't hear you? Check mic in Windows Settings"`.

**Colors:**
- Listening shimmer: charcoal at alpha ~90, sweeping highlight at alpha ~200
- Sending: charcoal at alpha 180
- Silence warning: amber

---

### 3. `_PulseAvatar` color state

`_PulseAvatar` gains a `set_sending(on: bool)` method that modifies the cached `_ring_color` and `_advance` speed.

| `set_sending` | Ring color | Phase increment |
|---|---|---|
| `False` | charcoal `(63,63,63)` | `0.035` (current) |
| `True` | amber `(200,140,50)` | `0.021` (~60% speed — "working on it") |

The color interpolates via direct RGB assignment on the cached `QColor` object (no extra allocations). Snaps back to charcoal the instant `recognize_google()` returns.

---

### 4. Adaptive status text

The `speaking_label` during recognition updates in two ways:

**Immediate on `recognize_google()` start:**
- Set `speaking_label` to `"Sending to Maria..."`

**`QTimer` fires every 500ms while recognition is in flight:**
- If elapsed < 2s: keep `"Sending to Maria..."`
- If elapsed ≥ 2s: change to `"Still sending... slow connection"`

**On result:**
- Success: briefly show `"Maria heard you..."` for 800ms, then restore `"Listening..."`
- `UnknownValueError`: restore `"Listening..."` immediately
- `RequestError`: show `"Network error — check connection"`, signal bars → 0, wait 1s

---

### 5. Periodic mic re-calibration

Inside `_listen_loop`, track `_last_speech_time = time.monotonic()` (updated after each successful transcript).

**Recalibration trigger:** if `time.monotonic() - _last_speech_time > 30` and the loop is in idle listening (not inside `r.listen()` timeout):

1. Show `"Recalibrating mic..."` in `speaking_label` for the duration
2. Call `r.adjust_for_ambient_noise(source, duration=0.5)` (same open stream, in-place)
3. After calibration, read `r.energy_threshold` and set dynamic `pause_threshold`:
   - `energy_threshold > 500` (noisy): `r.pause_threshold = 0.5`
   - `energy_threshold < 200` (quiet): `r.pause_threshold = 1.0`
   - Otherwise: `r.pause_threshold = 0.8` (default)
4. Reset `_last_speech_time` so the 30s clock restarts

---

## Data flow

```
_listen_loop (background thread)
  │
  ├─ [idle] → _MicActivityBar.set_state("listening")
  │            mic_silence_timer increments
  │            if silence > 15s → bar goes amber, hint shown
  │            if silence > 30s → recalibrate
  │
  ├─ [r.listen() returns audio]
  │    ├─ _MicActivityBar.set_state("sending")
  │    ├─ avatar_canvas.set_sending(True)   ← via QTimer.singleShot(0, ...)
  │    ├─ speaking_label = "Sending to Maria..."
  │    ├─ slow_timer.start(500ms)
  │    │
  │    └─ recognize_google(audio)
  │         ├─ record elapsed → _latency_samples → update signal bars
  │         ├─ [success] → slow_timer.stop()
  │         │              speaking_label = "Maria heard you..."
  │         │              avatar.set_sending(False)
  │         │              MicActivityBar.set_state("hidden")
  │         │              _handle_user_speech(text)
  │         │              _last_speech_time = now
  │         │
  │         ├─ [UnknownValueError] → restore "Listening...", set_sending(False)
  │         └─ [RequestError] → signal bars = 0, show error, set_sending(False)
  │
  └─ [_is_speaking clears] → _MicActivityBar.set_state("listening"), reset silence timer
```

---

## UI layout changes

**Top bar** (existing `top_layout` HBoxLayout):
```
[red dot] [yellow dot] [green dot]  [stretch]  [status_label]  [_SignalBarsWidget]
```

**Body** (between waveform and speaking_label):
```
[waveform: _WaveformWidget 300×44]
[spacing 6]
[_MicActivityBar 240×6, centered]
[spacing 6]
[speaking_label]
```

No other layout changes. Dialog size stays at 440×580.

---

## Thread safety

All UI updates from `_listen_loop` (background thread) go through `QTimer.singleShot(0, lambda: ...)` — same pattern already used by `_show_speaking`. No direct widget calls from the background thread.

---

## Out of scope

- Raw PyAudio amplitude metering (avoided — would conflict with `speech_recognition`'s internal stream)
- Streaming partial transcripts (not available in free Google STT API)
- Wake word detection

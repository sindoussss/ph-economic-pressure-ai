# Voice Call Adaptive UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a latency-adaptive status layer to the voice call dialog — signal bars, orb color change, mic activity bar, adaptive text, and periodic mic re-calibration — so the user always knows what is happening regardless of connection speed.

**Architecture:** All new UI lives in three new widget classes (`_SignalBarsWidget`, `_MicActivityBar`, and additions to `_PulseAvatar`) inserted near the existing call widget helpers. `CallDialog._listen_loop` drives all state changes via `QTimer.singleShot(0, fn)` for thread safety. A `_slow_timer` QTimer on the main thread upgrades the "Sending to Maria..." text when the call takes longer than 2 s.

**Tech Stack:** PyQt6, Python `speech_recognition`, `time.monotonic()` for latency measurement.

---

## File map

| File | Change |
|---|---|
| `Project_Maria/Maria_App_original.py` | Add 3 widget classes, modify `CallDialog` (7 methods) |
| `Project_Maria/test_adaptive_voice_ui.py` | New — pure-logic unit tests (no Qt required) |

---

### Task 1: Pure-logic helpers and tests

Extract the two decision functions from the design as standalone helpers in the test file. Run tests to establish a green baseline before touching the app.

**Files:**
- Create: `Project_Maria/test_adaptive_voice_ui.py`

- [ ] **Step 1: Create the test file with helpers and all tests**

```python
# Project_Maria/test_adaptive_voice_ui.py
"""Unit tests for adaptive voice call UI logic.

Run: pytest Project_Maria/test_adaptive_voice_ui.py -v
No Qt installation required.
"""


def _bars_from_latency(samples: list) -> int:
    """Compute signal-bar count from a rolling latency sample list."""
    if not samples:
        return 3
    avg = sum(samples) / len(samples)
    return 3 if avg < 0.8 else (2 if avg < 2.0 else 1)


def _pause_threshold_from_energy(et: float) -> float:
    """Compute dynamic pause_threshold from post-calibration energy_threshold."""
    return 0.5 if et > 500 else (1.0 if et < 200 else 0.8)


class TestBarsFromLatency:
    def test_fast_gives_3_bars(self):
        assert _bars_from_latency([0.3, 0.4, 0.5]) == 3

    def test_medium_gives_2_bars(self):
        assert _bars_from_latency([0.9, 1.2, 1.5]) == 2

    def test_slow_gives_1_bar(self):
        assert _bars_from_latency([2.5, 3.0, 2.8]) == 1

    def test_empty_gives_3_bars(self):
        assert _bars_from_latency([]) == 3

    def test_boundary_0_8_is_medium(self):
        assert _bars_from_latency([0.8]) == 2

    def test_boundary_2_0_is_slow(self):
        assert _bars_from_latency([2.0]) == 1

    def test_rolling_average_used(self):
        # one fast + four slow → avg > 2.0 → 1 bar
        assert _bars_from_latency([0.1, 2.5, 2.5, 2.5, 2.5]) == 1


class TestPauseThreshold:
    def test_noisy_room_lowers_threshold(self):
        assert _pause_threshold_from_energy(600) == 0.5

    def test_quiet_room_raises_threshold(self):
        assert _pause_threshold_from_energy(150) == 1.0

    def test_normal_room_uses_default(self):
        assert _pause_threshold_from_energy(350) == 0.8

    def test_boundary_501_is_noisy(self):
        assert _pause_threshold_from_energy(501) == 0.5

    def test_boundary_199_is_quiet(self):
        assert _pause_threshold_from_energy(199) == 1.0
```

- [ ] **Step 2: Run tests — expect all green**

```
pytest Project_Maria/test_adaptive_voice_ui.py -v
```

Expected output: `12 passed`

- [ ] **Step 3: Commit**

```bash
git add Project_Maria/test_adaptive_voice_ui.py
git commit -m "test: add pure-logic tests for adaptive voice call UI"
```

---

### Task 2: `_SignalBarsWidget` class

Insert immediately before `class _PulseAvatar` (~line 31351).

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` at the blank line before `class _PulseAvatar`

- [ ] **Step 1: Insert `_SignalBarsWidget` before `class _PulseAvatar`**

Find this exact anchor in the file:
```python
class _PulseAvatar(QWidget):
    """Warm-theme orb - soft charcoal rings on off-white, matches main UI."""
```

Insert the following block BEFORE it (keep one blank line between them):

```python
class _SignalBarsWidget(QWidget):
    """3-bar mobile-signal indicator driven by observed STT latency.

    set_bars(n): n=3 fast, n=2 medium, n=1 slow, n=0 no signal.
    Starts at 3 (optimistic) — self-calibrates as recognize_google() returns.
    """

    _BAR_W  = 4
    _BAR_HS = (4, 8, 13)   # heights: short / medium / tall
    _GAP    = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bars = 3
        total_w = 3 * self._BAR_W + 2 * self._GAP
        self.setFixedSize(total_w, 14)

    def set_bars(self, n: int) -> None:
        n = max(0, min(3, n))
        if n != self._bars:
            self._bars = n
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        color_on  = QColor(63, 63, 63, 200)
        color_off = QColor(63, 63, 63,  50)
        x = 0
        h_total = self.height()
        for i in range(3):
            bh    = self._BAR_HS[i]
            by    = h_total - bh
            color = color_on if i < self._bars else color_off
            p.setBrush(QBrush(color))
            p.drawRoundedRect(x, by, self._BAR_W, bh, 1, 1)
            x += self._BAR_W + self._GAP
        p.end()


```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 3: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add _SignalBarsWidget for latency-adaptive signal indicator"
```

---

### Task 3: `_MicActivityBar` class

Insert immediately after `_SignalBarsWidget` (before `class _PulseAvatar`).

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — same insertion zone

- [ ] **Step 1: Insert `_MicActivityBar` between `_SignalBarsWidget` and `_PulseAvatar`**

Find the anchor:
```python
class _PulseAvatar(QWidget):
    """Warm-theme orb - soft charcoal rings on off-white, matches main UI."""
```

Insert BEFORE it (after `_SignalBarsWidget`):

```python
class _MicActivityBar(QWidget):
    """Thin 240×6 px animated bar between waveform and speaking_label.

    States
    ------
    'listening' — slow left-to-right shimmer (loop is alive)
    'sending'   — fully solid charcoal (recognize_google in flight)
    'warning'   — amber shimmer + 15 s silence hint
    'hidden'    — invisible (Maria is speaking or call ended)
    """

    _COLOR_BASE    = QColor(63,  63,  63,  90)
    _COLOR_SHIMMER = QColor(63,  63,  63, 200)
    _COLOR_SEND    = QColor(63,  63,  63, 180)
    _COLOR_WARN    = QColor(200, 140,  50, 140)
    _COLOR_WARN_SH = QColor(200, 140,  50, 230)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(240, 6)
        self._state = "hidden"
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setVisible(False)

    def set_state(self, state: str) -> None:
        if state == self._state:
            return
        self._state = state
        if state == "hidden":
            self._timer.stop()
            self.setVisible(False)
        elif state == "sending":
            self._timer.stop()
            self.setVisible(True)
            self.update()
        else:  # 'listening' or 'warning'
            self.setVisible(True)
            self._timer.start(40)

    def _tick(self):
        self._phase = (self._phase + 0.04) % 1.0
        self.update()

    def paintEvent(self, event):
        if self._state == "hidden":
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        w, h = self.width(), self.height()
        r    = h / 2.0

        if self._state == "sending":
            p.setBrush(QBrush(self._COLOR_SEND))
            p.drawRoundedRect(0, 0, w, h, r, r)
        else:
            base    = self._COLOR_WARN    if self._state == "warning" else self._COLOR_BASE
            shimmer = self._COLOR_WARN_SH if self._state == "warning" else self._COLOR_SHIMMER
            # Base bar
            p.setBrush(QBrush(base))
            p.drawRoundedRect(0, 0, w, h, r, r)
            # Sweeping highlight (60 px wide, wraps around)
            sw = 60
            x  = int(self._phase * (w + sw)) - sw
            grad = QLinearGradient(x, 0, x + sw, 0)
            tr   = QColor(shimmer.red(), shimmer.green(), shimmer.blue(), 0)
            grad.setColorAt(0.0, tr)
            grad.setColorAt(0.5, shimmer)
            grad.setColorAt(1.0, tr)
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(0, 0, w, h, r, r)
        p.end()


```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 3: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add _MicActivityBar with listening/sending/warning/hidden states"
```

---

### Task 4: `_PulseAvatar.set_sending()` method

Add `_sending` state and `_phase_step` var to `_PulseAvatar` so the orb can turn amber and slow down while recognition is in flight.

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — `_PulseAvatar.__init__` and `_advance`

- [ ] **Step 1: Add `_phase_step` and `_sending` to `_PulseAvatar.__init__`**

Find:
```python
        self._ring_color    = QColor(63, 63, 63, 0)    # alpha updated in-place per ring
        self._ring_pen      = QPen(self._ring_color, 1.5)
```

Replace with:
```python
        self._ring_color    = QColor(63, 63, 63, 0)    # alpha updated in-place per ring
        self._ring_pen      = QPen(self._ring_color, 1.5)
        self._phase_step    = 0.035   # varied by set_sending()
        self._sending       = False
```

- [ ] **Step 2: Make `_advance` use `_phase_step`**

Find:
```python
    def _advance(self):
        self._phase = (self._phase + 0.035) % (2 * math.pi)
        self.update()
```

Replace with:
```python
    def _advance(self):
        self._phase = (self._phase + self._phase_step) % (2 * math.pi)
        self.update()
```

- [ ] **Step 3: Add `set_sending()` method after `_advance`**

Find:
```python
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2, self.height() / 2
        base_r = self._BASE_R
```

Insert BEFORE it:
```python
    def set_sending(self, on: bool) -> None:
        """Shift orb rings to amber + slow phase when STT is in flight."""
        self._sending = on
        if on:
            self._ring_color.setRgb(200, 140, 50)
            self._phase_step = 0.021
        else:
            self._ring_color.setRgb(63, 63, 63)
            self._phase_step = 0.035

```

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 5: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add _PulseAvatar.set_sending() for amber orb state during STT"
```

---

### Task 5: `CallDialog.__init__` additions

Add the three new instance variables needed by later tasks: `_latency_samples`, `_slow_timer`, and `_recog_start`.

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — `CallDialog.__init__`

- [ ] **Step 1: Add new instance vars after `_stt_available`**

Find:
```python
        self._coqui_tts     = None
        self._listening_thread = None
        self._stop_listening   = False
        self._is_speaking      = False
        self._tts_available    = False
        self._stt_available    = False

        # NOTE: _init_engines() is NOT called here
```

Replace with:
```python
        self._coqui_tts     = None
        self._listening_thread = None
        self._stop_listening   = False
        self._is_speaking      = False
        self._tts_available    = False
        self._stt_available    = False

        self._latency_samples: list = []   # rolling window, max 5 entries
        self._recog_start:     float = 0.0  # monotonic timestamp when recognize_google starts
        self._slow_timer = QTimer(self)
        self._slow_timer.setInterval(500)
        self._slow_timer.timeout.connect(self._on_slow_timer)

        # NOTE: _init_engines() is NOT called here
```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 3: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add _latency_samples/_slow_timer/_recog_start to CallDialog.__init__"
```

---

### Task 6: `CallDialog._setup_ui` — wire up new widgets

Add `_SignalBarsWidget` to the top bar and `_MicActivityBar` to the body.

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — `CallDialog._setup_ui`

- [ ] **Step 1: Add signal bars to the top bar**

Find:
```python
        top_layout.addWidget(self.status_label)
        layout.addWidget(top_bar)
```

Replace with:
```python
        top_layout.addWidget(self.status_label)
        top_layout.addSpacing(6)
        self._signal_bars = _SignalBarsWidget(top_bar)
        top_layout.addWidget(self._signal_bars)
        layout.addWidget(top_bar)
```

- [ ] **Step 2: Add mic activity bar between waveform and speaking_label**

Find:
```python
        body.addLayout(wrow)
        body.addSpacing(10)

        self.speaking_label = QLabel("")
```

Replace with:
```python
        body.addLayout(wrow)
        body.addSpacing(6)

        self._mic_bar = _MicActivityBar(self)
        mic_row = QHBoxLayout()
        mic_row.addStretch()
        mic_row.addWidget(self._mic_bar)
        mic_row.addStretch()
        body.addLayout(mic_row)
        body.addSpacing(4)

        self.speaking_label = QLabel("")
```

- [ ] **Step 3: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 4: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: wire _SignalBarsWidget and _MicActivityBar into CallDialog layout"
```

---

### Task 7: `_on_slow_timer`, `_toggle_mute`, `_end_call` updates

Add the main-thread timer slot and keep mute/end-call in sync with the new widgets.

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — three methods in `CallDialog`

- [ ] **Step 1: Add `_on_slow_timer` method after `_start_listening_loop`**

Find:
```python
    def _listen_loop(self):
        import time
        try:
            import speech_recognition as sr
```

Insert BEFORE it:
```python
    def _on_slow_timer(self):
        """Fires every 500 ms while recognize_google() is in flight.
        Upgrades status text once elapsed >= 2 s, then stops itself.
        """
        elapsed = time.monotonic() - self._recog_start
        if elapsed >= 2.0:
            self.speaking_label.setText("Still sending... slow connection")
            self._slow_timer.stop()

```

- [ ] **Step 2: Update `_toggle_mute` to hide mic bar when muted**

Find:
```python
    def _toggle_mute(self):
        self.is_muted = not self.is_muted
        self.mute_btn.setText("🔇" if self.is_muted else "🎙️")
        self._stop_listening = self.is_muted
        if not self.is_muted and self._stt_available:
            self._start_listening_loop()
        self._show_speaking("Muted" if self.is_muted else "")
```

Replace with:
```python
    def _toggle_mute(self):
        self.is_muted = not self.is_muted
        self.mute_btn.setText("🔇" if self.is_muted else "🎙️")
        self._stop_listening = self.is_muted
        if self.is_muted:
            self._mic_bar.set_state("hidden")
        if not self.is_muted and self._stt_available:
            self._start_listening_loop()
        self._show_speaking("Muted" if self.is_muted else "")
```

- [ ] **Step 3: Update `_end_call` to stop slow_timer**

Find:
```python
    def _end_call(self):
        self.is_active = False
        self._stop_listening = True
        self._wave_timer.stop()
        self._orb_timer.stop()
        try:
```

Replace with:
```python
    def _end_call(self):
        self.is_active = False
        self._stop_listening = True
        self._wave_timer.stop()
        self._orb_timer.stop()
        self._slow_timer.stop()
        try:
```

- [ ] **Step 4: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 5: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: add _on_slow_timer, update _toggle_mute and _end_call for new widgets"
```

---

### Task 8: Rewrite `_listen_loop` with all adaptive logic

This is the core task — replaces the current flat loop with latency tracking, signal bar updates, mic bar state changes, adaptive status text, periodic recalibration, and dynamic `pause_threshold`.

**Files:**
- Modify: `Project_Maria/Maria_App_original.py` — `CallDialog._listen_loop`

- [ ] **Step 1: Replace `_listen_loop` entirely**

Find the entire method (from `def _listen_loop(self):` through the final `print("Google STT listen loop ended")`):

```python
    def _listen_loop(self):
        import time
        try:
            import speech_recognition as sr
        except ImportError:
            self._show_speaking("pip install SpeechRecognition pyaudio")
            return

        r = sr.Recognizer()
        r.pause_threshold = 0.8
        r.dynamic_energy_threshold = True

        self._show_speaking("Listening...")
        print("Google STT listen loop started")

        try:
            mic = sr.Microphone()
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                while self.is_active and not self._stop_listening:
                    if getattr(self, "_is_speaking", False):
                        time.sleep(0.1)
                        continue
                    try:
                        audio = r.listen(source, timeout=5, phrase_time_limit=20)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Mic listen error: {e}")
                        break

                    if not self.is_active or self._stop_listening:
                        break

                    self._show_speaking("Recognising...")
                    try:
                        text = r.recognize_google(audio).strip()
                        if text:
                            print("Google STT result:", repr(text))
                            self._is_speaking = True
                            self._handle_user_speech(text)
                            while getattr(self, "_is_speaking", False) and self.is_active:
                                time.sleep(0.1)
                            if self.is_active and not self._stop_listening:
                                self._show_speaking("Listening...")
                    except sr.UnknownValueError:
                        if self.is_active and not self._stop_listening:
                            self._show_speaking("Listening...")
                    except sr.RequestError as e:
                        self._show_speaking(f"Network error: {e}")
                        time.sleep(1)
                    except Exception as e:
                        print(f"STT error: {e}")
                        if self.is_active:
                            self._show_speaking("Listening...")
        except Exception as e:
            print(f"Mic open error: {e}")
            self._show_speaking(f"Mic error: {e}")
        finally:
            print("Google STT listen loop ended")
```

Replace with:

```python
    def _listen_loop(self):
        try:
            import speech_recognition as sr
        except ImportError:
            self._show_speaking("pip install SpeechRecognition pyaudio")
            return

        r = sr.Recognizer()
        r.pause_threshold          = 0.8
        r.dynamic_energy_threshold = True

        # Shorthand: schedule fn on main thread (safe cross-thread UI call)
        def _ui(fn):
            QTimer.singleShot(0, fn)

        last_speech_time = time.monotonic()
        last_recal_time  = time.monotonic()
        silence_warned   = False

        self._show_speaking("Listening...")
        print("Google STT listen loop started")

        try:
            mic = sr.Microphone()
            with mic as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                _ui(lambda: self._mic_bar.set_state("listening"))

                while self.is_active and not self._stop_listening:
                    if getattr(self, "_is_speaking", False):
                        time.sleep(0.1)
                        continue

                    # ── 15 s silence warning ─────────────────────────────
                    if time.monotonic() - last_speech_time > 15 and not silence_warned:
                        silence_warned = True
                        _ui(lambda: self._mic_bar.set_state("warning"))
                        _ui(lambda: self._show_speaking(
                            "Can't hear you? Check mic in Windows Settings"))

                    # ── 30 s periodic recalibration ──────────────────────
                    if time.monotonic() - last_recal_time > 30:
                        _ui(lambda: self._show_speaking("Recalibrating mic..."))
                        r.adjust_for_ambient_noise(source, duration=0.5)
                        et = r.energy_threshold
                        r.pause_threshold = 0.5 if et > 500 else (1.0 if et < 200 else 0.8)
                        last_recal_time = time.monotonic()
                        if silence_warned:
                            silence_warned = False
                            _ui(lambda: self._mic_bar.set_state("listening"))
                        _ui(lambda: self._show_speaking("Listening..."))

                    # ── Capture audio ────────────────────────────────────
                    try:
                        audio = r.listen(source, timeout=5, phrase_time_limit=20)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        print(f"Mic listen error: {e}")
                        break

                    if not self.is_active or self._stop_listening:
                        break

                    # ── Recognition in flight ────────────────────────────
                    self._recog_start = time.monotonic()
                    _ui(lambda: self._mic_bar.set_state("sending"))
                    _ui(lambda: self.avatar_canvas.set_sending(True))
                    _ui(lambda: self._show_speaking("Sending to Maria..."))
                    _ui(lambda: self._slow_timer.start())

                    try:
                        text    = r.recognize_google(audio).strip()
                        elapsed = time.monotonic() - self._recog_start

                        _ui(lambda: self._slow_timer.stop())
                        _ui(lambda: self.avatar_canvas.set_sending(False))

                        # Update signal bars from observed latency
                        self._latency_samples.append(elapsed)
                        if len(self._latency_samples) > 5:
                            self._latency_samples.pop(0)
                        avg  = sum(self._latency_samples) / len(self._latency_samples)
                        bars = 3 if avg < 0.8 else (2 if avg < 2.0 else 1)
                        _ui(lambda b=bars: self._signal_bars.set_bars(b))

                        if text:
                            print("Google STT result:", repr(text))
                            last_speech_time = time.monotonic()
                            last_recal_time  = time.monotonic()
                            silence_warned   = False
                            _ui(lambda: self._show_speaking("Maria heard you..."))
                            _ui(lambda: self._mic_bar.set_state("hidden"))
                            self._is_speaking = True
                            self._handle_user_speech(text)
                            while getattr(self, "_is_speaking", False) and self.is_active:
                                time.sleep(0.1)
                            # 800 ms "Maria heard you..." pause before restoring
                            QTimer.singleShot(800, lambda: (
                                self._mic_bar.set_state("listening") or
                                self._show_speaking("Listening...")
                            ) if self.is_active and not self._stop_listening else None)

                    except sr.UnknownValueError:
                        _ui(lambda: self._slow_timer.stop())
                        _ui(lambda: self.avatar_canvas.set_sending(False))
                        _ui(lambda: self._mic_bar.set_state("listening"))
                        if self.is_active and not self._stop_listening:
                            _ui(lambda: self._show_speaking("Listening..."))

                    except sr.RequestError as e:
                        _ui(lambda: self._slow_timer.stop())
                        _ui(lambda: self.avatar_canvas.set_sending(False))
                        _ui(lambda: self._signal_bars.set_bars(0))
                        _ui(lambda: self._mic_bar.set_state("listening"))
                        self._show_speaking("Network error — check connection")
                        time.sleep(1)

                    except Exception as e:
                        _ui(lambda: self._slow_timer.stop())
                        _ui(lambda: self.avatar_canvas.set_sending(False))
                        _ui(lambda: self._mic_bar.set_state("listening"))
                        print(f"STT error: {e}")
                        if self.is_active:
                            _ui(lambda: self._show_speaking("Listening..."))

        except Exception as e:
            print(f"Mic open error: {e}")
            self._show_speaking(f"Mic error: {e}")
        finally:
            _ui(lambda: self._slow_timer.stop())
            _ui(lambda: self.avatar_canvas.set_sending(False))
            _ui(lambda: self._mic_bar.set_state("hidden"))
            print("Google STT listen loop ended")
```

- [ ] **Step 2: Verify syntax**

```
python -c "import ast; ast.parse(open('Project_Maria/Maria_App_original.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

Expected: `SYNTAX OK`

- [ ] **Step 3: Run the unit tests — all should still pass**

```
pytest Project_Maria/test_adaptive_voice_ui.py -v
```

Expected: `12 passed`

- [ ] **Step 4: Commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: rewrite _listen_loop with adaptive latency UI, recalibration, and mic bar"
```

---

### Task 9: Smoke-test the full call UI

Launch the app, open the voice call, and verify each visual element works.

**Files:** none — manual verification only

- [ ] **Step 1: Launch the app**

```
python Project_Maria/Maria_App_original.py
```

Expected: app opens without errors in the terminal.

- [ ] **Step 2: Open the voice call dialog**

Click the phone/call button in the main UI to open `CallDialog`.

Verify on open:
- Top bar shows 3 filled signal bars (charcoal, right of "Connected" label)
- Mic activity bar is hidden initially, appears as slow shimmer once "Listening..." starts
- Orb rings are charcoal as normal
- Status shows "Listening..." after the greeting

- [ ] **Step 3: Speak a phrase**

Say a short phrase ("hello" or "kamusta").

Verify:
- Mic bar goes solid charcoal as soon as audio is captured
- Orb rings shift to amber and slow down
- Speaking label shows "Sending to Maria..."
- After transcript returns: speaking label briefly shows "Maria heard you..." then "Listening..."
- Orb snaps back to charcoal
- Signal bars update (should be 3/3 on a normal connection)

- [ ] **Step 4: Simulate slow-timer text** (optional)

On a slow connection or by temporarily throttling network: keep mic open and wait for `recognize_google` to take >2 s. Verify speaking label changes to "Still sending... slow connection".

- [ ] **Step 5: Test 15 s silence warning**

Stay silent for 15 seconds during "Listening..." state. Verify:
- Mic bar turns amber shimmer
- Speaking label shows "Can't hear you? Check mic in Windows Settings"

- [ ] **Step 6: Final commit**

```bash
git add Project_Maria/Maria_App_original.py
git commit -m "feat: voice call adaptive UI complete — signal bars, mic bar, amber orb, recalibration"
```

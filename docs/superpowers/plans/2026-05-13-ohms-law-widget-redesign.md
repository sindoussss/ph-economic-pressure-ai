# Ohm's Law Widget Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three-column Ohm's Law chat widget with a compact two-panel card: left side has V/R sliders and derived-I formula, right side has a live scrolling sine-wave oscilloscope.

**Architecture:** All changes are confined to `Project_Maria/Maria_App.py`. `_OhmsLawVisualization` is deleted and replaced with a new `_OhmsLawOscilloscope` painter widget. `_OhmsLawWidget` is rewritten to a two-panel layout. The chat card builder `_insert_ohms_law_chat_widget` loses its "Open large" button and footer. Public API (`widget_data`, `apply_widget_data`, `state_changed`) is unchanged.

**Tech Stack:** Python 3, PyQt6, `math` (already imported)

---

## File Map

| File | What changes |
|------|-------------|
| `Project_Maria/Maria_App.py:25652–25790` | Delete `_OhmsLawVisualization`, insert `_OhmsLawOscilloscope` |
| `Project_Maria/Maria_App.py:25793–26068` | Rewrite `_OhmsLawWidget` — new `__init__`, new `_build_ui`, new `_make_slider_row`, update `_update_display`, `widget_data`, `apply_widget_data`; delete `_make_col` and animation fields |
| `Project_Maria/Maria_App.py:38086–38222` | Simplify `_insert_ohms_law_chat_widget` — remove pop-out button and footer |

---

## Task 1: Replace `_OhmsLawVisualization` with `_OhmsLawOscilloscope`

**Files:**
- Modify: `Project_Maria/Maria_App.py:25652–25790`

- [ ] **Step 1: Delete the entire `_OhmsLawVisualization` class**

  Remove lines 25652–25790 (the full `_OhmsLawVisualization` class, from `class _OhmsLawVisualization(QWidget):` through the final `p.end()` and blank line before `class _OhmsLawWidget`).

- [ ] **Step 2: Insert `_OhmsLawOscilloscope` in its place**

  Paste the following class at the same location (immediately before `class _OhmsLawWidget`):

  ```python
  class _OhmsLawOscilloscope(QWidget):
      """Scrolling sine-wave panel — right side of the Ohm's Law widget."""

      def __init__(self, parent=None):
          super().__init__(parent)
          self._voltage = 9.0
          self._current = 3.0
          self._phase   = 0.0
          self.setFixedWidth(200)
          self._timer = QTimer(self)
          self._timer.setInterval(33)
          self._timer.timeout.connect(self._tick)
          self._timer.start()

      def _tick(self):
          self._phase += 1.6 + self._current * 0.15
          self.update()

      def set_values(self, voltage: float, current: float):
          self._voltage = max(0.0, float(voltage))
          self._current = max(0.0, float(current))

      def paintEvent(self, event):
          p = QPainter(self)
          p.setRenderHint(QPainter.RenderHint.Antialiasing)
          w, h = self.width(), self.height()

          p.setPen(Qt.PenStyle.NoPen)
          p.setBrush(QBrush(QColor(250, 250, 249)))
          p.drawRect(QRectF(0, 0, w, h))

          grid_pen = QPen(QColor(0, 0, 0, 10))
          grid_pen.setWidthF(1.0)
          p.setPen(grid_pen)
          for i in range(1, 4):
              p.drawLine(QPointF(0, h * i / 4), QPointF(w, h * i / 4))
          for i in range(1, 5):
              p.drawLine(QPointF(w * i / 5, 0), QPointF(w * i / 5, h))

          p.setPen(QPen(QColor(0, 0, 0, 15)))
          p.drawLine(QPointF(0, h / 2), QPointF(w, h / 2))

          amp  = h * 0.26 * min(1.0, self._voltage / 24.0)
          freq = 0.032 + self._current * 0.002

          wave = QPainterPath()
          for x in range(w + 1):
              y = h / 2 + amp * math.sin((x + self._phase) * freq)
              if x == 0:
                  wave.moveTo(x, y)
              else:
                  wave.lineTo(x, y)

          wp = QPen(QColor(60, 140, 100, 190))
          wp.setWidthF(1.6)
          wp.setCapStyle(Qt.PenCapStyle.RoundCap)
          p.setPen(wp)
          p.setBrush(Qt.BrushStyle.NoBrush)
          p.drawPath(wave)

          lf = QFont("Segoe UI", 7)
          p.setFont(lf)
          p.setPen(QColor(200, 200, 200))
          p.drawText(QRectF(0, h - 18, w - 6, 14),
                     Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                     f"P = {self._voltage * self._current:.1f} W")
          p.end()
  ```

- [ ] **Step 3: Smoke-check the file parses**

  ```bash
  python -c "import ast, sys; ast.parse(open('Project_Maria/Maria_App.py').read()); print('OK')"
  ```

  Expected: `OK` with no errors.

- [ ] **Step 4: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "refactor: replace OhmsLawVisualization with OhmsLawOscilloscope"
  ```

---

## Task 2: Rewrite `_OhmsLawWidget`

**Files:**
- Modify: `Project_Maria/Maria_App.py:25793–26068`

The inputs change: V and R are now the two sliders; I is derived (I = V/R). The `_make_col` method is deleted. The animation timer is removed (no longer needed — I is computed directly from sliders). Public methods `widget_data()` and `apply_widget_data()` keep the same signature.

- [ ] **Step 1: Replace the `_OhmsLawWidget` class body**

  Delete everything from `class _OhmsLawWidget(QWidget):` through the end of the class (line 26068, the `def _animate_current_step` stub), and replace with:

  ```python
  class _OhmsLawWidget(QWidget):
      """Compact two-panel Ohm's Law widget — left controls, right oscilloscope."""

      state_changed   = pyqtSignal(dict)
      _MIN_RESISTANCE = 0.1

      def __init__(self, parent=None):
          super().__init__(parent)
          self._building = False
          self._build_ui()
          self.apply_widget_data({"voltage": 9, "resistance": 3.0})

      def paintEvent(self, event):
          painter = QPainter(self)
          painter.setPen(Qt.PenStyle.NoPen)
          painter.setBrush(QBrush(QColor(252, 252, 250)))
          painter.drawRect(QRectF(0, 0, self.width(), self.height()))
          painter.end()

      # ── UI construction ────────────────────────────────────────────────────

      def _make_slider_row(self, label_text: str, unit: str, layout,
                           *, range_max: int = 240):
          """Build a label/value header + slim QSlider row, add to layout.

          Returns (slider, value_label).
          """
          row = QWidget()
          row.setStyleSheet("background: transparent;")
          rl = QVBoxLayout(row)
          rl.setContentsMargins(0, 0, 0, 0)
          rl.setSpacing(3)

          top = QWidget()
          top.setStyleSheet("background: transparent;")
          tl = QHBoxLayout(top)
          tl.setContentsMargins(0, 0, 0, 0)
          tl.setSpacing(0)

          lbl = QLabel(label_text)
          lf  = QFont("Segoe UI", 8)
          lbl.setFont(lf)
          lbl.setStyleSheet("color: #bbbbbb; background: transparent;")
          tl.addWidget(lbl)
          tl.addStretch()

          val_lbl = QLabel(f"— {unit}")
          vf = QFont("Segoe UI", 8)
          vf.setWeight(QFont.Weight.DemiBold)
          val_lbl.setFont(vf)
          val_lbl.setStyleSheet("color: #555555; background: transparent;")
          tl.addWidget(val_lbl)
          rl.addWidget(top)

          slider = QSlider(Qt.Orientation.Horizontal)
          slider.setRange(1, range_max)
          slider.setSingleStep(1)
          slider.setCursor(Qt.CursorShape.PointingHandCursor)
          slider.setFixedHeight(16)
          slider.setStyleSheet("""
              QSlider::groove:horizontal {
                  height: 3px; border-radius: 1.5px; background: #eeece6;
              }
              QSlider::sub-page:horizontal {
                  background: #888888; border-radius: 1.5px;
              }
              QSlider::handle:horizontal {
                  width: 10px; height: 10px; margin: -3.5px 0;
                  border-radius: 5px; background: #ffffff;
                  border: 2px solid #666666;
              }
              QSlider::handle:horizontal:hover  { border-color: #444444; }
              QSlider::handle:horizontal:pressed { background: #f0f0f0; }
          """)
          slider.valueChanged.connect(self._update_display)
          rl.addWidget(slider)

          layout.addWidget(row)
          return slider, val_lbl

      def _build_ui(self):
          self.setStyleSheet("background: transparent;")

          root = QVBoxLayout(self)
          root.setContentsMargins(0, 0, 0, 0)
          root.setSpacing(0)

          body = QWidget()
          body.setStyleSheet("background: transparent;")
          body_l = QHBoxLayout(body)
          body_l.setContentsMargins(0, 0, 0, 0)
          body_l.setSpacing(0)

          # ── Left: controls ─────────────────────────────────────────────────
          left = QWidget()
          left.setStyleSheet("background: transparent;")
          left_l = QVBoxLayout(left)
          left_l.setContentsMargins(18, 14, 18, 14)
          left_l.setSpacing(0)

          formula = QLabel("V = IR")
          ff = QFont("Georgia", 13)
          ff.setItalic(True)
          formula.setFont(ff)
          formula.setStyleSheet("color: #555555; background: transparent;")
          left_l.addWidget(formula)
          left_l.addSpacing(10)

          self._voltage_slider, self._voltage_val_lbl = \
              self._make_slider_row("Vs", "V", left_l, range_max=240)
          left_l.addSpacing(8)

          self._resistance_slider, self._resistance_val_lbl = \
              self._make_slider_row("R", "Ω", left_l, range_max=200)
          left_l.addSpacing(10)

          self._derived_label = QLabel()
          self._derived_label.setTextFormat(Qt.TextFormat.RichText)
          df = QFont("Segoe UI", 9)
          self._derived_label.setFont(df)
          self._derived_label.setStyleSheet("color: #aaaaaa; background: transparent;")
          left_l.addWidget(self._derived_label)
          left_l.addStretch()

          # ── Vertical divider ────────────────────────────────────────────────
          vdiv = QFrame()
          vdiv.setFrameShape(QFrame.Shape.VLine)
          vdiv.setFixedWidth(1)
          vdiv.setStyleSheet("background: #eeece6; color: #eeece6;")

          # ── Right: oscilloscope ─────────────────────────────────────────────
          self._oscilloscope = _OhmsLawOscilloscope()

          body_l.addWidget(left, 1)
          body_l.addWidget(vdiv)
          body_l.addWidget(self._oscilloscope)

          root.addWidget(body)

      # ── Display update ─────────────────────────────────────────────────────

      def _update_display(self):
          v = self._voltage_slider.value() / 10.0
          r = max(self._MIN_RESISTANCE, self._resistance_slider.value() / 10.0)
          i = v / r

          self._voltage_val_lbl.setText(f"{v:.1f} V")
          self._resistance_val_lbl.setText(f"{r:.1f} Ω")
          self._derived_label.setText(
              f'I = V/R = '
              f'<span style="color:#777777">{v:.1f}</span> / '
              f'<span style="color:#777777">{r:.1f}</span> = '
              f'<span style="color:#222222; font-weight:700">{i:.2f} A</span>')
          self._oscilloscope.set_values(v, i)

          if not self._building:
              self.state_changed.emit(self.widget_data())

      # ── Public API (unchanged signatures) ──────────────────────────────────

      def widget_data(self) -> dict:
          v = self._voltage_slider.value() / 10.0
          r = max(self._MIN_RESISTANCE, self._resistance_slider.value() / 10.0)
          i = v / r
          return {"voltage": round(v, 2), "resistance": round(r, 1), "current": round(i, 2)}

      def apply_widget_data(self, data: dict):
          data = data or {}
          voltage    = float(data.get("voltage", 9.0))
          resistance = max(self._MIN_RESISTANCE, float(data.get("resistance", 3.0)))
          if "current" in data and "voltage" not in data:
              voltage = float(data["current"]) * resistance
          self._building = True
          self._voltage_slider.setValue(
              max(1, min(240, int(round(voltage * 10)))))
          self._resistance_slider.setValue(
              max(1, min(200, int(round(resistance * 10)))))
          self._building = False
          self._update_display()
  ```

- [ ] **Step 2: Verify the file still parses**

  ```bash
  python -c "import ast, sys; ast.parse(open('Project_Maria/Maria_App.py').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 3: Launch app and trigger the widget**

  Run `python Project_Maria/Maria_App.py`, send a message like `"show me ohm's law"` and confirm:
  - Card appears with white background, two panels separated by a thin line
  - Left panel has italic `V = IR`, two slim sliders labelled `Vs` and `R`, and the derived equation `I = V/R = ... = ... A`
  - Right panel shows a scrolling green sine wave on cream background with a faint grid
  - Dragging either slider updates all labels and the wave's amplitude/speed in real time

- [ ] **Step 4: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "feat: redesign OhmsLawWidget — compact two-panel layout with oscilloscope"
  ```

---

## Task 3: Simplify `_insert_ohms_law_chat_widget`

**Files:**
- Modify: `Project_Maria/Maria_App.py` — `_insert_ohms_law_chat_widget` method (~line 38086)

Remove the "Open large" button, its connect, and the footer ("Live simulation…") from the chat card.

- [ ] **Step 1: Simplify the header block**

  Find this block inside `_insert_ohms_law_chat_widget` (around line 38117):

  ```python
          # Header
          hdr = QWidget(card)
          hdr.setStyleSheet("background: transparent;")
          hdr_l = QHBoxLayout(hdr)
          hdr_l.setContentsMargins(20, 16, 16, 12)
          hdr_l.setSpacing(10)

          title_col = QVBoxLayout()
          title_col.setSpacing(2)
          title = QLabel("Ohm's Law Explorer")
          title.setStyleSheet("""
              color: #0f172a;
              font-size: 16px; font-weight: 700;
              font-family: Georgia, 'Times New Roman', serif;
              background: transparent; letter-spacing: 0.3px;
          """)
          sub = QLabel("I = V / R  —  interactive circuit")
          sub.setStyleSheet("""
              color: #9a9890;
              font-size: 10.5px; font-style: italic;
              font-family: Georgia, serif;
              background: transparent;
          """)
          title_col.addWidget(title)
          title_col.addWidget(sub)
          hdr_l.addLayout(title_col, 1)

          popout_btn = QPushButton("Open large")
          popout_btn.setCursor(Qt.CursorShape.PointingHandCursor)
          popout_btn.setFixedHeight(28)
          popout_btn.setStyleSheet("""
              QPushButton {
                  background: transparent;
                  border: 1px solid #dddad2;
                  border-radius: 14px;
                  color: #888880;
                  padding: 0 12px;
                  font-size: 10px; font-weight: 600;
                  font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
              }
              QPushButton:hover { background: #f5f3ee; color: #444440; border-color: #c8c5bc; }
          """)
          popout_btn.clicked.connect(
              lambda: self.open_ohms_law_widget(
                  getattr(shell, '_pending_widget_data', widget_data)))
          hdr_l.addWidget(popout_btn, 0, Qt.AlignmentFlag.AlignVCenter)
          card_layout.addWidget(hdr)
  ```

  Replace it with:

  ```python
          # Header
          hdr = QWidget(card)
          hdr.setStyleSheet("background: transparent;")
          hdr_l = QHBoxLayout(hdr)
          hdr_l.setContentsMargins(18, 12, 18, 10)
          hdr_l.setSpacing(10)

          title = QLabel("Ohm's Law")
          title.setStyleSheet("""
              color: #1a1a1a;
              font-size: 12px; font-weight: 700;
              font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
              background: transparent; letter-spacing: 0.3px;
          """)
          sub = QLabel("I = V / R")
          sub.setStyleSheet("""
              color: #bbbbbb;
              font-size: 10px; font-style: italic;
              font-family: Georgia, serif;
              background: transparent;
          """)
          hdr_l.addWidget(title)
          hdr_l.addWidget(sub)
          hdr_l.addStretch()
          card_layout.addWidget(hdr)
  ```

- [ ] **Step 2: Remove the footer**

  Find and delete the following block (around line 38192):

  ```python
          div2 = QFrame(card)
          div2.setFrameShape(QFrame.Shape.HLine)
          div2.setFixedHeight(1)
          div2.setStyleSheet("background: #eeece6; color: #eeece6;")
          card_layout.addWidget(div2)

          footer = QWidget(card)
          footer.setStyleSheet("background: transparent;")
          footer_l = QHBoxLayout(footer)
          footer_l.setContentsMargins(20, 8, 20, 12)
          footer_l.setSpacing(6)
          live_dot = QLabel("●")
          live_dot.setStyleSheet("color: #4ade80; font-size: 8px; background: transparent;")
          footer_l.addWidget(live_dot, 0, Qt.AlignmentFlag.AlignVCenter)
          caption = QLabel("Live simulation — values and particles update as you drag.")
          caption.setStyleSheet("""
              color: #b0ada4;
              font-size: 10.5px;
              font-family: 'Segoe UI Variable', 'Segoe UI', sans-serif;
              background: transparent;
          """)
          footer_l.addWidget(caption, 1)
          card_layout.addWidget(footer)
  ```

- [ ] **Step 3: Verify the file parses**

  ```bash
  python -c "import ast, sys; ast.parse(open('Project_Maria/Maria_App.py').read()); print('OK')"
  ```

  Expected: `OK`

- [ ] **Step 4: Launch and verify the card**

  Run `python Project_Maria/Maria_App.py`, ask `"explain ohm's law"` and confirm:
  - Card has the slim header `Ohm's Law  I = V / R` with no button
  - No footer row at the bottom of the card
  - Widget body looks correct (two panels, oscilloscope animating)

- [ ] **Step 5: Commit**

  ```bash
  git add Project_Maria/Maria_App.py
  git commit -m "feat: remove Open Large button and footer from Ohm's Law chat card"
  ```

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel,
    QSizePolicy, QVBoxLayout, QWidget,
)


# ── Palette ───────────────────────────────────────────────────────────────────
_STABLE   = '#27AE60'
_RISING   = '#E0A84A'
_HIGH     = '#E74C3C'
_CARD_BG  = '#FFFFFF'
_PAGE_BG  = '#F7F8FA'
_BORDER   = '#EAEAEA'
_TEXT_DIM = '#999999'
_TEXT_HI  = '#1A1A2E'

_PRESSURE_COLOR = {'Stable': _STABLE, 'Rising': _RISING, 'High': _HIGH, 'Critical': _HIGH}


def _pressure_color(pressure: str) -> str:
    return _PRESSURE_COLOR.get(pressure, _RISING)


def _card(radius: int = 12) -> QFrame:
    f = QFrame()
    f.setStyleSheet(
        f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:{radius}px;'
    )
    return f


def _label(text: str, size: int = 11, bold: bool = False, color: str = _TEXT_HI) -> QLabel:
    lbl = QLabel(text)
    weight = '700' if bold else '400'
    lbl.setStyleSheet(f'font-size:{size}px; font-weight:{weight}; color:{color}; border:none;')
    return lbl


# ── Sparkline canvas ──────────────────────────────────────────────────────────

class _Sparkline(FigureCanvasQTAgg):
    def __init__(self, color: str, parent=None):
        fig = Figure(figsize=(2.5, 0.9), dpi=90)
        fig.patch.set_facecolor(_CARD_BG)
        self._ax = fig.add_axes([0, 0.1, 1, 0.85])
        self._ax.set_facecolor(_CARD_BG)
        self._ax.axis('off')
        self._color = color
        super().__init__(fig)
        self.setFixedHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def plot(self, values: list[float]):
        self._ax.clear()
        self._ax.axis('off')
        if len(values) >= 2:
            xs = list(range(len(values)))
            self._ax.plot(xs, values, color=self._color, linewidth=2.0)
            self._ax.fill_between(xs, values, min(values), alpha=0.15, color=self._color)
        self.draw()


# ── Sector card ───────────────────────────────────────────────────────────────

class SectorCard(QFrame):
    def __init__(self, title: str, unit: str, spark_color: str, parent=None):
        super().__init__(parent)
        self._unit = unit
        self.setStyleSheet(
            f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:14px;'
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(4)

        self._title_lbl = _label(title.upper(), size=10, color=_TEXT_DIM)
        lay.addWidget(self._title_lbl)

        self._value_lbl = _label('—', size=28, bold=True)
        lay.addWidget(self._value_lbl)

        self._delta_lbl = _label('—', size=13, color=_TEXT_DIM)
        lay.addWidget(self._delta_lbl)

        self._spark = _Sparkline(spark_color)
        lay.addWidget(self._spark)

        self._signal_lbl = _label('—', size=11, color=_TEXT_DIM)
        lay.addWidget(self._signal_lbl)

        self._pending()

    def _pending(self):
        self._value_lbl.setText('Analyzing…')
        self._value_lbl.setStyleSheet(
            f'font-size:18px; font-weight:400; color:{_TEXT_DIM}; border:none;'
        )

    def update_data(
        self,
        value: float,
        delta: float,
        history: list[float],
        signal_text: str,
        pressure: str,
    ):
        color = _pressure_color(pressure)
        self._value_lbl.setText(f'{value:.2f} {self._unit}')
        self._value_lbl.setStyleSheet(
            f'font-size:26px; font-weight:700; color:{_TEXT_HI}; border:none;'
        )
        arrow = '↑' if delta >= 0 else '↓'
        sign  = '+' if delta >= 0 else ''
        self._delta_lbl.setText(f'{sign}{delta:.2f}  {arrow}')
        self._delta_lbl.setStyleSheet(
            f'font-size:13px; font-weight:600; color:{color}; border:none;'
        )
        self._spark.plot(history)
        self._signal_lbl.setText(signal_text)


# ── Weather panel ─────────────────────────────────────────────────────────────

class _WeatherPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:14px;'
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)
        lay.addWidget(_label('WEATHER SIGNAL', size=10, color=_TEXT_DIM))

        self._rain_lbl = _label('Rainfall: —', size=12)
        self._temp_lbl = _label('Avg Temp: —', size=12)
        lay.addWidget(self._rain_lbl)
        lay.addWidget(self._temp_lbl)

        fig = Figure(figsize=(3, 0.7), dpi=90)
        fig.patch.set_facecolor(_CARD_BG)
        self._ax = fig.add_axes([0.02, 0.1, 0.96, 0.85])
        self._ax.set_facecolor(_CARD_BG)
        self._ax.axis('off')
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedHeight(65)
        lay.addWidget(self._canvas)

    def update_data(self, rainfall_history: list[float], temp_history: list[float]):
        if rainfall_history:
            self._rain_lbl.setText(f'Rainfall: {rainfall_history[-1]:.0f} mm')
        if temp_history:
            self._temp_lbl.setText(f'Avg Temp: {temp_history[-1]:.1f} °C')
        self._ax.clear()
        self._ax.axis('off')
        if len(rainfall_history) >= 2:
            xs = list(range(len(rainfall_history)))
            self._ax.bar(xs, rainfall_history, color='#4A90E2', alpha=0.7, width=0.8)
        self._canvas.draw()


# ── Gas→Food influence panel ──────────────────────────────────────────────────

class _InfluencePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f'background:{_CARD_BG}; border:1px solid {_BORDER}; border-radius:14px;'
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)
        lay.addWidget(_label('GAS → FOOD INFLUENCE', size=10, color=_TEXT_DIM))

        self._transport_lbl = _label('Transport cost: —', size=12)
        self._rainfall_lbl  = _label('Rainfall deficit: —', size=12)
        lay.addWidget(self._transport_lbl)
        lay.addWidget(self._rainfall_lbl)

        fig = Figure(figsize=(3, 0.7), dpi=90)
        fig.patch.set_facecolor(_CARD_BG)
        self._ax = fig.add_axes([0.05, 0.1, 0.92, 0.85])
        self._ax.set_facecolor(_CARD_BG)
        self._canvas = FigureCanvasQTAgg(fig)
        self._canvas.setFixedHeight(65)
        lay.addWidget(self._canvas)

    def update_data(self, gas_delta: float, rainfall_deficit_pct: float):
        transport = gas_delta * 0.22
        rainfall  = rainfall_deficit_pct * 0.15
        self._transport_lbl.setText(f'Transport cost:    {transport:+.2f} idx pts')
        self._rainfall_lbl.setText( f'Rainfall deficit:  {rainfall:+.2f} idx pts')
        self._ax.clear()
        labels = ['Transport', 'Rainfall']
        values = [transport, rainfall]
        colors = [_RISING, '#4A90E2']
        bars = self._ax.barh(labels, values, color=colors, height=0.5)
        self._ax.axvline(0, color=_TEXT_DIM, linewidth=0.8)
        self._ax.axis('off')
        self._canvas.draw()


# ── Macro summary banner ──────────────────────────────────────────────────────

class _SummaryBanner(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f'background:#FFFBF0; border:1px solid {_BORDER};'
            'border-left:4px solid #E0A84A; border-radius:10px;'
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 14)
        self._lbl = QLabel('Awaiting sector analysis…')
        self._lbl.setStyleSheet(
            f'font-size:12px; font-style:italic; color:{_TEXT_DIM}; border:none;'
        )
        self._lbl.setWordWrap(True)
        lay.addWidget(self._lbl)

    def set_text(self, text: str):
        self._lbl.setText(text)
        self._lbl.setStyleSheet(
            f'font-size:12px; font-style:italic; color:{_TEXT_HI}; border:none;'
        )


# ── Main Economy Overview widget ──────────────────────────────────────────────

class EconomyOverviewWidget(QWidget):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self._df = df
        self.setStyleSheet(f'background:{_PAGE_BG};')

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)

        # ── Macro summary banner ──────────────────────────────────────────────
        self._summary = _SummaryBanner()
        outer.addWidget(self._summary)

        # ── Three sector cards ────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(14)

        self._gas_card  = SectorCard('Gas',                  '₱/L',   '#4A90E2')
        self._food_card = SectorCard('Food Index (derived)', 'pts',   '#27AE60')
        self._elec_card = SectorCard('Electricity (derived)', '₱/kWh', '#E0A84A')

        # Honest framing: only gas is independently forecast. Food and electricity
        # are deterministic pass-through transforms of the gas price, not
        # independent predictions — make that explicit on hover.
        _derived_tip = ('Derived from the gas price via a fixed pass-through '
                        'coefficient — not an independent forecast.')
        self._food_card.setToolTip(_derived_tip)
        self._elec_card.setToolTip(_derived_tip)

        for card in (self._gas_card, self._food_card, self._elec_card):
            cards_row.addWidget(card)
        outer.addLayout(cards_row, stretch=3)

        # ── Bottom row: influence + weather ───────────────────────────────────
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(14)
        self._influence = _InfluencePanel()
        self._weather   = _WeatherPanel()
        bottom_row.addWidget(self._influence, stretch=1)
        bottom_row.addWidget(self._weather,   stretch=1)
        outer.addLayout(bottom_row, stretch=2)

        # Pre-populate from historical data
        self._populate_from_df()

    def _history(self, col: str, n: int = 6) -> list[float]:
        if col not in self._df.columns:
            return []
        return self._df[col].dropna().tail(n).tolist()

    def _populate_from_df(self):
        df = self._df

        if 'gas_price' in df.columns and len(df) >= 2:
            hist = self._history('gas_price')
            delta = float(df['gas_price'].iloc[-1] - df['gas_price'].iloc[-2])
            self._gas_card.update_data(
                value=float(df['gas_price'].iloc[-1]),
                delta=delta,
                history=hist,
                signal_text='Pressure: —',
                pressure='Rising' if delta > 0 else 'Stable',
            )

        if 'food_price_idx' in df.columns and len(df) >= 2:
            hist = self._history('food_price_idx')
            delta = float(df['food_price_idx'].iloc[-1] - df['food_price_idx'].iloc[-2])
            self._food_card.update_data(
                value=float(df['food_price_idx'].iloc[-1]),
                delta=delta,
                history=hist,
                signal_text=f'Rainfall: {df["rainfall_mm"].iloc[-1]:.0f} mm' if 'rainfall_mm' in df.columns else '—',
                pressure='Rising' if delta > 0 else 'Stable',
            )

        if 'electricity_rate' in df.columns and len(df) >= 2:
            hist = self._history('electricity_rate')
            delta = float(df['electricity_rate'].iloc[-1] - df['electricity_rate'].iloc[-2])
            self._elec_card.update_data(
                value=float(df['electricity_rate'].iloc[-1]),
                delta=delta,
                history=hist,
                signal_text='Fuel share: 18%',
                pressure='Rising' if delta > 0 else 'Stable',
            )

        if 'rainfall_mm' in df.columns:
            self._weather.update_data(
                rainfall_history=self._history('rainfall_mm'),
                temp_history=self._history('temp_c') if 'temp_c' in df.columns else [],
            )

        if 'gas_price' in df.columns and len(df) >= 2:
            gas_delta = float(df['gas_price'].iloc[-1] - df['gas_price'].iloc[-2])
            rain_norm = 100.0
            rain_actual = float(df['rainfall_mm'].iloc[-1]) if 'rainfall_mm' in df.columns else rain_norm
            deficit_pct = max(0.0, (rain_norm - rain_actual) / rain_norm)
            self._influence.update_data(gas_delta, deficit_pct)

    # ── Public update slots ───────────────────────────────────────────────────

    def update_gas(self, result: dict):
        """Called when gas debate completes. result keys: value, delta, history, pressure, verdict."""
        self._gas_card.update_data(
            value=result.get('value', 0.0),
            delta=result.get('delta', 0.0),
            history=result.get('history', []),
            signal_text=f'Pressure: {result.get("pressure", "—")}',
            pressure=result.get('pressure', 'Stable'),
        )

    def update_food(self, result: dict):
        self._food_card.update_data(
            value=result.get('value', 0.0),
            delta=result.get('delta', 0.0),
            history=result.get('history', []),
            signal_text=result.get('signal_text', '—'),
            pressure=result.get('pressure', 'Stable'),
        )

    def update_electricity(self, result: dict):
        self._elec_card.update_data(
            value=result.get('value', 0.0),
            delta=result.get('delta', 0.0),
            history=result.get('history', []),
            signal_text='Fuel share: 18%',
            pressure=result.get('pressure', 'Stable'),
        )

    def update_summary(self, text: str):
        self._summary.set_text(text)

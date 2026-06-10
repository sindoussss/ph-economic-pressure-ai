# coding: utf-8
"""Pure formatting of the three sector forecasts for display (no PyQt).

A run produces a next-month estimate for each sector in its own unit:
gas/fuel ₱/L, food %, electricity ₱/kWh. This turns the raw numbers into
display rows; any sector may be None (debate unavailable) -> shown as an em dash.
"""
from typing import Optional

# (key, label, value format string)
_SECTORS = [
    ('gas',  'Gas / fuel',  '{:+.2f} ₱/L'),
    ('food', 'Food',        '{:+.2f} %'),
    ('elec', 'Electricity', '{:+.4f} ₱/kWh'),
]


def _direction(v: Optional[float]) -> str:
    if v is None:
        return 'na'
    if v > 0:
        return 'up'
    if v < 0:
        return 'down'
    return 'flat'


def sector_forecast_rows(gas: Optional[float] = None, food: Optional[float] = None,
                         elec: Optional[float] = None) -> list:
    """Return one display row per sector (always gas, food, elec in that order):
    {key, label, value, value_str, direction}. direction in up/down/flat/na."""
    vals = {'gas': gas, 'food': food, 'elec': elec}
    rows = []
    for key, label, fmt in _SECTORS:
        v = vals[key]
        rows.append({
            'key': key,
            'label': label,
            'value': v,
            'value_str': fmt.format(v) if v is not None else '—',
            'direction': _direction(v),
        })
    return rows

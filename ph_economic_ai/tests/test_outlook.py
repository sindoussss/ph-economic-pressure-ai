import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ph_economic_ai.engine.pressure_brief import PressureBrief, SectorReading
from ph_economic_ai.engine.outlook import (
    ForecastResult, forecast_outlook, sector_basis)


def _brief():
    return PressureBrief(
        as_of='2026-07-24', window='this_week',
        readings=[
            SectorReading('gas', 'rising', 1.0, '₱/L', 100, ['news'], ['RedditPH']),
            SectorReading('food', 'rising', 0.5, '%', 90, ['news'], ['NFARiceRetail']),
            SectorReading('electricity', 'rising', 0.3, '₱/kWh', 80, ['news'], ['MeralcoCharge']),
        ],
        narrative='Pressure rising.')


_REPORT = {
    'audit': [{'target': 'fuel', 'verdict': 'efficient'}],
    'electricity_nowcast': {'driver_edge_robust': True},
    'food_nowcast': {'mom': {'verdict': 'beats_best_naive'}},
    'conformal_widths': {'0.9': 3.0},
}


def test_verdict_gate_maps_each_sector():
    assert sector_basis('gas', _REPORT) == 'efficient'
    assert sector_basis('electricity', _REPORT) == 'mechanical'
    assert sector_basis('food', _REPORT) == 'own-dynamics'


def test_no_report_defaults_to_efficient():
    out = forecast_outlook(_brief(), {}, tournament=None)
    assert all(s.basis == 'efficient' for s in out.sectors)


def test_tournament_number_is_bounded_to_the_present_read():
    """A wild tournament number (+50/L) must be reeled back toward the present read
    (+1.0), while the raw number is kept for transparency."""
    def wild(sector, prior, scenario):
        if sector == 'gas':
            return ForecastResult(point=50.0, agreement=80, raw=50.0)
        return ForecastResult(point=prior, agreement=60, raw=prior)

    out = forecast_outlook(_brief(), _REPORT, tournament=wild)
    gas = next(s for s in out.sectors if s.sector == 'gas')
    assert gas.tournament_estimate == 50.0        # raw preserved
    assert abs(gas.point) < 10.0                  # bounded back toward +1.0
    assert gas.basis == 'efficient'
    assert 'no exploitable edge' in gas.note
    assert gas.interval == [round(gas.point - 3.0, 2), round(gas.point + 3.0, 2)]  # conformal band
    assert gas.agreement == 80                    # carried, labeled not-a-probability


def test_naive_fallback_persists_the_present_read():
    out = forecast_outlook(_brief(), _REPORT, tournament=None)
    by = {s.sector: s for s in out.sectors}
    assert by['gas'].point == 1.0 and by['gas'].agreement == 100
    assert by['food'].point == 0.5 and by['food'].basis == 'own-dynamics'
    assert by['electricity'].point == 0.3 and by['electricity'].basis == 'mechanical'
    assert out.horizon == 'next month'            # monthly only — never weekly/daily


def test_outlook_serialises():
    out = forecast_outlook(_brief(), _REPORT, tournament=None)
    d = out.to_dict()
    assert set(d) == {'as_of', 'horizon', 'sectors'}
    assert d['sectors'][0]['sector'] == 'gas' and 'note' in d['sectors'][0]

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from unittest.mock import MagicMock, patch

from ph_economic_ai.engine.pressure_brief import PressureBrief, SectorReading
from ph_economic_ai.engine.outlook import (
    ForecastResult, forecast_outlook, make_swarm_tournament, sector_basis)


def _brief():
    return PressureBrief(
        as_of='2026-07-24', window='this_week',
        readings=[
            SectorReading('gas', 'rising', 1.0, '₱/L', 100,
                          drivers=['news'], sources=['RedditPH']),
            SectorReading('food', 'rising', 0.5, '%', 90,
                          drivers=['news'], sources=['NFARiceRetail']),
            SectorReading('electricity', 'rising', 0.3, '₱/kWh', 80,
                          drivers=['news'], sources=['MeralcoCharge']),
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


def test_swarm_tournament_feeds_real_market_data_to_the_swarm():
    """RSK follow-up: forum.run_monitor never fetches Brent/USD-PHP, only the
    social snapshot, so the scenario forecast_outlook builds has no oil_pct/
    usd_pct at all. Without deriving them from a live brief here,
    compute_physical_anchor's scenario.get('oil_pct', 0.0) silently defaults to
    zero and the swarm reconciles every estimate against a fabricated PHP 0.00
    anchor. This pins that the tournament now fetches a live brief and passes
    both the derived scenario and the brief itself through to SwarmOrchestrator."""
    fake_brief = MagicMock()
    fake_mv = MagicMock(final_estimate=1.23, confidence_pct=77)

    with patch('ph_economic_ai.engine.live_data.LiveDataBrief') as MockBrief, \
         patch('ph_economic_ai.engine.live_data.derive_scenario_from_brief',
              return_value={'oil_pct': 4.5, 'usd_pct': -1.2,
                            'bsp_rate': 6.5, 'demand_index': 80.0,
                            'current_price': 98.82}) as mock_derive, \
         patch('ph_economic_ai.engine.swarm.SwarmOrchestrator') as MockOrch:
        MockBrief.return_value.fetch.return_value = fake_brief
        MockOrch.return_value.run.return_value = fake_mv

        tournament = make_swarm_tournament(rag=object())
        result = tournament('gas', 1.0, {'as_of': '2026-08-08',
                                         'window': 'this_week', 'sector': 'gas'})

    assert result.point == 1.23 and result.agreement == 77
    mock_derive.assert_called_once_with(fake_brief)
    _, kwargs = MockOrch.call_args
    scenario_passed = MockOrch.call_args[0][1]
    assert scenario_passed['oil_pct'] == 4.5 and scenario_passed['usd_pct'] == -1.2
    assert scenario_passed['sector'] == 'gas'          # original scenario keys kept
    assert kwargs['data_brief'] is fake_brief


def test_swarm_tournament_survives_a_failed_live_fetch():
    """A dead network must degrade to the pre-fix default (no brief), not fail
    the forecast outright."""
    fake_mv = MagicMock(final_estimate=0.5, confidence_pct=40)

    with patch('ph_economic_ai.engine.live_data.LiveDataBrief') as MockBrief, \
         patch('ph_economic_ai.engine.swarm.SwarmOrchestrator') as MockOrch:
        MockBrief.return_value.fetch.side_effect = Exception('network down')
        MockOrch.return_value.run.return_value = fake_mv

        tournament = make_swarm_tournament(rag=object())
        result = tournament('gas', 1.0, {'sector': 'gas'})

    assert result.point == 0.5
    _, kwargs = MockOrch.call_args
    assert kwargs['data_brief'] is None
    scenario_passed = MockOrch.call_args[0][1]
    assert scenario_passed['oil_pct'] == 0.0    # derive_scenario_from_brief(None) default


def test_outlook_serialises():
    out = forecast_outlook(_brief(), _REPORT, tournament=None)
    d = out.to_dict()
    assert set(d) == {'as_of', 'horizon', 'sectors'}
    assert d['sectors'][0]['sector'] == 'gas' and 'note' in d['sectors'][0]

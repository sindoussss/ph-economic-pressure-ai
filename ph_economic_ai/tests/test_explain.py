"""Why / when / where, without letting the model invent any of it.

The risk this guards is specific. Asked to explain a price move, a model will
supply a fluent mechanism it was never given, and an invented reason is worse
than no reason because it reads as evidence. Every explanation is therefore
checked against the sources the forum actually read.
"""
import pytest

from ph_economic_ai.engine import explain
from ph_economic_ai.engine.pressure_brief import SectorReading


def _reading(sector='gas', estimate=-0.60, unit='PHP/L', drivers=None, sources=None):
    return SectorReading(
        sector=sector, direction='easing' if (estimate or 0) < 0 else 'rising',
        estimate=estimate, unit=unit, confidence=54,
        drivers=drivers if drivers is not None else ['Brent crude fell about 3 percent'],
        sources=sources if sources is not None else ['DOE', 'BusinessWorld'])


# ── grounding ────────────────────────────────────────────────────────────────

def test_an_invented_source_is_caught():
    bad = lambda msgs, **kw: 'Prices fall because OPEC and the IMF announced a deal.'
    out = explain.build_explanation(_reading(), complete=bad)
    assert out['grounded'] is False
    assert set(out['ungrounded_mentions']) == {'IMF', 'OPEC'}


def test_an_explanation_using_only_given_sources_is_grounded():
    good = lambda msgs, **kw: 'Gas is going down because Brent crude fell about 3 percent.'
    out = explain.build_explanation(_reading(), complete=good)
    assert out['grounded'] is True
    assert out['ungrounded_mentions'] == []


def test_the_readings_own_sources_are_allowed():
    cites = lambda msgs, **kw: 'DOE and BusinessWorld both report a drop.'
    out = explain.build_explanation(_reading(), complete=cites)
    assert out['grounded'] is True


def test_units_and_common_words_are_not_treated_as_citations():
    txt = lambda msgs, **kw: 'PHP per L is easing across NCR, tracking CPI.'
    assert explain.check_grounding(txt(None), ['DOE'])['grounded'] is True


# ── degrading honestly ───────────────────────────────────────────────────────

def test_no_model_still_produces_a_grounded_explanation():
    out = explain.build_explanation(_reading(), complete=None)
    assert out['why']
    assert out['why_generated'] is False
    assert out['grounded'] is True


def test_a_failing_model_falls_back_rather_than_raising():
    boom = lambda msgs, **kw: (_ for _ in ()).throw(RuntimeError('offline'))
    out = explain.build_explanation(_reading(), complete=boom)
    assert out['why_generated'] is False
    assert 'Brent' in out['why'] or 'brent' in out['why']


def test_no_drivers_says_so_instead_of_inventing_one():
    out = explain.build_explanation(_reading(drivers=[]), complete=None)
    assert 'did not agree on a clear driver' in out['why']


def test_a_model_is_not_even_called_without_drivers():
    """Nothing to ground against means nothing to explain from."""
    called = []
    spy = lambda msgs, **kw: called.append(1) or 'invented reason'
    explain.build_explanation(_reading(drivers=[]), complete=spy)
    assert called == []


def test_an_empty_reply_keeps_the_fallback():
    out = explain.build_explanation(_reading(), complete=lambda m, **k: '   ')
    assert out['why_generated'] is False


def test_a_rambling_reply_is_truncated():
    long = lambda msgs, **kw: ' '.join(['word'] * 100)
    out = explain.build_explanation(_reading(), complete=long)
    assert len(out['why'].split()) <= explain._MAX_WORDS + 1
    assert out['why'].endswith('...')


# ── when ─────────────────────────────────────────────────────────────────────

def test_when_is_scheduled_never_forecast():
    out = explain.build_explanation(_reading(), complete=None)
    assert out['when_basis'] == 'scheduled'
    assert out['when'] in ('today', 'tomorrow') or out['when'].startswith('in ')
    assert '20' in out['when_label']


def test_fuel_and_cpi_sectors_use_different_schedules():
    gas = explain.build_explanation(_reading('gas'), complete=None)
    food = explain.build_explanation(_reading('food', 0.29, '%'), complete=None)
    assert gas['when_date'] != food['when_date']


@pytest.mark.parametrize('sector,unit', [
    ('gas', 'PHP/L'), ('food', '%'), ('electricity', 'PHP/kWh')])
def test_every_sector_produces_a_complete_card(sector, unit):
    out = explain.build_explanation(_reading(sector, 0.2, unit), complete=None)
    for key in ('why', 'when', 'when_label', 'sources', 'grounded'):
        assert out[key] is not None and out[key] != ''


# ── direction wording ────────────────────────────────────────────────────────

@pytest.mark.parametrize('estimate,expected', [
    (0.5, 'going up'), (-0.5, 'going down'), (0.0, 'holding steady')])
def test_direction_matches_the_sign(estimate, expected):
    out = explain.build_explanation(_reading(estimate=estimate), complete=None)
    assert expected in out['why']


def test_build_all_covers_every_reading():
    readings = [_reading('gas'), _reading('food', 0.29, '%'),
                _reading('electricity', -0.11, 'PHP/kWh')]
    out = explain.build_all(readings, complete=None)
    assert [o['sector'] for o in out] == ['gas', 'food', 'electricity']


def test_acronyms_survive_mid_sentence_use():
    """'WESM spot prices eased' must not become 'wESM spot prices eased'."""
    out = explain.build_explanation(
        _reading('electricity', -0.11, 'PHP/kWh',
                 drivers=['WESM spot prices eased through the billing period'],
                 sources=['WESM']),
        complete=None)
    assert 'WESM' in out['why']
    assert 'wESM' not in out['why']


def test_ordinary_words_are_still_lowered_mid_sentence():
    out = explain.build_explanation(
        _reading(drivers=['Brent crude fell about 3 percent']), complete=None)
    assert 'because brent crude' in out['why']


def test_one_other_factor_is_singular():
    out = explain.build_explanation(
        _reading(drivers=['Brent crude fell', 'the peso firmed']), complete=None)
    assert '1 other factor cited' in out['why']
    assert 'factors' not in out['why']

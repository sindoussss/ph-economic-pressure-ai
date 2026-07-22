"""The gas verdict string is consumed by two downstream systems, so its shape
is load-bearing, not cosmetic.

It used to be `str(master_verdict)` — a dataclass repr. The BSP banner regexed
it and matched a nested regional verdict instead of the consensus, and the
causal-chain prompt truncates to 600 characters, so the model received a
mangled Python object and produced numbers unrelated to the run.
"""
import pytest

from ph_economic_ai.ui.main_window import _format_gas_verdict


def test_consensus_is_the_first_peso_figure():
    """Anything scanning left-to-right must hit the consensus first — this is
    exactly what the BSP banner got wrong."""
    text = _format_gas_verdict(
        estimate=2.54,
        agreement_pct=64,
        regional=[(('NCR', 'Central Luzon'), 6.19),
                  (('Western Visayas', 'Davao Region'), 2.54)],
    )
    assert text.index('2.54') < text.index('6.19')


def test_no_python_repr_leaks_into_the_prompt():
    text = _format_gas_verdict(
        estimate=2.54, agreement_pct=64,
        regional=[(('NCR', 'Central Luzon'), 6.19)],
    )
    for token in ('MasterVerdict(', 'RegionalVerdict(', 'final_estimate=', '[', '{'):
        assert token not in text


def test_survives_the_600_character_truncation():
    """CausalChainThread sends only the first 600 chars. The consensus has to
    be inside that window even with every region listed."""
    text = _format_gas_verdict(
        estimate=2.54, agreement_pct=64,
        regional=[((f'Region {i}', f'Region {i + 1}'), 1.0 + i) for i in range(8)],
    )
    assert '2.54' in text[:600]


def test_includes_agreement_and_range_when_known():
    text = _format_gas_verdict(estimate=2.54, agreement_pct=64, low=2.54, high=6.19)
    assert '64%' in text
    assert '+2.54' in text and '+6.19' in text


def test_negative_change_keeps_its_sign():
    assert '-1.20' in _format_gas_verdict(estimate=-1.20)


def test_missing_estimate_is_explicit_not_blank():
    """A blank verdict would silently give the chain model nothing to ground on."""
    assert 'unavailable' in _format_gas_verdict(estimate=None)


def test_regional_entries_without_an_estimate_are_skipped():
    text = _format_gas_verdict(
        estimate=2.54,
        regional=[(('NCR', 'Central Luzon'), None), (('Davao', 'Region XI'), 3.10)],
    )
    assert 'NCR' not in text
    assert '+3.10' in text


@pytest.mark.parametrize('agreement', [0, 53, 100])
def test_agreement_renders_without_decimals(agreement):
    assert f'{agreement}%' in _format_gas_verdict(estimate=1.0, agreement_pct=agreement)

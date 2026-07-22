"""The causal chain must narrate the same CPI the banner computes.

In one run the deterministic banner said 4.10% while the chain's final step said
4.52% — the LLM had computed its own figure. The fix passes the authoritative
projected CPI into the chain prompt.
"""
from ph_economic_ai.engine.live_data import CausalChainThread

SCENARIO = {'oil_pct': 6.8, 'usd_pct': 0.0, 'bsp_rate': 6.5, 'demand_index': 72}


def _thread(projected_cpi=None):
    return CausalChainThread(
        gas_verdict='Retail gasoline monthly change: +0.47 ₱/L',
        food_verdict='Food price index monthly change: +0.48%',
        elec_verdict='Electricity rate monthly change: +0.0330 ₱/kWh',
        scenario=SCENARIO,
        projected_cpi=projected_cpi,
    )


def test_authoritative_cpi_is_injected_into_the_prompt():
    msg = _thread(projected_cpi=4.10)._build_user_msg()
    assert '4.10%' in msg
    assert 'AUTHORITATIVE PROJECTED CPI' in msg
    assert 'MUST NOT compute' in msg


def test_no_cpi_directive_when_none_supplied():
    """Backward compatible: without a CPI the chain behaves as before."""
    msg = _thread(projected_cpi=None)._build_user_msg()
    assert 'AUTHORITATIVE PROJECTED CPI' not in msg


def test_the_sector_verdicts_are_still_present():
    msg = _thread(projected_cpi=4.10)._build_user_msg()
    assert '+0.47' in msg and '+0.48' in msg and '+0.0330' in msg


def test_cpi_is_formatted_to_two_decimals():
    msg = _thread(projected_cpi=4.1)._build_user_msg()
    assert '4.10%' in msg     # not '4.1%'

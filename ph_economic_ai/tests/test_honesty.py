import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from ph_economic_ai import ui  # noqa
from ph_economic_ai.ui import honesty


def test_honesty_constants():
    assert honesty.EXPLORATORY == 'exploratory'
    assert honesty.VALIDATED == 'validated'
    assert 'varies per run' in honesty.AGREEMENT_NOTE
    # the composed consensus note carries both signals
    assert 'exploratory' in honesty.consensus_note()
    assert 'varies per run' in honesty.consensus_note()
    assert 'exploratory' in honesty.interact_caption().lower()

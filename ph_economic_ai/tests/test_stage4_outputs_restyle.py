import sys
import pytest
from PyQt6.QtWidgets import QApplication, QLabel


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication(sys.argv)


def test_final_outputs_and_accuracy_text_survive_the_restyle(app):
    """Pins Task 9's two style substitutions in `_build_right`: the metrics-grid
    value labels and the Validated Accuracy summary lines must keep rendering
    their exact text once the hex literals move to theme tokens.

    The brief's own fixture (`df` with only a `gas_price` column) makes
    `build_features` raise a KeyError *before* `_build_right`'s try/except even
    starts -- `oil_price`, `usd_php`, and `demand_index` are required base
    columns (see `build_gas_features` in ph_economic_ai/utils/preprocessing.py)
    and that call sits above the try/except at lines ~833-841. So the fixture
    here adds those three columns. With `regressor=None`, `ml.forecast` still
    raises (inside that try/except, caught silently) so the chart and the
    3-month/6-month ML figures never populate -- but the metrics grid (all four
    labels, rendered unconditionally per lines 846-863) and the Validated
    Accuracy card (built in its own try/except, independent of the regressor)
    both render normally. Confirmed by manually probing `_build_right` against
    both fixtures before writing this assertion.
    """
    from ph_economic_ai.ui.stage4_report import Stage4ReportPanel
    from ph_economic_ai.ui import honest_surface as _hs
    import numpy as np, pandas as pd

    panel = Stage4ReportPanel()
    n = 30
    df = pd.DataFrame({
        'gas_price': np.linspace(70, 80, n),
        'oil_price': np.linspace(75, 85, n),
        'usd_php': np.linspace(55, 57, n),
        'demand_index': np.linspace(60, 70, n),
    })
    consensus = {'weighted_avg': -0.08}
    panel._build_right(regressor=None, df=df, cv_rmse=1.5,
                       scenario={'current_price': 80.0}, consensus=consensus)
    texts = [c.text() for c in panel.findChildren(QLabel)]

    # Metrics grid: four labels (uppercased by self._muted) + four values.
    for label in ('NEXT WEEK (AI EST.)', 'NEXT MONTH (AI EST.)',
                  '3-MONTH (ML)', '6-MONTH (ML)'):
        assert label in texts, f'missing metrics-grid label: {label!r}'
    assert any('-0.02' in t for t in texts)   # week_est = avg / 4.0
    assert any('-0.08' in t for t in texts)   # avg itself
    # ml_3m/ml_6m are None (regressor=None makes ml.forecast raise, caught
    # silently) so both ML columns fall back to the placeholder.
    assert texts.count('—') >= 2

    # Validated Accuracy card: exact summary lines from the honest_surface
    # helper must survive untouched, computed dynamically (not hardcoded) so
    # this test does not depend on whatever benchmark report happens to be
    # checked in.
    expected_lines = _hs.validated_summary_lines(_hs.load_validated())
    for line in expected_lines:
        assert line in texts, f'missing validated-accuracy line: {line!r}'

"""The manuscripts may not silently disagree with the artifacts.

This is Gate 6 enforcement. The failure it prevents already happened once: the
manuscript header guaranteed that its numbers came verbatim from the artifacts and
"cannot diverge", and when the calendar correction changed every sample size the
guarantee stayed on the page while all twelve numbers went stale.
"""
import json

import pytest

from ph_economic_ai.benchmark import manuscript_check as mc


REPORT = {
    'audit': [{'target': 'fuel', 'verdict': 'predictable', 'n': 72},
              {'target': 'fx', 'verdict': 'efficient', 'n': 72}],
    'nowcast_mom': {'n': 82},
    'mom_longsample': {'n_long': 190, 'mom': {'n': 190}},
}


def test_artifact_sample_sizes_collects_every_count():
    assert mc.artifact_sample_sizes(REPORT) == {72, 82, 190}


def test_a_sample_size_no_artifact_reports_is_a_mismatch():
    text = 'The nowcast was run over n = 61 backtest months.'
    (finding,) = mc.check_sample_sizes(text, REPORT)
    assert finding['severity'] == 'mismatch'
    assert finding['claimed'] == 61
    assert finding['line'] == 1


def test_a_sample_size_the_artifacts_report_is_accepted():
    assert mc.check_sample_sizes('Evaluated over n = 82 months.', REPORT) == []


def test_design_constants_are_not_flagged():
    """min_train is a design parameter, not a measured sample."""
    assert mc.check_sample_sizes('Training starts at n = 24 observations.', REPORT) == []


def test_simulation_cells_are_flagged_for_review_not_as_mismatches():
    """The size study chooses its own n. That is an author decision, and only needs
    revisiting when the empirical n it was anchored to moves."""
    text = 'On simulated targets at n = 151 the protocol rejects on 99.7% of replications.'
    (finding,) = mc.check_sample_sizes(text, REPORT)
    assert finding['severity'] == 'review'


def test_a_contradicted_verdict_is_caught():
    text = 'One-month-ahead forecasts of fuel are informationally efficient.'
    (finding,) = mc.check_verdicts(text, REPORT)
    assert finding['kind'] == 'verdict'
    assert finding['claimed'] == 'efficient'
    assert 'predictable' in finding['detail']


def test_a_verdict_that_agrees_is_not_flagged():
    assert mc.check_verdicts('The fuel target is predictable.', REPORT) == []


def test_run_reports_the_marker(tmp_path):
    report = tmp_path / 'r.json'
    report.write_text(json.dumps(REPORT), encoding='utf-8')

    silent = tmp_path / 'silent.md'
    silent.write_text('Results over n = 61 months.', encoding='utf-8')
    result = mc.run(manuscripts=(silent,), report_path=report)
    assert result['undeclared_divergence'] == ['silent.md']
    assert not result['consistent']

    honest = tmp_path / 'honest.md'
    honest.write_text(f'{mc.STALE_MARKER}: stale.\n\nResults over n = 61 months.',
                      encoding='utf-8')
    result = mc.run(manuscripts=(honest,), report_path=report)
    assert result['undeclared_divergence'] == []


def test_no_shipped_manuscript_diverges_without_saying_so():
    """The invariant. A manuscript may be ahead of or behind the artifacts -- drafts
    legitimately are -- but it may not claim agreement it does not have.

    If this fails, either correct the manuscript numbers or add the divergence
    notice. Do not delete the assertion.
    """
    result = mc.run()
    assert result['undeclared_divergence'] == [], (
        'these manuscripts disagree with the committed artifacts and do not declare '
        f'it: {result["undeclared_divergence"]}. Run '
        'python -m ph_economic_ai.benchmark.manuscript_check for the details.')


def test_the_checker_actually_has_something_to_check():
    """Guards against the checker silently passing because it stopped finding the
    manuscripts or the report."""
    result = mc.run()
    assert result['manuscripts'], 'no manuscripts were located'
    assert result['artifact_sample_sizes'], 'no sample sizes were read from artifacts'

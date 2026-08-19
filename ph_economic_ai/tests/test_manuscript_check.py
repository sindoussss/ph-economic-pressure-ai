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


def test_blockquote_lines_are_not_scanned_for_sample_sizes():
    """The ARTIFACT-DIVERGENCE notice documents a past correction by naming the
    old number -- 'n = 51/52 -> 72' -- inside a blockquote. That historical
    mention is not a live claim to re-check."""
    text = '> Reconciled: n = 51/52 became n = 72.'
    assert mc.check_sample_sizes(text, REPORT) == []


def test_a_sample_size_a_different_artifact_reports_is_accepted():
    """sentiment_nowcast.json reports food's n = 102 while accuracy_report.json
    never mentions it. Pooling across artifacts must not flag a number just
    because one file happens to omit it."""
    other_report = {'targets': {'food': {'n': 102}}}
    pool = [REPORT, other_report]
    assert mc.check_sample_sizes('Food MoM (n = 102).', pool) == []


def test_all_committed_artifacts_pools_every_json_file(tmp_path):
    (tmp_path / 'a.json').write_text(json.dumps({'n': 42}), encoding='utf-8')
    (tmp_path / 'b.json').write_text(json.dumps({'thing': {'n': 99}}), encoding='utf-8')
    (tmp_path / 'not_json.txt').write_text('n = 1', encoding='utf-8')
    reports = mc.all_committed_artifacts(artifacts_dir=tmp_path)
    assert mc.artifact_sample_sizes(reports) == {42, 99}


def test_all_committed_artifacts_excludes_the_given_path(tmp_path):
    excluded = tmp_path / 'accuracy_report.json'
    excluded.write_text(json.dumps({'n': 1}), encoding='utf-8')
    (tmp_path / 'other.json').write_text(json.dumps({'n': 2}), encoding='utf-8')
    reports = mc.all_committed_artifacts(exclude=excluded, artifacts_dir=tmp_path)
    assert mc.artifact_sample_sizes(reports) == {2}


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


def test_readme_is_in_scope():
    """README.md makes the same kind of claims (verdicts, sample sizes) to a far
    larger audience than either manuscript, and was not checked until 2026-08-07:
    its own "fuel: efficient" row was never caught here, only in the manuscript,
    because README.md was simply never in MANUSCRIPTS. Pinned so it cannot be
    dropped from scope silently."""
    assert any(p.name == 'README.md' for p in mc.MANUSCRIPTS)
    assert (mc.REPO_ROOT / 'README.md').exists()


def test_talking_points_is_in_scope():
    """talking-points.md carries the identical undisclosed "fuel: efficient" row
    and untraceable skill number README's had before RSK-017, found under RSK-020
    the same way -- neither the manuscript's fix nor README's reached this sibling.
    Pinned so it cannot be dropped from scope silently."""
    assert any(p.name == 'talking-points.md' for p in mc.MANUSCRIPTS)
    assert (mc.DOCS / 'defense' / 'talking-points.md').exists()


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


# ── The checker must survive the console it is run on ────────────────────────
#
# Found 2026-08-20. Running `python -m ph_economic_ai.benchmark.manuscript_check`
# on the maintainer machine printed nine findings and then died:
#
#     UnicodeEncodeError: 'charmap' codec can't encode characters in position
#     84-85: character maps to <undefined>
#
# Windows gives an interactive console the cp1252 codepage. The manuscripts are
# a statistics thesis and contain 155 U+2212 MINUS SIGN, 149 Greek rho, 31 PESO
# SIGN and a tail of arrows and inequalities, none of which cp1252 can encode.
# The tool crashed while printing the CONTEXT of a finding, never while
# analysing anything.
#
# It is not the em dash, which cp1252 encodes fine at 0x97. That guess was made
# first and was wrong, so the characters are named here explicitly.
#
# This is Gate 6 enforcement. A gate that exits on a traceback instead of a
# verdict cannot report the gate, and a CI or scheduled invocation would see
# only the crash.

import io
import sys

from ph_economic_ai.benchmark import manuscript_check as _mc

#: Real characters from the manuscripts, chosen because cp1252 has no mapping.
UNENCODABLE = 'the anchor is ₱1.40/L, ρ ≈ 0.95, bias − 0.3'


def _cp1252_console():
    """A stand-in for the Windows console that caused the crash."""
    return io.TextIOWrapper(io.BytesIO(), encoding='cp1252', errors='strict')


def test_the_console_really_cannot_encode_these(): 
    """The precondition, asserted so the tests below cannot pass vacuously."""
    console = _cp1252_console()
    with pytest.raises(UnicodeEncodeError):
        console.write(UNENCODABLE)
        console.flush()


def test_console_safe_survives_characters_the_console_lacks():
    out = _mc.console_safe(UNENCODABLE, _cp1252_console())
    console = _cp1252_console()
    console.write(out)
    console.flush()                       # must not raise
    assert 'anchor' in out and '0.95' in out, 'the readable part must survive'


def test_console_safe_leaves_text_alone_when_the_console_can_take_it():
    utf8 = io.TextIOWrapper(io.BytesIO(), encoding='utf-8')
    assert _mc.console_safe(UNENCODABLE, utf8) == UNENCODABLE


def test_main_completes_on_a_cp1252_console(monkeypatch):
    """The regression. It printed nine findings, then raised."""
    console = _cp1252_console()
    monkeypatch.setattr(sys, 'stdout', console)
    try:
        rc = _mc.main()
        console.flush()
    finally:
        monkeypatch.undo()

    printed = console.buffer.getvalue().decode('cp1252')
    assert isinstance(rc, int)
    assert 'mismatches' in printed, 'the report must actually reach the console'


def test_sanitising_the_display_does_not_touch_the_findings():
    """Only printing is made safe. The findings keep the real text, or a consumer
    reading them would get mangled evidence."""
    result = _mc.run()
    contexts = [f['context']
                for entry in result['manuscripts'] for f in entry['findings']]
    assert contexts, 'expected findings to exist to make this meaningful'
    assert any(any(ord(c) > 255 for c in ctx) for ctx in contexts), (
        'the findings should still carry the characters that broke printing')

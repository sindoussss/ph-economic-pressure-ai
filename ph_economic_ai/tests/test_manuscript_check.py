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
    """Only printing is made safe. A finding keeps the real text, or a consumer
    reading it as data would get evidence with holes punched in it.

    Built from a constructed claim rather than the live manuscripts. Those
    reported findings when this test was written and report none now, which is
    the gate working, but it leaves no natural specimen to assert against.
    """
    text = ('The nowcast used n = 61111 months at rho ≈ 0.95, '
            'anchored at ₱1.40/L, bias − 0.3.')
    (finding,) = _mc.check_sample_sizes(text, REPORT)
    assert '₱' in finding['context'] and '≈' in finding['context'], (
        'the stored finding must carry the original characters')
    safe = _mc.console_safe(finding['context'], _cp1252_console())
    assert '₱' not in safe and '≈' not in safe, (
        'the printable form must not carry what the console cannot encode')
    _cp1252_console().write(safe)                       # must not raise


# ── A correct number reported as wrong is worse than no checker ──────────────
#
# Found 2026-08-20. `talking-points.md` says, naming its source:
#
#     **"Why 20 agents?" -- the ablation** (`swarm_ablation.json`, n=8)
#
# and the checker reported "no artifact reports n = 8". The artifact does report
# it, as `repeats: 8`. The pool already spans every committed artifact; it
# collected only keys literally named n, n_long, n_calib and n_eval, and this one
# is a run count under a different name.
#
# The cost of leaving it is not the one wrong line. This module's whole value is
# that its output can be trusted without re-deriving it, and a reader who checks
# the first finding, discovers the number is fine, and concludes the tool cries
# wolf will not check the twenty-fifth. That is how a gate stops being read.
#
# The fix must not overshoot in the other direction. Widening the pool until any
# integer resolves would silence real findings, so `reps` (300 simulation
# replications) is deliberately left out: no manuscript cites `n = 300`, and an
# unnecessary value in the pool is a false negative waiting to happen.

def test_run_counts_are_recognised_as_sample_sizes():
    sizes = _mc.artifact_sample_sizes(
        {'repeats': 8, 'inner': {'n_runs': 12}, 'unrelated': {'reps': 300}})
    assert 8 in sizes and 12 in sizes
    assert 300 not in sizes, 'reps is simulation replications, not a sample count'


def test_the_ablation_run_count_is_no_longer_a_false_positive():
    """The regression, against the real committed artifacts."""
    result = _mc.run()
    flagged = [f for entry in result['manuscripts'] for f in entry['findings']
               if 'n = 8' in f.get('detail', '')]
    assert not flagged, (
        'swarm_ablation.json reports repeats: 8, so n = 8 is a correct claim')


def test_widening_the_pool_did_not_blind_the_checker():
    """The other direction. A number no artifact reports must still be caught."""
    report = {'audit': [{'target': 'fuel', 'verdict': 'efficient', 'n': 72}]}
    text = 'The nowcast was run over n = 61111 backtest months.'
    (finding,) = _mc.check_sample_sizes(text, report)
    assert finding['severity'] == 'mismatch'


def test_the_pool_grew_by_exactly_the_run_counts():
    """Pins the size of the concession.

    Recomputing the pool without the two new keys must differ by exactly the run
    counts, so a later edit cannot quietly widen it into uselessness.
    """
    pool = _mc.all_committed_artifacts(exclude=None)
    wide = _mc.artifact_sample_sizes(pool)
    narrow = _mc.artifact_sample_sizes(pool, keys=('n', 'n_long', 'n_calib', 'n_eval'))
    assert wide - narrow == {8}, f'unexpected widening: {sorted(wide - narrow)}'


# ── A paragraph's vocabulary is not its assertions ───────────────────────────
#
# Found 2026-08-20. `check_verdicts` asked whether a target name and the opposite
# verdict word appear anywhere on the same LINE. These manuscripts are written in
# paragraph-per-line markdown, so a single "line" can run 1200 characters and
# mention four series. Every remaining Gate 7 finding was a false positive:
#
#   thesis 570   "predictable" describes FOOD's own dynamics; the line also
#                contains "Fuel and electricity receive a ... anchor"
#   thesis 551   "MoM is predictable" is quoted as an overclaim the design
#                PREVENTED
#   talking 174  "MoM inflation / electricity is predictable" is quoted and
#                marked withdrawn, on a line that ends "both are nulls"
#
# Two rules fix all three without touching a word of correct prose. A verdict
# word is attributed to the NEAREST series named, because "food ... predictable"
# is a claim about food however many other series share the paragraph. And a
# claim inside quotation marks is being discussed, not asserted -- a manuscript
# that quotes an overclaim in order to disown it must not be recorded as making
# it.
#
# The risk is silencing the tool, so both directions are pinned below: the same
# sentence unquoted is still caught, and a genuine contradiction still fails.

VERDICTS = {'audit': [{'target': 'fuel', 'verdict': 'efficient', 'n': 72},
                      {'target': 'inflation', 'verdict': 'efficient', 'n': 80}]}


def test_a_verdict_is_attributed_to_the_nearest_series():
    """thesis line 570, reduced to its shape."""
    text = ('Fuel and electricity receive a mechanical fuel pass-through anchor; '
            'food, which the audit found a clean null on commodity drivers but '
            'predictable from own dynamics, is anchored to the trailing trend.')
    assert _mc.check_verdicts(text, VERDICTS) == []


def test_the_nearest_series_rule_still_catches_a_real_claim():
    """The same shape with the words the other way round must still fail."""
    text = ('Food receives a trailing-trend anchor; fuel, on the evidence of the '
            'audit, is predictable from commodity drivers.')
    (finding,) = _mc.check_verdicts(text, VERDICTS)
    assert finding['detail'] == 'artifacts report fuel as efficient'


def test_a_quoted_claim_is_not_an_asserted_claim():
    """thesis 551 and talking-points 174. Both quote a claim to disown it."""
    disowned = ('the driver-only ablation (which stopped "MoM is predictable" '
                'from silently becoming "the drivers predict inflation")')
    withdrawn = '- "MoM inflation / electricity is predictable" -- withdrawn; both are nulls'
    assert _mc.check_verdicts(disowned, VERDICTS) == []
    assert _mc.check_verdicts(withdrawn, VERDICTS) == []


def test_the_same_sentence_unquoted_is_still_caught():
    """The quote rule must not become a way to smuggle a claim past the gate."""
    asserted = 'MoM inflation is predictable from the drivers.'
    (finding,) = _mc.check_verdicts(asserted, VERDICTS)
    assert finding['detail'] == 'artifacts report inflation as efficient'


def test_a_line_already_stating_the_right_verdict_is_left_alone():
    text = 'Fuel is efficient at one month, though earlier drafts called it predictable.'
    assert _mc.check_verdicts(text, VERDICTS) == []


def test_the_real_documents_have_no_verdict_findings_left():
    """The regression, against the committed manuscripts."""
    result = _mc.run()
    verdicts = [f for e in result['manuscripts'] for f in e['findings']
                if f.get('kind') == 'verdict']
    assert verdicts == [], f'unexpected verdict findings: {verdicts}'


# ── The second vocabulary: nowcast verdicts ──────────────────────────────────
#
# `RSK-059`, opened 2026-08-20 and fixed here. `check_verdicts` reads only the
# audit's `efficient`/`predictable` pair, scoped to `accuracy_report.json`'s
# audit panel. The nowcast pair, `beats_best_naive`/`no_better_than_naive`, was
# never compared against anything, so no document's nowcast claim was checked at
# all -- which is how three design specs published four withdrawn positives for
# seven weeks while this tool reported `consistent` (`RSK-057`).
#
# The narrowing that makes this safe is the same one PR #64's banner guard used,
# and for the same reason. In these documents a prose mention of the token is
# almost always meta rather than asserted:
#
#   thesis 219   defines the test family as the nodes "returning a
#                `beats_best_naive` verdict"
#   thesis 419   "there are no `beats_best_naive` positives left to correct"
#   thesis 691   "Every `beats_best_naive` verdict in the earlier draft ...
#                none survives the mean column"
#
# All three are correct writing and none is a claim. A results TABLE row is a
# claim, so rows are what this reads. Appendix B's own convention supplies the
# second rule: a row labelled `*Superseded (vs random walk)*` is preserving
# history on purpose and is left alone.
#
# Attribution has to look past the row, because a row like
# "| Driver-only, full sample (n = 204) | beats_best_naive | ... |" names no
# series at all. It inherits the nearest series named above it, which is the
# section it sits in.

NOWCASTS = {'transport_nowcast': {'n': 204,
                                  'mom': {'verdict': 'no_better_than_naive'},
                                  'driver_ablation': {'verdict': 'no_better_than_naive'},
                                  'robust': {'driver_ablation':
                                             {'verdict': 'no_better_than_naive'}}}}


def test_nowcast_verdicts_are_collected_per_series():
    assert _mc.nowcast_verdicts(NOWCASTS) == {'transport': {'no_better_than_naive'}}


def test_a_table_row_claiming_a_withdrawn_nowcast_positive_is_caught():
    """thesis 345, reduced to its shape: the defect this check exists for."""
    text = ('A robustness re-test dissolved it. Transport CPI was anomalous.\n'
            '\n'
            '| Test | Verdict | skill vs best naive | DM p |\n'
            '|---|---|---|---|\n'
            '| Driver-only, full sample (n = 204) | beats_best_naive | +14.8% | 0.021 |\n')
    (finding,) = _mc.check_nowcast_verdicts(text, NOWCASTS)
    assert finding['kind'] == 'nowcast-verdict'
    assert finding['severity'] == 'mismatch'
    assert finding['claimed'] == 'beats_best_naive'
    assert 'transport' in finding['detail']


def test_a_row_marked_superseded_is_left_alone():
    """Appendix B.5's convention: the old value, preserved and labelled."""
    text = ('Transport panel.\n'
            '| **Verdict (corrected)** | no_better_than_naive |\n'
            '| *Superseded (vs random walk)* | *beats_best_naive +14.8%, p = 0.021* |\n')
    assert _mc.check_nowcast_verdicts(text, NOWCASTS) == []


def test_prose_naming_the_token_is_not_a_claim():
    """thesis 219, 419 and 691. Definitions and negations, all correct writing."""
    for line in (
        'The family is every node returning a `beats_best_naive` verdict for transport.',
        'Under the corrected pool there are no `beats_best_naive` positives left.',
        'Every `beats_best_naive` verdict in the earlier transport draft has been withdrawn.',
    ):
        assert _mc.check_nowcast_verdicts(line, NOWCASTS) == [], line


def test_a_row_that_also_states_the_reported_verdict_is_left_alone():
    text = ('Transport panel.\n'
            '| Driver-only | no_better_than_naive, was beats_best_naive | 0.0 |\n')
    assert _mc.check_nowcast_verdicts(text, NOWCASTS) == []


def test_a_row_inside_a_code_fence_is_not_a_claim():
    text = ('Transport panel.\n'
            '```\n'
            '| Driver-only | beats_best_naive | +14.8% |\n'
            '```\n')
    assert _mc.check_nowcast_verdicts(text, NOWCASTS) == []


def test_a_row_about_a_series_the_artifacts_never_nowcast_is_left_alone():
    """Attribution failure must mean silence, not a guess."""
    text = ('The swarm ablation.\n'
            '| Roster | beats_best_naive | +2.0% |\n')
    assert _mc.check_nowcast_verdicts(text, NOWCASTS) == []


def test_the_real_documents_have_no_nowcast_verdict_findings_left():
    """The regression. Fails until thesis 345 states the corrected verdict."""
    result = _mc.run()
    found = [f for e in result['manuscripts'] for f in e['findings']
             if f.get('kind') == 'nowcast-verdict']
    assert found == [], f'unexpected nowcast verdict findings: {found}'


def test_the_specs_are_uncovered_by_choice_not_by_blindness():
    """`RSK-057`'s three specs sit outside `MANUSCRIPTS` deliberately.

    The whole `RSK-017`/`RSK-020`/`RSK-014`/`RSK-057` sequence is documents that
    were never in that tuple, so "absent" has to mean a decision that someone
    recorded rather than a gap nobody noticed. The decision is that a dated design
    note owes a reader disclosure, not currency, and
    `test_withdrawn_findings_are_marked.py` enforces that instead.

    What this pins is the other half: the check can see them perfectly well.
    """
    spec = (_mc.DOCS / 'superpowers' / 'specs' /
            '2026-06-10-electricity-cpi-nowcast-design.md')
    assert spec.resolve() not in {p.resolve() for p in _mc.MANUSCRIPTS}

    text = spec.read_text(encoding='utf-8')
    if 'beats_best_naive' not in text:
        pytest.skip('the electricity spec no longer tables a withdrawn verdict')

    report = json.loads(_mc.ACCURACY_REPORT.read_text(encoding='utf-8'))
    findings = _mc.check_nowcast_verdicts(text, report)
    assert findings, 'the check must see the documents it deliberately does not police'
    assert all(f['detail'].startswith('artifacts report electricity') for f in findings)

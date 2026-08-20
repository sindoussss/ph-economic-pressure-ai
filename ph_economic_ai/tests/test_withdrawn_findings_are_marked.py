"""A document asserting a withdrawn positive must say that it is superseded.

Found 2026-08-20, sweeping the documents `manuscript_check.py` does not cover.
Three design specs under `docs/superpowers/specs/` still published the four
withdrawn nowcast positives as measured results, with no banner of any kind:

    electricity-cpi-nowcast-design.md   S9 "The contribution -- measured result
                                        (a robust positive)", three rows
    food-cpi-nowcast-design.md          full nowcast, ARIMA, +16.0%
    transport-cpi-nowcast-design.md     driver-only ablation, ridge, +14.8%

Every one of those verdicts is `no_better_than_naive` in the committed artifacts
and has been since the baseline pool gained the historical mean. The electricity
spec also argues the mechanism -- "the Meralco generation charge is a formulaic,
near-deterministic pass-through" -- which is the overclaim `RSK-008` exists to
prevent, presented there as the explanation for an edge that no longer exists.

This is the fourth time the same shape has been found. `RSK-017` fixed it in
`README.md`, `RSK-020` in `talking-points.md`, `RSK-014` in
`reviewer-critique.md`, each time because that document was simply not in the
list of things anything checked. Bannering three more files fixes today's
instance and does nothing about the fifth, so the rule is enforced here instead.

**Why table rows rather than any mention.** `manuscript_check.check_verdicts`
reads only the audit vocabulary (`efficient` / `predictable`) and never the
nowcast vocabulary, so it cannot see these claims at all; and a rule that fired
on every occurrence of the token would flag
`docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md`, which
names it in prose to explain what the corrected pool changed. Naming a verdict
is not claiming it. Putting one in a results table is, so that is what this
checks -- the narrowest form that still catches all three defects, per the
"unnecessary checks are false positives waiting to happen" note in
`manuscript_check.SIZE_KEYS`.

Code fences are stripped first: `plans/2026-06-10-transport-cpi-nowcast.md`
builds `{'verdict': 'beats_best_naive'}` inside test fixtures, which is source,
not a claim about the world.
"""
import pathlib
import re

DOCS = pathlib.Path(__file__).resolve().parents[2] / 'docs'

#: Verdicts that no target in the repository currently returns. A results table
#: asserting one is describing a superseded run whatever else it says.
WITHDRAWN_VERDICTS = ('beats_best_naive',)

#: Any one of these anywhere in the file discharges the requirement. The point is
#: that a reader meets the disclosure before they act on the number, not that it
#: is worded a particular way. `ARTIFACT-DIVERGENCE` is
#: `manuscript_check.STALE_MARKER`, kept in step deliberately.
DISCLOSURE_MARKERS = ('ARTIFACT-DIVERGENCE', 'superseded', 'Superseded', 'SUPERSEDED')

_FENCE = re.compile(r'```.*?```', re.DOTALL)


def _asserting_rows(text: str) -> list[str]:
    """Markdown table rows outside code fences that state a withdrawn verdict."""
    return [line.strip()
            for line in _FENCE.sub('', text).splitlines()
            if line.lstrip().startswith('|')
            and any(v in line for v in WITHDRAWN_VERDICTS)]


def test_no_document_tables_a_withdrawn_verdict_without_disclosing_it():
    offenders = []
    for path in sorted(DOCS.rglob('*.md')):
        text = path.read_text(encoding='utf-8')
        rows = _asserting_rows(text)
        if rows and not any(m in text for m in DISCLOSURE_MARKERS):
            offenders.append(f'{path.relative_to(DOCS)}: {len(rows)} row(s), '
                             f'first is {rows[0][:90]!r}')

    assert not offenders, (
        'These documents present a withdrawn verdict as a measured result and '
        'carry no supersession notice:\n  ' + '\n  '.join(offenders))


def test_the_rule_does_not_fire_on_naming_a_verdict_in_prose():
    """The guard must stay narrow enough that it is worth reading its output.

    A document that explains what `beats_best_naive` means, or reports that
    nothing returns it any more, is doing the opposite of overclaiming.
    """
    prose = ('Under the old pool the verdict was `beats_best_naive`; under the '
             'corrected pool no target returns it.\n')
    assert _asserting_rows(prose) == []


def test_the_rule_ignores_fixtures_inside_code_fences():
    fenced = ("```python\n"
              "expected = {'verdict': 'beats_best_naive'}\n"
              "```\n")
    assert _asserting_rows(fenced) == []


def test_the_rule_catches_a_results_row():
    row = '| Driver-only ablation | beats_best_naive | ridge | +14.8% | 0.021 |\n'
    assert _asserting_rows(row) == [row.strip()]

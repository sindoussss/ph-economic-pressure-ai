"""Cross-check the manuscripts against the committed artifacts.

Gate 6 of `06 Work/Publication Remediation` asks for "an automated cross-check
between manuscript tables and committed artifacts". This is it.

The risk it removes is specific and it has already happened once. The manuscript
states that every empirical value is taken verbatim from the frozen artifacts and
that the two documents "cannot diverge". Nothing enforced that. When the calendar
correction changed every sample size on 2026-07-28, all twelve sample sizes in the
manuscript silently became wrong while the guarantee stayed on the page.

This module makes divergence detectable instead of a matter of someone
remembering. It does not rewrite prose and it does not decide what the manuscript
should say -- it reports what the manuscript asserts that the artifacts do not
support.

Deliberately conservative: it only flags claims it can resolve to an artifact
value. A number it cannot interpret is left alone rather than guessed at, because
a checker that cries wolf gets switched off.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Optional
from pathlib import Path

from ph_economic_ai.benchmark.paths import ACCURACY_REPORT, BENCHMARK_DIR

DOCS = BENCHMARK_DIR.parent.parent / 'docs'
REPO_ROOT = DOCS.parent
MANUSCRIPTS = (
    DOCS / 'manuscript' / '2026-06-10-thesis-manuscript.md',
    DOCS / 'manuscript' / '2026-07-26-baseline-specification-note.md',
    # README.md makes the same kind of claims (verdicts, sample sizes) to a far
    # larger audience than either manuscript, and had never been checked: its own
    # "fuel: efficient" row was never caught by this tool, unlike the identical
    # claim in the manuscript, because README.md was simply never in this tuple.
    REPO_ROOT / 'README.md',
    # Same defect, third sibling: talking-points.md carries its own verdict table
    # with the identical undisclosed "fuel: efficient" row and an untraceable
    # skill number, found under RSK-020 the same way README's was found under
    # RSK-017 -- neither the manuscript's fix nor README's reached it.
    DOCS / 'defense' / 'talking-points.md',
)

# `n = 24` is min_train, a design parameter, not a sample size. Simulation cells
# in the size study are also design choices rather than measured samples; they are
# declared here so the checker does not flag an author's deliberate choice.
DESIGN_CONSTANTS = {24}

# Marker a manuscript carries while it is knowingly ahead of or behind the
# artifacts. Its presence is what makes a stale draft honest rather than wrong.
STALE_MARKER = 'ARTIFACT-DIVERGENCE'

_N_PATTERN = re.compile(r'n\s*=\s*(\d+)')
_SIM_CONTEXT = re.compile(r'simulat|replicat|power|size study|pool without|either n',
                          re.IGNORECASE)


#: Keys whose integer value is a count a manuscript may legitimately cite as `n`.
#:
#: `repeats` and `n_runs` were added 2026-08-20. `talking-points.md` cites the
#: swarm ablation as `n=8` and names its source in the same breath, yet the
#: checker reported "no artifact reports n = 8" because `swarm_ablation.json`
#: stores that count as `repeats`. A correct claim reported as wrong is worse
#: than a missed one: a reader who checks the first finding, sees the number is
#: fine and concludes the tool cries wolf will not check the twenty-fifth.
#:
#: `reps` is deliberately absent. It is the 300 replications per simulation cell,
#: no manuscript cites `n = 300`, and every unnecessary value in this pool is a
#: false negative waiting to happen. The pool is a concession to be kept small,
#: which `test_the_pool_grew_by_exactly_the_run_counts` enforces.
SIZE_KEYS = ('n', 'n_long', 'n_calib', 'n_eval', 'repeats', 'n_runs')


def artifact_sample_sizes(report: dict, keys=SIZE_KEYS) -> set[int]:
    """Every integer the artifacts describe as a sample or evaluation count."""
    found: set[int] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in keys and isinstance(value, int):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(report)
    return found


def scan_sample_sizes(text: str) -> list[dict]:
    """Every `n = <int>` a manuscript asserts, with its line and context.

    A line inside a markdown blockquote (leading `>`) is skipped. This
    manuscript uses blockquotes only for the ARTIFACT-DIVERGENCE notice, which
    documents a past correction by naming the old number -- "n = 51/52 -> 72"
    -- and that historical mention is not a live claim to check.
    """
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith('>'):
            continue
        for match in _N_PATTERN.finditer(line):
            out.append({'line': lineno, 'n': int(match.group(1)),
                        'context': line.strip()[:160]})
    return out


def check_sample_sizes(text: str, report: dict) -> list[dict]:
    """Sample sizes asserted in prose that no artifact reports.

    Lines that read as simulation design (the size and power study chooses its own
    n) are reported at lower severity, because those are author choices rather
    than measurements and only need review when the empirical n they were anchored
    to moves.
    """
    available = artifact_sample_sizes(report)
    findings = []
    for hit in scan_sample_sizes(text):
        if hit['n'] in DESIGN_CONSTANTS or hit['n'] in available:
            continue
        simulated = bool(_SIM_CONTEXT.search(hit['context']))
        findings.append({
            'kind': 'sample-size',
            'severity': 'review' if simulated else 'mismatch',
            'line': hit['line'],
            'claimed': hit['n'],
            'context': hit['context'],
            'detail': ('simulation cell, anchored to an empirical n that has moved'
                       if simulated else
                       f'no artifact reports n = {hit["n"]}'),
        })
    return findings


#: Series named in prose. Used to attribute a verdict word to the nearest one,
#: so a paragraph mentioning four series does not register four claims.
SERIES_NAMES = ('electricity', 'transport', 'inflation', 'food', 'fuel', 'fx')

_QUOTE_PATTERNS = (r'"[^"]*"', '“[^”]*”', "'[^']*'")


def _quoted_spans(line: str) -> list:
    """Character ranges of every quoted run, straight or curly."""
    spans = []
    for pattern in _QUOTE_PATTERNS:
        spans.extend((m.start(), m.end()) for m in re.finditer(pattern, line))
    return spans


def _nearest_series(low: str, pos: int, names) -> Optional[str]:
    """The series named closest to `pos`, or None if none is named.

    "food ... predictable" is a claim about food no matter how many other series
    share the paragraph, so proximity decides attribution.
    """
    best, best_distance = None, None
    for name in names:
        for m in re.finditer(rf'\b{re.escape(name)}\b', low):
            distance = min(abs(m.start() - pos), abs(m.end() - pos))
            if best_distance is None or distance < best_distance:
                best, best_distance = name, distance
    return best


def check_verdicts(text: str, report: dict) -> list[dict]:
    """Audit verdicts asserted in prose that disagree with the artifacts.

    The rule was once "the target name and the opposite verdict both appear on
    this line". These manuscripts are paragraph-per-line markdown, so a line can
    run twelve hundred characters and name four series, and on 2026-08-20 every
    remaining finding it produced was a false positive: a sentence about FOOD
    being predictable from its own dynamics counted as a claim about fuel, and
    two passages quoting an overclaim in order to disown it counted as making it.

    Two narrower rules replace it. A verdict word is attributed to the nearest
    series named, and a claim inside quotation marks is treated as discussed
    rather than asserted. Both directions are tested: the same sentence unquoted
    is still caught, and moving the series name nearer still fails the check.

    A false positive is not a harmless surplus here. This gate is only worth
    reading if its output can be trusted without re-deriving it, and a reader who
    finds the first finding spurious stops reading the rest.
    """
    targets = {row['target']: row['verdict'] for row in report.get('audit', [])
               if row.get('target') and row.get('verdict')}
    names = tuple(dict.fromkeys(tuple(targets) + SERIES_NAMES))

    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        spans = _quoted_spans(line)
        for target, verdict in targets.items():
            if verdict in low:            # the line already states the right one
                continue
            opposite = 'efficient' if verdict == 'predictable' else 'predictable'
            for m in re.finditer(opposite, low):
                if any(start <= m.start() < end for start, end in spans):
                    continue
                if _nearest_series(low, m.start(), names) != target:
                    continue
                findings.append({
                    'kind': 'verdict',
                    'severity': 'mismatch',
                    'line': lineno,
                    'claimed': opposite,
                    'context': line.strip()[:160],
                    'detail': f'artifacts report {target} as {verdict}',
                })
                break
    return findings


def all_committed_artifacts(exclude: Path = None, artifacts_dir: Path = None) -> list[dict]:
    """Every committed artifact JSON, parsed, for pooling sample sizes across
    all of them rather than reading `accuracy_report.json` alone. A manuscript
    number is flagged only if no artifact anywhere reports it -- narrower
    coverage here only produces false positives, such as `sentiment_nowcast.json`
    reporting food's n = 102 while `accuracy_report.json` never mentions it.
    """
    directory = artifacts_dir or ACCURACY_REPORT.parent
    out = []
    for path in sorted(directory.glob('*.json')):
        if exclude is not None and path == exclude:
            continue
        try:
            out.append(json.loads(path.read_text(encoding='utf-8')))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def check_manuscript(path: Path, verdict_report: dict, size_pool: list) -> dict:
    text = path.read_text(encoding='utf-8')
    findings = check_sample_sizes(text, size_pool) + check_verdicts(text, verdict_report)
    mismatches = [f for f in findings if f['severity'] == 'mismatch']
    return {
        'manuscript': path.name,
        'exists': True,
        'declares_divergence': STALE_MARKER in text,
        'findings': findings,
        'n_mismatches': len(mismatches),
        'n_review': len(findings) - len(mismatches),
    }


def run(manuscripts=MANUSCRIPTS, report_path: Path = None) -> dict:
    accuracy_path = Path(report_path or ACCURACY_REPORT)
    verdict_report = json.loads(accuracy_path.read_text(encoding='utf-8'))
    # Verdicts (efficient/predictable) are specific to the audit panel in
    # accuracy_report.json and stay scoped to it. Sample sizes are pooled
    # across every committed artifact so a number correct in one file is not
    # flagged for being absent from another.
    if report_path is None:
        size_pool = [verdict_report] + all_committed_artifacts(exclude=accuracy_path)
    else:
        size_pool = [verdict_report]
    results = [check_manuscript(p, verdict_report, size_pool) for p in manuscripts if p.exists()]
    total = sum(r['n_mismatches'] for r in results)
    undeclared = [r['manuscript'] for r in results
                  if r['n_mismatches'] and not r['declares_divergence']]
    return {
        'artifact_sample_sizes': sorted(artifact_sample_sizes(size_pool)),
        'manuscripts': results,
        'total_mismatches': total,
        'undeclared_divergence': undeclared,
        'consistent': total == 0,
    }


def console_safe(text, stream=None) -> str:
    """`text` rendered so it can be printed whatever codepage `stream` uses.

    Windows gives an interactive console cp1252, and the manuscripts are a
    statistics thesis: 155 U+2212 MINUS SIGN, 149 Greek rho, 31 PESO SIGN, plus
    arrows and inequalities, none of which cp1252 can encode. Printing a finding's
    context therefore killed the run partway through the report on 2026-08-20.

    Note it is NOT the em dash, which cp1252 encodes at 0x97. That was the first
    guess and it was wrong.

    Only the DISPLAY is sanitised. The findings keep their original text, because
    a consumer reading them as data must not receive evidence with holes punched
    in it.
    """
    stream = sys.stdout if stream is None else stream
    encoding = getattr(stream, 'encoding', None) or 'utf-8'
    text = str(text)
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, errors='replace').decode(encoding, errors='replace')
    return text


def main() -> int:
    result = run()
    print(f'artifact sample sizes: {result["artifact_sample_sizes"]}\n')
    for entry in result['manuscripts']:
        flag = 'declares divergence' if entry['declares_divergence'] else 'claims consistency'
        print(f'{entry["manuscript"]}  [{flag}]')
        print(f'  {entry["n_mismatches"]} mismatches, {entry["n_review"]} to review')
        for f in entry['findings'][:40]:
            print(console_safe(
                f'    line {f["line"]:>4}  {f["severity"]:<8} {f["claimed"]}  {f["detail"]}'))
            print(console_safe(f'              {f["context"]}'))
        print()
    if result['undeclared_divergence']:
        print('UNDECLARED DIVERGENCE: '
              + ', '.join(result['undeclared_divergence']))
        print(f'A manuscript that disagrees with the artifacts must say so. Add the '
              f'{STALE_MARKER} marker or correct the numbers.')
        return 1
    print('consistent' if result['consistent'] else 'divergence declared, not yet corrected')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

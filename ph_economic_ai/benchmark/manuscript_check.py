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
from pathlib import Path

from ph_economic_ai.benchmark.paths import ACCURACY_REPORT, BENCHMARK_DIR

DOCS = BENCHMARK_DIR.parent.parent / 'docs'
MANUSCRIPTS = (
    DOCS / 'manuscript' / '2026-06-10-thesis-manuscript.md',
    DOCS / 'manuscript' / '2026-07-26-baseline-specification-note.md',
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


def artifact_sample_sizes(report: dict) -> set[int]:
    """Every integer the artifacts describe as a sample or evaluation count."""
    found: set[int] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ('n', 'n_long', 'n_calib', 'n_eval') and isinstance(value, int):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(report)
    return found


def scan_sample_sizes(text: str) -> list[dict]:
    """Every `n = <int>` a manuscript asserts, with its line and context."""
    out = []
    for lineno, line in enumerate(text.splitlines(), start=1):
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


def check_verdicts(text: str, report: dict) -> list[dict]:
    """Audit verdicts asserted in prose that disagree with the artifacts."""
    findings = []
    for row in report.get('audit', []):
        target, verdict = row.get('target'), row.get('verdict')
        if not target or not verdict:
            continue
        opposite = 'efficient' if verdict == 'predictable' else 'predictable'
        for lineno, line in enumerate(text.splitlines(), start=1):
            low = line.lower()
            if target in low and opposite in low and verdict not in low:
                findings.append({
                    'kind': 'verdict',
                    'severity': 'mismatch',
                    'line': lineno,
                    'claimed': opposite,
                    'context': line.strip()[:160],
                    'detail': f'artifacts report {target} as {verdict}',
                })
    return findings


def check_manuscript(path: Path, report: dict) -> dict:
    text = path.read_text(encoding='utf-8')
    findings = check_sample_sizes(text, report) + check_verdicts(text, report)
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
    report = json.loads(Path(report_path or ACCURACY_REPORT).read_text(encoding='utf-8'))
    results = [check_manuscript(p, report) for p in manuscripts if p.exists()]
    total = sum(r['n_mismatches'] for r in results)
    undeclared = [r['manuscript'] for r in results
                  if r['n_mismatches'] and not r['declares_divergence']]
    return {
        'artifact_sample_sizes': sorted(artifact_sample_sizes(report)),
        'manuscripts': results,
        'total_mismatches': total,
        'undeclared_divergence': undeclared,
        'consistent': total == 0,
    }


def main() -> int:
    result = run()
    print(f'artifact sample sizes: {result["artifact_sample_sizes"]}\n')
    for entry in result['manuscripts']:
        flag = 'declares divergence' if entry['declares_divergence'] else 'claims consistency'
        print(f'{entry["manuscript"]}  [{flag}]')
        print(f'  {entry["n_mismatches"]} mismatches, {entry["n_review"]} to review')
        for f in entry['findings'][:40]:
            print(f'    line {f["line"]:>4}  {f["severity"]:<8} {f["claimed"]}  {f["detail"]}')
            print(f'              {f["context"]}')
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

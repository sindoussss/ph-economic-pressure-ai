"""Multiple-comparison correction over the family of confirmatory DM tests.

The audit runs several Diebold–Mariano tests for "beats the strongest naive
baseline". Testing many hypotheses at α = 0.05 inflates the family-wise false
positive rate, so a reviewer will (rightly) ask which findings survive a
correction. This module answers that with two standard procedures:

  * **Bonferroni** — controls the family-wise error rate (FWER); the strictest,
    protects against *any* false positive.
  * **Benjamini–Hochberg** — controls the false discovery rate (FDR); less
    conservative, the modern default for screening several hypotheses.

It reads two frozen artifacts and writes `multiple_testing.json`. Pure stdlib —
no LLM, no Qt — so it stays inside the validated benchmark. Reproduce with
`python -m ph_economic_ai.benchmark.multiple_testing`.

## What is in the family, and why

  1. `accuracy_report.json` — nodes with verdict `beats_best_naive` and a real DM
     p-value. Under the corrected baseline pool (§4.7) there are none, which is
     how this record came to be empty.
  2. `selection_holdout.json` — every row `selection.run()` actually tested. These
     are the benchmark's live DM tests, and the family the one positive result
     (`fuel_audit`, `confirmed_on_holdout`) was in fact selected from.

Leaving (2) out is what produced the defect this module was rewritten to fix: an
`n_tests = 0` multiplicity record sitting behind a claimed positive, while 23
DM tests went uncorrected next to it. `docs/preregistration/2026-08-12-food-subcategory-selection-holdout.md`
required a review before wiring them in, on the grounds that growing the family
retroactively tightens `alpha/m` for existing members. The review's finding is
that the risk did not apply: the family was empty, so nothing could be flipped.

Excluded, deliberately, and each for a reason that is not "it would spoil the
answer":

  * **Efficiency panel nulls** — they accept the null, so they raise a power
    question (`power.json`, §5.1), not a false-positive one.
  * **Per-method panel rows** (`audit_table.json`, `nowcast_table.json`) — a
    panel is one hypothesis tested with K candidates, not K hypotheses. Counting
    each method separately would inflate m with the very multiplicity that
    `selection.py`'s holdout protocol already handles by construction.
  * **The audit's own fuel row** (`accuracy_report.json` `audit[0]`, DM
    p = 0.0337) — this is the SAME hypothesis as `selection_holdout`'s
    `fuel_audit`, re-tested honestly on a holdout. Counting both would charge one
    claim twice.
  * **`corrected_predictability_map.json` `old/` p-values** — the superseded
    mean-free pool, retained for comparison and not a live test.

A change to that list changes m, and m divides every threshold here. Anything
added must be a hypothesis the benchmark actually tested and does not already
count elsewhere.
"""
from __future__ import annotations

import json

from ph_economic_ai.benchmark.paths import ACCURACY_REPORT, ARTIFACTS_DIR, artifact

_ARTIFACTS = ARTIFACTS_DIR
_REPORT = ACCURACY_REPORT
_HOLDOUT = artifact('selection_holdout.json')
_OUT = artifact('multiple_testing.json')


# ── Corrections (pure functions) ──────────────────────────────────────────────

def bonferroni_reject(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """FWER control: reject p ≤ α/m."""
    m = len(pvalues)
    thr = alpha / m if m else alpha
    return [p <= thr for p in pvalues]


def bonferroni_adjusted(pvalues: list[float]) -> list[float]:
    """Bonferroni-adjusted p-values: min(1, p·m)."""
    m = len(pvalues)
    return [min(1.0, p * m) for p in pvalues]


def benjamini_hochberg_reject(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """FDR control (step-up). Reject the k smallest p-values, where k is the
    largest rank i with p_(i) ≤ (i/m)·α."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    kmax = 0
    for rank, idx in enumerate(order, start=1):
        if pvalues[idx] <= rank / m * alpha:
            kmax = rank
    reject = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= kmax:
            reject[idx] = True
    return reject


def benjamini_hochberg_adjusted(pvalues: list[float]) -> list[float]:
    """BH-adjusted p-values (q-values), monotone and capped at 1."""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    q = [0.0] * m
    prev = 1.0
    # walk from the largest p-value down, enforcing monotonicity
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = min(prev, pvalues[idx] * m / rank)
        q[idx] = min(1.0, val)
        prev = q[idx]
    return q


# ── Family assembly ───────────────────────────────────────────────────────────

def _direction(skill) -> str:
    """Which side of the two-sided DM test a rejection would fall on.

    A DM p-value is two-sided: it rejects just as happily when the model is
    significantly WORSE than the naive baseline. Two of the three sub-0.05
    p-values in `selection_holdout.json` are exactly that (`dairy_eggs` at −35.0%,
    `sugar` at −82.1%), so a record that reported only "3 significant at 0.05"
    would read as three near-findings when one of them is the model losing badly.
    """
    if skill is None:
        return 'unknown'
    return 'favours_model' if float(skill) > 0 else 'favours_naive'


def build_family(report: dict) -> list[dict]:
    """The `accuracy_report.json` half: every claim of the form 'beats the
    strongest naive baseline' with a real p-value. Efficiency nulls are
    excluded — they accept the null, so they raise a power question (a separate
    issue), not a false-positive one.

    Empty under the corrected baseline pool. Kept, and kept dynamic, because a
    future genuine panel positive must land in the family without anyone
    remembering to put it there.
    """
    candidates = [
        ('MoM headline inflation (short, n=61)', ('nowcast_mom',)),
        ('MoM headline inflation (long, n=143)', ('mom_longsample', 'mom')),
        ('Food MoM inflation', ('food_nowcast', 'mom')),
        ('Electricity MoM inflation', ('electricity_nowcast', 'mom')),
        ('Electricity within-month driver', ('electricity_nowcast', 'driver_ablation')),
        ('Transport within-month driver (pre-robustness)',
         ('transport_nowcast', 'driver_ablation')),
    ]
    family = []
    for label, path in candidates:
        node = report
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if node.get('verdict') == 'beats_best_naive' and node.get('dm_p') is not None:
            skill = node.get('best_skill_vs_naive')
            family.append({
                'test': label,
                # Full path, not path[-1]: two candidates end in `driver_ablation`
                # and a bare leaf name would give them the same key.
                'key': '.'.join(path),
                'source': 'accuracy_report.json',
                'skill_vs_naive': skill,
                'dm_p': float(node['dm_p']),
                'direction': _direction(skill),
            })
    return family


def build_selection_family(holdout: dict) -> list[dict]:
    """The selection-honest DM tests: one per row `selection.run()` scored.

    Membership is read off the artifact rather than listed here, so extending
    `selection.run()` widens the family automatically. A row that returned
    `insufficient_data` carries no p-value and was never tested — including it
    would inflate the m that divides everyone else's threshold.
    """
    family = []
    for key, row in holdout.items():
        if not isinstance(row, dict) or row.get('holdout_dm_p') is None:
            continue
        skill = row.get('holdout_skill')
        family.append({
            'test': f'selection holdout: {key}',
            'key': key,
            'source': 'selection_holdout.json',
            'skill_vs_naive': skill,
            'dm_p': float(row['holdout_dm_p']),
            'direction': _direction(skill),
            'holdout_verdict': row.get('verdict'),
        })
    return family


def assemble_family(report: dict, holdout: dict) -> list[dict]:
    """Every DM test the benchmark ran, from both artifacts that hold one."""
    return build_family(report) + build_selection_family(holdout)


def correct(family: list[dict], alpha: float = 0.05) -> dict:
    ps = [f['dm_p'] for f in family]
    bonf = bonferroni_reject(ps, alpha)
    bonf_adj = bonferroni_adjusted(ps)
    bh = benjamini_hochberg_reject(ps, alpha)
    bh_adj = benjamini_hochberg_adjusted(ps)
    rows = []
    for f, br, ba, hr, ha in zip(family, bonf, bonf_adj, bh, bh_adj):
        rows.append({
            **f,
            # Derived here as well as in the builders so `correct()` stays usable
            # on a hand-built family — the synthetic-family guard passes one.
            'direction': f.get('direction') or _direction(f.get('skill_vs_naive')),
            'nominally_significant': bool(f['dm_p'] < alpha),
            'bonferroni_p': round(ba, 4),
            'survives_bonferroni': bool(br),
            'bh_q': round(ha, 4),
            'survives_bh': bool(hr),
        })
    rows.sort(key=lambda r: r['dm_p'])
    nominal = [r for r in rows if r['nominally_significant']]
    return {
        'alpha': alpha,
        'n_tests': len(family),
        'bonferroni_threshold': round(alpha / len(family), 5) if family else None,
        # The number a reader needs next to "3 of 23 came in under 0.05" before
        # that count means anything: alpha*m is what pure chance already buys.
        'expected_false_positives': round(alpha * len(family), 2),
        'n_nominally_significant': len(nominal),
        'n_nominally_significant_favouring_model': sum(
            1 for r in nominal if r['direction'] == 'favours_model'),
        'n_nominally_significant_favouring_naive': sum(
            1 for r in nominal if r['direction'] == 'favours_naive'),
        'survive_bonferroni': [r['test'] for r in rows if r['survives_bonferroni']],
        'survive_bh_only': [r['test'] for r in rows
                            if r['survives_bh'] and not r['survives_bonferroni']],
        'survive_neither': [r['test'] for r in rows if not r['survives_bh']],
        'tests': rows,
    }


def run() -> dict:
    report = json.loads(_REPORT.read_text())
    # A missing holdout artifact means `selection.run()` has not been run yet on a
    # fresh checkout, not that the family is empty. Correcting over the report
    # alone is still the right answer for that state; `benchmark.run` orders the
    # two so a full rebuild never lands here.
    holdout = json.loads(_HOLDOUT.read_text()) if _HOLDOUT.exists() else {}
    result = correct(assemble_family(report, holdout))
    # newline='\n' explicitly: text mode translates to CRLF on Windows, which made
    # this builder emit a different artifact than the same builder on Linux CI —
    # the platform-dependent-checkout problem `.gitattributes` documents for the
    # data CSVs, reached here through the writer instead of through git.
    _OUT.write_text(json.dumps(result, indent=2), encoding='utf-8', newline='\n')
    return result


def _main() -> int:
    r = run()
    # ASCII only in console output: this is a documented reproduce command, and a
    # Greek alpha crashes the default Windows console (cp1252 cannot encode it).
    print(f"Multiple-comparison correction over {r['n_tests']} DM tests "
          f"(alpha={r['alpha']}, Bonferroni threshold {r['bonferroni_threshold']}):\n")
    print(f"  {'test':44} {'DM p':>7} {'Bonf p':>7} {'BH q':>6}  {'side':<13} survives")
    for t in r['tests']:
        tag = ('Bonferroni+BH' if t['survives_bonferroni']
               else 'BH only' if t['survives_bh'] else 'neither')
        side = 'model' if t['direction'] == 'favours_model' else 'naive'
        star = '*' if t['nominally_significant'] else ' '
        print(f"  {t['test']:44} {t['dm_p']:>7}{star}{t['bonferroni_p']:>6} "
              f"{t['bh_q']:>6}  favours {side:<5} {tag}")
    print(f"\n* {r['n_nominally_significant']} of {r['n_tests']} land under "
          f"alpha={r['alpha']} uncorrected "
          f"({r['n_nominally_significant_favouring_model']} favouring the model, "
          f"{r['n_nominally_significant_favouring_naive']} favouring the naive "
          f"baseline); chance alone buys {r['expected_false_positives']}.")
    print(f"Survive the strict Bonferroni (FWER): {r['survive_bonferroni'] or 'none'}")
    print(f"Survive BH-FDR only (suggestive):     {r['survive_bh_only'] or 'none'}")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())

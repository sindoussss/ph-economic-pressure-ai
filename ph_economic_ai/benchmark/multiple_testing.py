"""Multiple-comparison correction over the family of confirmatory DM tests.

The audit runs several Diebold–Mariano tests for "beats the strongest naive
baseline". Testing many hypotheses at α = 0.05 inflates the family-wise false
positive rate, so a reviewer will (rightly) ask which findings survive a
correction. This module answers that with two standard procedures:

  * **Bonferroni** — controls the family-wise error rate (FWER); the strictest,
    protects against *any* false positive.
  * **Benjamini–Hochberg** — controls the false discovery rate (FDR); less
    conservative, the modern default for screening several hypotheses.

It reads the frozen `accuracy_report.json`, assembles the family of
confirmatory tests (only nodes with verdict `beats_best_naive` and a real DM
p-value), and writes `multiple_testing.json`. Pure numpy — no LLM, no Qt — so
it stays inside the validated benchmark. Reproduce with
`python -m ph_economic_ai.benchmark.multiple_testing`.
"""
from __future__ import annotations

import json
from pathlib import Path

_ARTIFACTS = Path(__file__).resolve().parent / 'artifacts'
_REPORT = _ARTIFACTS / 'accuracy_report.json'
_OUT = _ARTIFACTS / 'multiple_testing.json'


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

def build_family(report: dict) -> list[dict]:
    """The confirmatory DM tests: every claim of the form 'beats the strongest
    naive baseline' with a real p-value. Efficiency nulls are excluded — they
    accept the null, so they raise a power question (a separate issue), not a
    false-positive one.
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
            family.append({
                'test': label,
                'skill_vs_naive': node.get('best_skill_vs_naive'),
                'dm_p': float(node['dm_p']),
            })
    return family


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
            'bonferroni_p': round(ba, 4),
            'survives_bonferroni': bool(br),
            'bh_q': round(ha, 4),
            'survives_bh': bool(hr),
        })
    rows.sort(key=lambda r: r['dm_p'])
    return {
        'alpha': alpha,
        'n_tests': len(family),
        'bonferroni_threshold': round(alpha / len(family), 5) if family else None,
        'survive_bonferroni': [r['test'] for r in rows if r['survives_bonferroni']],
        'survive_bh_only': [r['test'] for r in rows
                            if r['survives_bh'] and not r['survives_bonferroni']],
        'survive_neither': [r['test'] for r in rows if not r['survives_bh']],
        'tests': rows,
    }


def run() -> dict:
    report = json.loads(_REPORT.read_text())
    result = correct(build_family(report))
    _OUT.write_text(json.dumps(result, indent=2))
    return result


def _main() -> int:
    r = run()
    print(f"Multiple-comparison correction over {r['n_tests']} confirmatory DM tests "
          f"(α={r['alpha']}, Bonferroni threshold {r['bonferroni_threshold']}):\n")
    print(f"  {'test':46} {'DM p':>7} {'Bonf p':>7} {'BH q':>6}  survives")
    for t in r['tests']:
        tag = ('Bonferroni+BH' if t['survives_bonferroni']
               else 'BH only' if t['survives_bh'] else 'neither')
        print(f"  {t['test']:46} {t['dm_p']:>7} {t['bonferroni_p']:>7} {t['bh_q']:>6}  {tag}")
    print(f"\nSurvive the strict Bonferroni (FWER): {r['survive_bonferroni']}")
    print(f"Survive BH-FDR only (suggestive):     {r['survive_bh_only']}")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())

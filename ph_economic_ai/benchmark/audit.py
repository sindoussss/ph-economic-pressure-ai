"""Predictability audit: run each economic target through the forecaster panel +
Diebold-Mariano test and assign an efficient/predictable verdict.
"""
from ph_economic_ai.benchmark.efficiency import run_panel

PANEL_METHODS = ['random_walk', 'drift', 'seasonal_naive', 'mean',
                 'arima', 'ets', 'ridge', 'hgb']

# Naive baselines. One baseline out-performing another says nothing about
# predictability — a mean-reverting series makes the historical mean beat the
# random walk by construction — so baselines are never eligible to be the method
# that earns a 'predictable' verdict. (Mirrors nowcast.mom_verdict, which already
# excludes its pool members from the candidate set.)
BASELINES = frozenset({'random_walk', 'drift', 'seasonal_naive', 'mean'})

#: Family-wise error rate the verdict is held to.
SELECTION_ALPHA = 0.05


def _candidates(panel: list) -> list:
    """Non-baseline rows that actually produced a test."""
    return [r for r in panel
            if r['method'] not in BASELINES and r.get('dm_p') is not None]


def selection_threshold(panel: list) -> float:
    """The p a candidate must clear, Bonferroni-corrected for the selection.

    The verdict does not test one hypothesis. It takes the BEST of k candidates
    and reports whether that one is significant, so judging the maximum at the
    uncorrected alpha inflates the false-positive rate by construction.

    k counts tests performed, not methods named: a method that returned no
    p-value was never a comparison and must not make the bar stricter for the
    ones that were.
    """
    k = len(_candidates(panel))
    return SELECTION_ALPHA / k if k else SELECTION_ALPHA


def selection_detail(panel: list) -> dict:
    """What the correction did, so the verdict can be audited rather than trusted.

    Correcting a verdict must not delete the evidence behind it. A reader has to
    be able to see that a method was nominally significant and that the
    correction is why the verdict says otherwise; hiding that would trade one
    misleading artifact for another.
    """
    cands = _candidates(panel)
    threshold = selection_threshold(panel)
    nominal = [r for r in cands if r['dm_p'] < SELECTION_ALPHA and r['skill_vs_rw'] > 0]
    best_nominal = max(nominal, key=lambda r: r['skill_vs_rw']) if nominal else None
    return {
        'n_candidates': len(cands),
        'alpha': SELECTION_ALPHA,
        'threshold': threshold,
        'nominally_significant': bool(nominal),
        'nominal_best_method': best_nominal['method'] if best_nominal else None,
        'nominal_best_dm_p': best_nominal['dm_p'] if best_nominal else None,
        'nominal_best_skill': best_nominal['skill_vs_rw'] if best_nominal else None,
    }


def verdict_from_panel(panel: list):
    """('predictable', best_row) if a *candidate* beats random walk at the
    selection-corrected threshold; else ('efficient', random_walk_row).

    The threshold was a bare 0.05 until 2026-08-20. On the committed panel that
    read `fuel predictable, ridge, skill 0.1211, dm_p 0.0337` -- a maximum over
    four candidates judged as though it were a single planned test. The same
    result scored in the declared 24-test family reads bonferroni_p 0.7104 and
    bh_q 0.2368, surviving neither, so the audit was asserting a verdict the
    project's own multiple-testing artifact contradicted, and four documents
    inherited it.

    This correction is over the k candidates this panel compares, which is
    self-contained: the family-wide correction is computed downstream of this
    verdict and depending on it here would be circular. The family remains the
    stricter authority. This only stops the verdict claiming more than its own
    comparison supports.
    """
    threshold = selection_threshold(panel)
    beats = [r for r in _candidates(panel)
             if r['dm_p'] < threshold and r['skill_vs_rw'] > 0]
    if beats:
        return 'predictable', max(beats, key=lambda r: r['skill_vs_rw'])
    rw = next((r for r in panel if r['method'] == 'random_walk'), panel[0])
    return 'efficient', rw


def run_audit(target_names, min_train: int = 24, registry=None) -> list:
    """Audit each named target. registry defaults to targets.TARGETS."""
    if registry is None:
        from ph_economic_ai.benchmark.targets import TARGETS
        registry = TARGETS

    rows = []
    for name in target_names:
        target = registry[name]
        try:
            frame = target.build_frame()
        except Exception as e:
            rows.append({'target': name, 'verdict': 'insufficient_data',
                         'error': str(e)[:120], 'n': 0})
            continue
        if len(frame) < min_train + 5:
            rows.append({'target': name, 'verdict': 'insufficient_data',
                         'n': int(len(frame))})
            continue
        feature_cols = [c for c in frame.columns if c != 'target']
        panel = run_panel(frame, PANEL_METHODS, min_train, feature_cols, target_col='target')
        verdict, best = verdict_from_panel(panel)
        rows.append({
            'target': name,
            'verdict': verdict,
            'best_method': best['method'],
            'best_skill': best['skill_vs_rw'],
            'best_dm_p': best.get('dm_p'),
            'n': int(panel[0]['n']),
            'selection': selection_detail(panel),
            'panel': panel,
        })
    return rows

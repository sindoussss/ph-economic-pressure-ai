"""Predictability audit: run each economic target through the forecaster panel +
Diebold-Mariano test and assign an efficient/predictable verdict.
"""
from ph_economic_ai.benchmark.efficiency import run_panel

PANEL_METHODS = ['random_walk', 'drift', 'seasonal_naive', 'arima', 'ets', 'ridge', 'hgb']


def verdict_from_panel(panel: list):
    """('predictable', best_row) if any method significantly beats random walk
    (dm_p < 0.05 and skill > 0); else ('efficient', random_walk_row)."""
    beats = [r for r in panel
             if r.get('dm_p') is not None and r['dm_p'] < 0.05 and r['skill_vs_rw'] > 0]
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
            'panel': panel,
        })
    return rows

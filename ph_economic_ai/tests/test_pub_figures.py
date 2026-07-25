import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_predictability_map_renders(tmp_path):
    from ph_economic_ai.benchmark import figures
    rows = [
        {'label': 'Electricity', 'skill': 0.28, 'verdict': 'predictable', 'note': 'robust'},
        {'label': 'MoM inflation', 'skill': 0.16, 'verdict': 'predictable', 'note': 'p<0.01'},
        {'label': 'Transport', 'skill': 0.15, 'verdict': 'rejected', 'note': 'artifact'},
        {'label': 'Food drivers', 'skill': 0.0, 'verdict': 'efficient', 'note': 'no edge'},
        {'label': '1-mo fuel', 'skill': -0.01, 'verdict': 'efficient', 'note': 'no edge'},
    ]
    out = tmp_path / 'pmap.png'
    figures.plot_predictability_map(rows, [out])
    assert out.exists() and out.stat().st_size > 0


def _beats(skill, p, n=151):
    return {'verdict': 'beats_best_naive', 'best_skill_vs_naive': skill,
            'dm_p': p, 'n': n}


def _null():
    return {'verdict': 'no_better_than_naive', 'best_naive': 'mean',
            'best_skill_vs_naive': 0.0, 'dm_p': None}


def test_build_rows_from_report():
    from ph_economic_ai.benchmark.render_pub_figures import build_rows
    report = {
        'skill': {'vs_random_walk': -0.0075},
        'mom_longsample': {'mom': _beats(0.1627, 0.001, 143)},
        'food_nowcast': {'mom': _beats(0.16, 0.0046),
                         'driver_ablation': _null(), 'driver_edge_robust': False},
        'electricity_nowcast': {'driver_ablation': _beats(0.2833, 0.0011),
                                'driver_edge_robust': True},
        'transport_nowcast': {'driver_ablation': _beats(0.1475, 0.021),
                              'driver_edge_robust': False},
    }
    rows = build_rows(report)
    by = {r['label'].splitlines()[0]: r for r in rows}
    assert by['Electricity inflation']['verdict'] == 'predictable'
    assert by['Transport inflation']['verdict'] == 'rejected'      # beats but not robust
    assert by['MoM inflation']['verdict'] == 'predictable'
    assert by['Food'][ 'verdict'] == 'efficient'                   # driver ablation null
    assert by['1-mo fuel · FX · YoY inflation']['verdict'] == 'efficient'
    assert abs(by['Electricity inflation']['skill'] - 0.2833) < 1e-9


def test_build_rows_reflects_null_verdicts():
    """Regression guard: the MoM and Food labels were hardcoded to 'predictable'
    with fixed p-values, so the published figure went on asserting a positive after
    the underlying verdicts had become nulls. Every label must be derived."""
    from ph_economic_ai.benchmark.render_pub_figures import build_rows
    report = {
        'skill': {'vs_random_walk': -0.0075},
        'mom_longsample': {'mom': _null()},
        'food_nowcast': {'mom': _null(), 'driver_ablation': _null(),
                         'driver_edge_robust': False},
        'electricity_nowcast': {'driver_ablation': _null(), 'driver_edge_robust': False},
        'transport_nowcast': {'driver_ablation': _null(), 'driver_edge_robust': False},
    }
    rows = build_rows(report)
    assert all(r['verdict'] != 'predictable' for r in rows)
    assert all(r['skill'] == 0.0 or r['skill'] < 0 for r in rows)

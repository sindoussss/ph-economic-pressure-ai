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


def test_build_rows_from_report():
    from ph_economic_ai.benchmark.render_pub_figures import build_rows
    report = {
        'skill': {'vs_random_walk': -0.0075},
        'mom_longsample': {'mom': {'best_skill_vs_naive': 0.1627}},
        'food_nowcast': {'mom': {'best_skill_vs_naive': 0.16},
                         'driver_ablation': {'best_skill_vs_naive': 0.0}, 'driver_edge_robust': False},
        'electricity_nowcast': {'driver_ablation': {'best_skill_vs_naive': 0.2833},
                                'driver_edge_robust': True},
        'transport_nowcast': {'driver_ablation': {'best_skill_vs_naive': 0.1475},
                              'driver_edge_robust': False},
    }
    rows = build_rows(report)
    by = {r['label'].splitlines()[0]: r for r in rows}
    assert by['Electricity inflation']['verdict'] == 'predictable'
    assert by['Transport inflation']['verdict'] == 'rejected'      # robust=False -> rejected
    assert by['1-mo fuel · FX · YoY inflation']['verdict'] == 'efficient'
    assert abs(by['Electricity inflation']['skill'] - 0.2833) < 1e-9

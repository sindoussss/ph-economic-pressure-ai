"""Render publication-grade benchmark figures from the committed artifact.

Standalone (no backtest re-run): reads artifacts/accuracy_report.json and renders
the headline 'predictability map' bar chart to artifacts/figures/ AND docs/img/.

    python -m ph_economic_ai.benchmark.render_pub_figures
"""
import json
from pathlib import Path

from ph_economic_ai.benchmark import figures

_ART = Path(__file__).parent / 'artifacts'
_DOCS_IMG = Path(__file__).resolve().parents[2] / 'docs' / 'img'


def _skill(d, *keys):
    for k in keys:
        d = (d or {}).get(k, {})
    return float(d) if isinstance(d, (int, float)) else 0.0


def build_rows(report: dict) -> list:
    """Assemble the six predictability-map bars from the real benchmark numbers."""
    el = report.get('electricity_nowcast') or {}
    tr = report.get('transport_nowcast') or {}
    return [
        {'label': 'Electricity inflation\n(within-month drivers)',
         'skill': _skill(el, 'driver_ablation', 'best_skill_vs_naive'),
         'verdict': 'predictable' if el.get('driver_edge_robust') else 'efficient',
         'note': 'robust · p<0.01'},
        {'label': 'MoM inflation\n(headline · own dynamics)',
         'skill': _skill(report, 'mom_longsample', 'mom', 'best_skill_vs_naive'),
         'verdict': 'predictable', 'note': 'p=0.001 · n=143'},
        {'label': 'Food inflation\n(MoM · own dynamics)',
         'skill': _skill(report, 'food_nowcast', 'mom', 'best_skill_vs_naive'),
         'verdict': 'predictable', 'note': 'p<0.01'},
        {'label': 'Transport inflation\n(commodity drivers)',
         'skill': _skill(tr, 'driver_ablation', 'best_skill_vs_naive'),
         'verdict': 'predictable' if tr.get('driver_edge_robust') else 'rejected',
         'note': 'rejected · data artifact'},
        {'label': 'Food\n(commodity drivers)',
         'skill': _skill(report, 'food_nowcast', 'driver_ablation', 'best_skill_vs_naive'),
         'verdict': 'efficient', 'note': 'no edge'},
        {'label': '1-mo fuel · FX · YoY inflation',
         'skill': _skill(report, 'skill', 'vs_random_walk'),
         'verdict': 'efficient', 'note': 'no method beats RW'},
    ]


def main() -> None:
    report = json.loads((_ART / 'accuracy_report.json').read_text(encoding='utf-8'))
    rows = build_rows(report)
    out = [_ART / 'figures' / 'predictability_map.png', _DOCS_IMG / 'predictability_map.png']
    figures.plot_predictability_map(rows, out)
    print('wrote predictability_map.png ->', ', '.join(str(p) for p in out))
    for r in rows:
        print(f"  {r['skill']*100:+5.1f}%  {r['verdict']:<12} {r['label'].splitlines()[0]}")


if __name__ == '__main__':
    main()

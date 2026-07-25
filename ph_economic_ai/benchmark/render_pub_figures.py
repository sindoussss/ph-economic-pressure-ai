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


def _node(d, *keys):
    """Walk to a nested dict node, tolerating absent branches."""
    for k in keys:
        d = (d or {}).get(k, {})
    return d if isinstance(d, dict) else {}


def _verdict_note(node: dict, positive: str = 'predictable',
                  negative: str = 'efficient') -> tuple[str, str]:
    """Derive the label and its annotation from the REPORT, never a literal.

    These were hardcoded to 'predictable' with fixed p-values, so the figure kept
    asserting a positive after the underlying verdict had changed — the chart is a
    claim, and it has to be re-derived from the artifact like every other number.
    """
    beat = node.get('verdict') == 'beats_best_naive'
    if not beat:
        return negative, f"no edge vs {node.get('best_naive') or 'naive'}"
    p, n = node.get('dm_p'), node.get('n')
    bits = [b for b in (f'p={p:.3g}' if isinstance(p, (int, float)) else None,
                        f'n={n}' if n else None) if b]
    return positive, ' · '.join(bits) or 'significant'


def build_rows(report: dict) -> list:
    """Assemble the six predictability-map bars from the real benchmark numbers."""
    el = report.get('electricity_nowcast') or {}
    tr = report.get('transport_nowcast') or {}
    el_drv, tr_drv = _node(el, 'driver_ablation'), _node(tr, 'driver_ablation')
    mom = _node(report, 'mom_longsample', 'mom')
    food_mom = _node(report, 'food_nowcast', 'mom')
    food_drv = _node(report, 'food_nowcast', 'driver_ablation')

    el_v, el_n = _verdict_note(el_drv)
    if el_v == 'predictable' and not el.get('driver_edge_robust'):
        el_v, el_n = 'rejected', 'not robust'      # robustness gate overrides
    tr_v, tr_n = _verdict_note(tr_drv)
    if tr_v == 'predictable' and not tr.get('driver_edge_robust'):
        tr_v, tr_n = 'rejected', 'data artifact'
    mom_v, mom_n = _verdict_note(mom)
    fm_v, fm_n = _verdict_note(food_mom)
    fd_v, fd_n = _verdict_note(food_drv)

    return [
        {'label': 'Electricity inflation\n(within-month drivers)',
         'skill': _skill(el_drv, 'best_skill_vs_naive'),
         'verdict': el_v, 'note': el_n},
        {'label': 'MoM inflation\n(headline · own dynamics)',
         'skill': _skill(mom, 'best_skill_vs_naive'),
         'verdict': mom_v, 'note': mom_n},
        {'label': 'Food inflation\n(MoM · own dynamics)',
         'skill': _skill(food_mom, 'best_skill_vs_naive'),
         'verdict': fm_v, 'note': fm_n},
        {'label': 'Transport inflation\n(commodity drivers)',
         'skill': _skill(tr_drv, 'best_skill_vs_naive'),
         'verdict': tr_v, 'note': tr_n},
        {'label': 'Food\n(commodity drivers)',
         'skill': _skill(food_drv, 'best_skill_vs_naive'),
         'verdict': fd_v, 'note': fd_n},
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

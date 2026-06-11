"""Render the three backtest figures into artifacts/figures/."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                      # headless
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent / 'artifacts' / 'figures'


def _ensure_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_pred_vs_actual(dates, y_true, y_pred, low, high):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, y_true, label='Actual (World Bank RON95)', color='black')
    ax.plot(dates, y_pred, label='Forecast', color='tab:blue')
    ax.fill_between(dates, low, high, alpha=0.2, color='tab:blue', label='90% conformal band')
    ax.set_title('1-month-ahead RON95 forecast vs actual')
    ax.set_ylabel('PHP/liter'); ax.legend(); ax.tick_params(axis='x', rotation=45)
    fig.tight_layout(); fig.savefig(FIG_DIR / 'pred_vs_actual.png', dpi=120); plt.close(fig)


def plot_baseline_bars(rmse_model, rmse_rw, rmse_sn):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(['Model', 'Random walk', 'Seasonal naive'], [rmse_model, rmse_rw, rmse_sn],
           color=['tab:blue', 'tab:gray', 'tab:gray'])
    ax.set_title('RMSE vs baselines (lower is better)'); ax.set_ylabel('RMSE (PHP/liter)')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'baseline_bars.png', dpi=120); plt.close(fig)


def plot_proxy_scatter(proxy_vals, gold_vals):
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(gold_vals, proxy_vals, s=12, alpha=0.6)
    lo = min(min(gold_vals), min(proxy_vals)); hi = max(max(gold_vals), max(proxy_vals))
    ax.plot([lo, hi], [lo, hi], color='black', linewidth=1, label='y = x')
    ax.set_xlabel('World Bank RON95'); ax.set_ylabel('RBOB proxy'); ax.legend()
    ax.set_title('Proxy vs gold')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'proxy_scatter.png', dpi=120); plt.close(fig)


def plot_method_skill_bar(rows):
    """Bar of skill-vs-random-walk per method; red where DM p<0.05 (sig. different)."""
    _ensure_dir()
    rows = sorted(rows, key=lambda r: r['skill_vs_rw'])
    names = [r['method'] for r in rows]
    skills = [r['skill_vs_rw'] for r in rows]
    colors = ['tab:red' if (r['dm_p'] is not None and r['dm_p'] < 0.05) else 'tab:gray'
              for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, skills, color=colors)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Skill vs random walk (>0 beats naive)')
    ax.set_title('Forecaster panel - none beats random walk')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'method_skill_bar.png', dpi=120); plt.close(fig)


def plot_passthrough(cost_delta, pump_delta, beta_total):
    """Scatter of delta-pump vs delta-cost with the fitted pass-through slope."""
    _ensure_dir()
    import numpy as np
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(cost_delta, pump_delta, s=12, alpha=0.6)
    xs = np.array([min(cost_delta), max(cost_delta)])
    ax.plot(xs, beta_total * xs, color='black', linewidth=1,
            label=f'pass-through beta={beta_total:.2f}')
    ax.set_xlabel('delta landed cost (PHP/L)'); ax.set_ylabel('delta pump price (PHP/L)')
    ax.legend(); ax.set_title('DOE pass-through')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'passthrough.png', dpi=120); plt.close(fig)


def plot_audit_verdicts(rows):
    """Per-target best-skill bar, colored green=predictable / gray=efficient."""
    _ensure_dir()
    rows = [r for r in rows if r.get('verdict') in ('efficient', 'predictable')]
    names = [r['target'] for r in rows]
    skills = [r.get('best_skill', 0.0) for r in rows]
    colors = ['tab:green' if r['verdict'] == 'predictable' else 'tab:gray' for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(names, skills, color=colors)
    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel('Best skill vs random walk')
    ax.set_title('Predictability audit (green = predictable)')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'audit_verdicts.png', dpi=120); plt.close(fig)


def plot_nowcast(dates, actual, nowcast, naive):
    """Inflation: actual vs nowcast vs naive (last published) over the backtest."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, actual, label='Actual inflation', color='black')
    ax.plot(dates, nowcast, label='Nowcast', color='tab:blue')
    ax.plot(dates, naive, label='Naive (last published)', color='tab:gray', linestyle='--')
    ax.set_ylabel('YoY inflation (%)'); ax.legend(); ax.tick_params(axis='x', rotation=45)
    ax.set_title('CPI nowcast vs actual vs naive')
    fig.tight_layout(); fig.savefig(FIG_DIR / 'nowcast.png', dpi=120); plt.close(fig)


# ── Publication-grade headline benchmark (the "what can/can't we forecast" chart) ──

_PMAP_COLORS = {'predictable': '#15803D', 'efficient': '#C0C4CC', 'rejected': '#B3261E'}


def plot_predictability_map(rows, out_paths):
    """Headline benchmark bar chart: skill-vs-naive per target, coloured by verdict.

    rows: list of dicts {label, skill (float fraction, e.g. 0.28), verdict in
          'predictable'|'efficient'|'rejected', note (short annotation)}.
    out_paths: iterable of Path — the same PNG is saved to each (e.g. artifacts + docs/img).
    Editorial / AI-release style: off-white, serif title, value labels, 0 = naive baseline,
    rejected bars hatched. Honest by design — shows the efficient/rejected bars too.
    """
    from matplotlib.patches import Patch
    rows = sorted(rows, key=lambda r: r['skill'])              # best ends up on top (barh)
    labels = [r['label'] for r in rows]
    skills = [r['skill'] * 100.0 for r in rows]
    colors = [_PMAP_COLORS.get(r['verdict'], '#C0C4CC') for r in rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.0), facecolor='#FBFBFA')
    ax.set_facecolor('#FBFBFA')
    bars = ax.barh(labels, skills, color=colors, height=0.62, zorder=3)
    for r, b in zip(rows, bars):
        if r['verdict'] == 'rejected':
            b.set_hatch('////'); b.set_edgecolor('#FFFFFF')

    ax.axvline(0, color='#1C1E26', linewidth=1.2, zorder=4)
    for i, r in enumerate(rows):
        s = r['skill'] * 100.0
        ax.text(s + (0.9 if s >= 0 else -0.9), i, f'{s:+.0f}%', va='center',
                ha='left' if s >= 0 else 'right', fontsize=11, fontweight='bold',
                color='#1C1E26', zorder=5)
        if r.get('note'):
            ax.text(s + (7.0 if s >= 0 else 3.0), i, r['note'], va='center', ha='left',
                    fontsize=7.5, color='#9AA0AA', zorder=5)

    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#E5E7EB')
    ax.tick_params(axis='y', labelsize=9.5, colors='#1C1E26', length=0)
    ax.tick_params(axis='x', labelsize=8, colors='#9AA0AA')
    ax.set_xlabel('Skill vs naive baseline  ·  % RMSE improvement  ·  >0 beats naive',
                  fontsize=9, color='#6B7280')
    ax.set_xlim(min(min(skills) - 8, -8), max(skills) + 20)
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.6, zorder=0); ax.set_axisbelow(True)

    fig.suptitle('What Strata can — and can’t — forecast', x=0.015, ha='left',
                 fontsize=16, fontweight='bold', color='#1C1E26', family='serif')
    ax.set_title('Philippine fuel & inflation  ·  strictly-causal walk-forward backtest, DM-tested',
                 loc='left', fontsize=9.5, color='#9AA0AA', pad=12)
    leg = [Patch(facecolor=_PMAP_COLORS['predictable'], label='Predictable (beats naive)'),
           Patch(facecolor=_PMAP_COLORS['efficient'], label='Efficient (no edge)'),
           Patch(facecolor=_PMAP_COLORS['rejected'], hatch='////', edgecolor='#FFFFFF',
                 label='Rejected (data artifact)')]
    ax.legend(handles=leg, loc='lower right', frameon=False, fontsize=8)
    fig.text(0.015, 0.012,
             'Skill = % RMSE improvement over the strongest naive baseline (random walk / '
             'seasonal naive). Rejected = an apparent edge that fails the preliminary-data '
             'robustness check.', fontsize=6.5, color='#9AA0AA')

    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200, facecolor='#FBFBFA')
    plt.close(fig)

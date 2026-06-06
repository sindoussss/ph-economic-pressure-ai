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

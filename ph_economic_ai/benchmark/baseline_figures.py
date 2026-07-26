"""Render the three baseline-specification figures from the frozen artifacts.

These carry the paper's methodological result, which is otherwise text and tables:

  fig5_spurious_skill.png   S(rho) with the rho = 1/2 crossover, the simulation
                            check, and every real target at its measured rho
  fig6_size_distortion.png  false-positive rate against rho, for both sample
                            sizes and both baseline pools, against nominal alpha
  fig7_fredmd_exposure.png  the distribution of rho across FRED-MD, split by
                            whether the series was differenced

Standalone: reads only committed artifacts, renders to artifacts/figures/ and
mirrors to docs/img/. No backtest re-run.

    python -m ph_economic_ai.benchmark.baseline_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')                       # headless
import matplotlib.pyplot as plt
import numpy as np

from ph_economic_ai.benchmark.baseline_theory import CROSSOVER_RHO, spurious_skill
from ph_economic_ai.benchmark.paths import DOCS_IMG_DIR, FIGURES_DIR, artifact

_VULNERABLE = '#c44536'      # below the crossover: the random walk is a weak baseline
_SAFE = '#2a6f97'            # at or above it: the random walk is appropriate


def _save(fig, name: str) -> list:
    outs = []
    for d in (FIGURES_DIR, DOCS_IMG_DIR):
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        fig.savefig(p, dpi=140, bbox_inches='tight')
        outs.append(p)
    plt.close(fig)
    return outs


def fig_spurious_skill(theory: dict) -> list:
    """S(rho): the closed form, the simulation check, and the real targets."""
    fig, ax = plt.subplots(figsize=(8, 4.6))

    rho = np.linspace(-0.7, 0.9, 400)
    s = np.array([spurious_skill(r) for r in rho])
    ax.plot(rho, s * 100, color='black', lw=2,
            label=r'$S(\rho)=1-[2(1-\rho)]^{-1/2}$  (closed form)')
    ax.fill_between(rho, 0, s * 100, where=s > 0, color=_VULNERABLE, alpha=0.13)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(CROSSOVER_RHO, color=_SAFE, ls='--', lw=1.4)
    ax.annotate(r'crossover  $\rho=\frac{1}{2}$', xy=(CROSSOVER_RHO, 26),
                xytext=(0.56, 30), fontsize=9, color=_SAFE)
    ax.annotate('a model carrying NO information\nis credited with this much "skill"',
                xy=(-0.66, 14), fontsize=8.5, color=_VULNERABLE, style='italic')

    sim = theory['simulation']
    ax.scatter([c['rho'] for c in sim], [c['simulated'] * 100 for c in sim],
               marker='x', s=44, color='dimgray', zorder=4,
               label='simulated through walk_forward')

    tg = theory['targets']
    ax.scatter([t['rho'] for t in tg], [t['observed_mean_vs_rw_skill'] * 100 for t in tg],
               s=64, facecolor=_VULNERABLE, edgecolor='black', lw=0.7, zorder=5,
               label='real MoM targets (observed)')
    # Three targets sit within 0.03 of each other in rho, so a uniform offset makes
    # their labels illegible. Fan them out explicitly and draw leader lines.
    offsets = {
        'electricity MoM': (10, 6),
        'transport MoM': (10, 6),
        'headline MoM': (-18, -26),
        'headline MoM (long)': (26, -14),
        'food MoM': (30, 10),
    }
    for t in tg:
        lbl = t['target'].replace(' MoM', '')
        ax.annotate(lbl, xy=(t['rho'], t['observed_mean_vs_rw_skill'] * 100),
                    xytext=offsets.get(t['target'], (8, -10)),
                    textcoords='offset points', fontsize=7.5,
                    arrowprops=dict(arrowstyle='-', lw=0.6, color='dimgray',
                                    shrinkA=0, shrinkB=3))

    ax.set_xlabel(r'lag-1 autocorrelation of the target, $\rho$')
    ax.set_ylabel('apparent skill over the random walk (%)')
    ax.set_title('A mean-predictor out-scores the random walk whenever ' r'$\rho<\frac{1}{2}$',
                 fontsize=11)
    ax.set_xlim(-0.7, 0.9); ax.set_ylim(-60, 46)
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(alpha=0.18)
    return _save(fig, 'fig5_spurious_skill.png')


def fig_size_distortion(size: dict) -> list:
    """False-positive rate by rho, both sample sizes, both pools."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cells = size['size']
    ns = sorted({c['n'] for c in cells})
    styles = {ns[0]: ('o--', 0.55), ns[-1]: ('o-', 1.0)}

    for n in ns:
        rows = sorted([c for c in cells if c['n'] == n], key=lambda c: c['rho'])
        fmt, alpha = styles[n]
        ax.plot([r['rho'] for r in rows], [r['rate_pool_without_mean'] * 100 for r in rows],
                fmt, color=_VULNERABLE, alpha=alpha, lw=2, ms=6,
                label=f'pool WITHOUT the mean, n={n}')

    rows = sorted([c for c in cells if c['n'] == ns[-1]], key=lambda c: c['rho'])
    ax.plot([r['rho'] for r in rows], [r['rate_pool_with_mean'] * 100 for r in rows],
            'o-', color=_SAFE, lw=2, ms=6, label='pool WITH the mean (either n)')

    ax.axhline(size['alpha'] * 100, color='black', ls=':', lw=1.4)
    ax.annotate(f"nominal $\\alpha$ = {size['alpha']:.0%}", xy=(0.60, 8), fontsize=9)
    ax.axvline(CROSSOVER_RHO, color=_SAFE, ls='--', lw=1.2, alpha=0.7)
    ax.annotate(r'$\rho=\frac{1}{2}$', xy=(CROSSOVER_RHO, 74), xytext=(0.52, 74),
                fontsize=9, color=_SAFE)
    ax.annotate('more data makes it WORSE', xy=(0.0, 99.7), xytext=(0.055, 88),
                fontsize=9, color=_VULNERABLE, style='italic',
                arrowprops=dict(arrowstyle='->', color=_VULNERABLE, lw=1.1))

    ax.set_xlabel(r'lag-1 autocorrelation of the target, $\rho$')
    ax.set_ylabel('false-positive rate (%)')
    ax.set_title('Rejection rate on data containing NO signal — every rejection is a false positive',
                 fontsize=11)
    ax.set_ylim(-4, 108)
    ax.legend(fontsize=8.5, loc='center right', framealpha=0.9)
    ax.grid(alpha=0.18)
    return _save(fig, 'fig6_size_distortion.png')


def fig_fredmd_exposure(vs: dict) -> list:
    """Distribution of rho across FRED-MD, split by differenced vs level."""
    rows = vs['series']
    diffed = [r['rho'] for r in rows if 'difference' in r['transform']]
    levels = [r['rho'] for r in rows if 'difference' not in r['transform']]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.linspace(-0.8, 1.0, 37)
    ax.hist(diffed, bins=bins, color=_VULNERABLE, alpha=0.85,
            label=f'differenced / growth-rate  (n={len(diffed)})')
    ax.hist(levels, bins=bins, color=_SAFE, alpha=0.85,
            label=f'level / log level  (n={len(levels)})')

    ax.axvline(CROSSOVER_RHO, color='black', ls='--', lw=1.6)
    ax.axvspan(-0.8, CROSSOVER_RHO, color=_VULNERABLE, alpha=0.05)
    # Annotate in AXES fractions, not data units: the y-scale is a bin count whose
    # height depends on the snapshot, so data coordinates would drift off-axes on a
    # refresh (they did — one label left the canvas entirely).
    ax.annotate(r'$\rho=\frac{1}{2}$: at or above this,' '\na random walk is an\nappropriate baseline',
                xy=(0.72, 0.62), xycoords='axes fraction', fontsize=8.5, ha='left')
    ax.annotate(f"{vs['share_vulnerable']:.1%} of the panel\nsits below the threshold",
                xy=(0.03, 0.86), xycoords='axes fraction', fontsize=9.5,
                color=_VULNERABLE, weight='bold', ha='left')
    ax.annotate(f"median $\\rho$ = {vs['median_rho']:+.3f}",
                xy=(0.03, 0.78), xycoords='axes fraction', fontsize=8.5, ha='left')

    ax.set_xlabel(r'lag-1 autocorrelation after the recommended transform, $\rho$')
    ax.set_ylabel('number of FRED-MD series')
    ax.set_title('Differencing moves a target out of the regime where a random walk is valid',
                 fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.18, axis='y')
    return _save(fig, 'fig7_fredmd_exposure.png')


def main() -> None:
    theory = json.loads(artifact('baseline_theory.json').read_text(encoding='utf-8'))
    size = json.loads(artifact('baseline_size.json').read_text(encoding='utf-8'))
    vs = json.loads(artifact('vulnerability_survey.json').read_text(encoding='utf-8'))

    for fn, arg in ((fig_spurious_skill, theory), (fig_size_distortion, size),
                    (fig_fredmd_exposure, vs)):
        outs = fn(arg)
        print(f"  wrote {outs[0].name} -> {', '.join(str(p.parent) for p in outs)}")


if __name__ == '__main__':
    main()

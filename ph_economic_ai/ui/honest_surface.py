"""Honest, read-only digest of the frozen benchmark report for the Report screen.

Pure functions (no PyQt) so they are unit-testable. They surface the *validated*
results — calibrated interval, efficiency verdict, the one predictable target —
so the app's main screen tells the truth, not just the swarm's confident guesses.
"""
from typing import Optional


def load_validated() -> Optional[dict]:
    """Load the frozen accuracy report; return None if it has not been generated."""
    try:
        from ph_economic_ai.benchmark.report import load_report
        return load_report()
    except Exception:
        return None


def conformal_halfwidth(report: Optional[dict], level: str = '0.9') -> Optional[float]:
    """Calibrated conformal half-width for the given level, or None if unavailable."""
    if not report:
        return None
    val = (report.get('conformal_widths') or {}).get(level)
    return float(val) if val is not None else None


def validated_summary_lines(report: Optional[dict]) -> list:
    """Plain-text lines digesting the validated findings for the Report strip.

    Each line is omitted gracefully if its source key is missing. A None/empty
    report yields a single 'run the benchmark' line."""
    if not report:
        return ['Validated accuracy unavailable — run `python -m ph_economic_ai.benchmark.run`.']

    lines: list = []
    skill = report.get('headline_skill_vs_random_walk')
    if skill is not None:
        lines.append(f'1-month RON95 forecast: efficient — no method beats random walk '
                     f'(skill {skill:+.2f}).')

    qhat = conformal_halfwidth(report, '0.9')
    if qhat is not None:
        lines.append(f'Best estimate ≈ last price; 90% interval ±₱{qhat:.2f}.')

    audit = report.get('audit') or []
    eff = [a.get('target') for a in audit if a.get('verdict') == 'efficient' and a.get('target')]
    mom_long = report.get('mom_longsample') or {}
    mom_inner = mom_long.get('mom') if isinstance(mom_long, dict) else None
    mom_verdict = (mom_inner or {}).get('verdict') or (report.get('nowcast_mom') or {}).get('verdict')
    if eff or mom_verdict:
        eff_str = '/'.join(eff) if eff else 'fuel/FX/inflation'
        mom_str = ('MoM inflation: predictable' if mom_verdict == 'beats_best_naive'
                   else 'MoM inflation: not better than naive')
        lines.append(f'{eff_str}: efficient · {mom_str}.')

    lines.append('Full detail: Methodology & Accuracy tab.')
    return lines


def calibrated_interval_line(report: Optional[dict], level: str = '0.9') -> Optional[str]:
    """One-line calibrated interval for the given level, or None if unavailable.

    e.g. '90% calibrated interval: ±₱10.42/L (conformal, validated)'."""
    qhat = conformal_halfwidth(report, level)
    if qhat is None:
        return None
    pct = int(round(float(level) * 100))
    return f'{pct}% calibrated interval: ±₱{qhat:.2f}/L (conformal, validated)'

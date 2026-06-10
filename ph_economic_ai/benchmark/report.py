"""Assemble and persist the frozen accuracy_report.json."""
import json
from datetime import datetime, timezone
from pathlib import Path

ARTIFACTS = Path(__file__).parent / 'artifacts'
REPORT_PATH = ARTIFACTS / 'accuracy_report.json'

REQUIRED_KEYS = (
    'generated_at', 'horizon', 'date_range', 'n_months',
    'model_metrics', 'baseline_metrics', 'skill',
    'headline_skill_vs_random_walk', 'conformal_widths', 'calibration',
    'proxy_validation', 'data_hash', 'limitations',
    'ablation', 'selected_variant',
    'efficiency', 'passthrough',
    'audit',
    'nowcast',
    'nowcast_mom',
    'mom_driver_ablation',
    'mom_longsample',
    'transport_nowcast',
    'food_nowcast',
    'electricity_nowcast',
)

_LIMITATIONS = [
    'World Bank gold series lags ~1 year; live grading uses DOE prices.',
    'Conformal assumes exchangeable residuals; q-hat uses a rolling recent window.',
    'Food and electricity are deterministic transforms of gas, not independent forecasts.',
]


def build_report(date_range, n_months, model_metrics, baseline_metrics, skill,
                 calibration, proxy, data_hash, ablation=None, selected_variant=None,
                 efficiency=None, passthrough=None, audit=None, nowcast=None,
                 nowcast_mom=None, mom_driver_ablation=None, mom_longsample=None,
                 transport_nowcast=None, food_nowcast=None,
                 electricity_nowcast=None) -> dict:
    conformal_widths = {str(r['nominal']): r['qhat'] for r in calibration}
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'horizon': '1_month',
        'date_range': list(date_range),
        'n_months': n_months,
        'model_metrics': model_metrics,
        'baseline_metrics': baseline_metrics,
        'skill': skill,
        'headline_skill_vs_random_walk': skill.get('vs_random_walk'),
        'conformal_widths': conformal_widths,
        'calibration': calibration,
        'proxy_validation': proxy,
        'data_hash': data_hash,
        'ablation': ablation if ablation is not None else [],
        'selected_variant': selected_variant,
        'efficiency': efficiency if efficiency is not None else [],
        'passthrough': passthrough if passthrough is not None else {},
        'audit': audit if audit is not None else [],
        'nowcast': nowcast if nowcast is not None else {},
        'nowcast_mom': nowcast_mom if nowcast_mom is not None else {},
        'mom_driver_ablation': mom_driver_ablation if mom_driver_ablation is not None else {},
        'mom_longsample': mom_longsample if mom_longsample is not None else {},
        'transport_nowcast': transport_nowcast if transport_nowcast is not None else {},
        'food_nowcast': food_nowcast if food_nowcast is not None else {},
        'electricity_nowcast': electricity_nowcast if electricity_nowcast is not None else {},
        'limitations': _LIMITATIONS,
    }


def write_report(report: dict, path: Path = REPORT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


def load_report(path: Path = REPORT_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding='utf-8'))

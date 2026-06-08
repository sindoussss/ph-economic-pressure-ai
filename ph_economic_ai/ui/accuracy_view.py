"""Read-only Methodology & Accuracy view. Renders frozen artifacts + live log;
performs no computation of its own."""
from pathlib import Path

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QLabel, QScrollArea, QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem,
)

from ph_economic_ai.benchmark.report import load_report, REPORT_PATH
from ph_economic_ai.benchmark.report import ARTIFACTS

_FIG_DIR = ARTIFACTS / 'figures'


class AccuracyView(QWidget):
    def __init__(self, report_path: Path = REPORT_PATH, parent=None):
        super().__init__(parent)
        self._report_path = Path(report_path)
        self._report = self._safe_load()
        self._build()

    def _safe_load(self):
        try:
            return load_report(self._report_path)
        except FileNotFoundError:
            return None

    def headline_text(self) -> str:
        if self._report is None:
            return ('Accuracy report not found — run '
                    '`python -m ph_economic_ai.benchmark.run` to generate it.')
        r = self._report
        skill = r['headline_skill_vs_random_walk']
        m = r['model_metrics']
        lo, hi = r['date_range']
        verdict = 'beats' if skill > 0 else ('matches' if skill == 0 else 'does NOT beat')
        return (f"1-month RON95 forecast: MAE ₱{m['mae']:.2f}, "
                f"skill {skill:+.2f} vs random walk ({verdict} baseline), "
                f"over {r['n_months']} months ({lo}–{hi}).")

    def _build(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); col = QVBoxLayout(inner)

        col.addWidget(QLabel(f"<h2>Methodology &amp; Accuracy</h2>"))
        headline = QLabel(self.headline_text()); headline.setWordWrap(True)
        col.addWidget(headline)

        if self._report is not None:
            for name in ('pred_vs_actual.png', 'baseline_bars.png', 'proxy_scatter.png'):
                fp = _FIG_DIR / name
                if fp.exists():
                    lbl = QLabel(); lbl.setPixmap(QPixmap(str(fp)))
                    col.addWidget(lbl)
            col.addWidget(self._calibration_table())
            if self._report.get('ablation'):
                abl = QLabel('<b>Lever comparison (Phase 2)</b><br>'
                             + self.ablation_summary().replace('\n', '<br>'))
                abl.setWordWrap(True)
                col.addWidget(abl)
            if self._report.get('efficiency'):
                eff = QLabel('<b>Forecaster panel - efficiency</b><br>'
                             + self.efficiency_summary().replace('\n', '<br>'))
                eff.setWordWrap(True)
                col.addWidget(eff)
            _pt = self.passthrough_summary()
            if _pt:
                ptl = QLabel('<b>Mechanism</b><br>' + _pt)
                ptl.setWordWrap(True)
                col.addWidget(ptl)
            if self._report.get('audit'):
                aud = QLabel('<b>Predictability audit (PH economy)</b><br>'
                             + self.audit_summary().replace('\n', '<br>'))
                aud.setWordWrap(True)
                col.addWidget(aud)
            _nc = self.nowcast_summary()
            if _nc:
                ncl = QLabel('<b>Nowcast (present-before-release)</b><br>' + _nc)
                ncl.setWordWrap(True)
                col.addWidget(ncl)
            col.addWidget(self._limitations_label())

        col.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _calibration_table(self) -> QTableWidget:
        rows = self._report['calibration']
        t = QTableWidget(len(rows), 3)
        t.setHorizontalHeaderLabels(['Nominal', 'q-hat', 'Measured coverage'])
        for i, r in enumerate(rows):
            t.setItem(i, 0, QTableWidgetItem(f"{r['nominal']:.0%}"))
            t.setItem(i, 1, QTableWidgetItem(f"₱{r['qhat']:.2f}"))
            t.setItem(i, 2, QTableWidgetItem(f"{r['measured']:.0%}"))
        return t

    def _limitations_label(self) -> QLabel:
        items = ''.join(f'<li>{x}</li>' for x in self._report.get('limitations', []))
        lbl = QLabel(f"<b>Limitations</b><ul>{items}</ul>"); lbl.setWordWrap(True)
        return lbl

    def efficiency_summary(self) -> str:
        if not self._report:
            return ''
        rows = self._report.get('efficiency') or []
        lines = []
        for r in sorted(rows, key=lambda x: -x['skill_vs_rw']):
            p = 'n/a' if r.get('dm_p') is None else f"p={r['dm_p']:.2f}"
            lines.append(f"{r['method']}: skill {r['skill_vs_rw']:+.2f} vs RW ({p})")
        return '\n'.join(lines)

    def passthrough_summary(self) -> str:
        if not self._report:
            return ''
        p = self._report.get('passthrough') or {}
        if not p or p.get('beta_total') is None:
            return ''
        return (f"DOE pass-through: total beta={p['beta_total']:.2f} "
                f"(contemporaneous {p['beta0']:.2f}, lag-1 {p['beta1']:.2f}), "
                f"R2={p['r2']:.2f}; driver delta-autocorrelation={p['driver_acf1']:.2f} "
                f"(near 0 => random-walk input).")

    def audit_summary(self) -> str:
        if not self._report:
            return ''
        rows = self._report.get('audit') or []
        lines = []
        for r in rows:
            if r.get('verdict') == 'insufficient_data':
                lines.append(f"{r['target']}: insufficient data")
            else:
                lines.append(f"{r['target']}: {r['verdict']} "
                             f"(best {r['best_method']}, skill {r['best_skill']:+.2f})")
        return '\n'.join(lines)

    def nowcast_summary(self) -> str:
        if not self._report:
            return ''
        n = self._report.get('nowcast') or {}
        if not n or n.get('verdict') == 'insufficient_data':
            return ''
        return (f"CPI nowcast (estimate inflation before release): {n['verdict']} "
                f"— best {n['best_method']}, skill {n['best_skill']:+.2f} vs naive "
                f"(DM p={n['best_dm_p']}).")

    def ablation_summary(self) -> str:
        if not self._report:
            return ''
        rows = self._report.get('ablation') or []
        sel = self._report.get('selected_variant')
        lines = []
        for r in sorted(rows, key=lambda x: -x['skill_vs_rw']):
            mark = '  <- selected' if r['name'] == sel else ''
            lines.append(f"{r['name']}: skill {r['skill_vs_rw']:+.2f} vs RW, "
                         f"90% band P{r['band90']:.2f}{mark}")
        return '\n'.join(lines)

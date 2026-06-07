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

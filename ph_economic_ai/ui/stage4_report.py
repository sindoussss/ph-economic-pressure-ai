import calendar
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QScrollArea, QPushButton, QFileDialog,
)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from ph_economic_ai.engine.debate import AgentResponse
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.causal_chain_widget import CausalChainWidget, BSPAlertBanner
from ph_economic_ai.ui.regional_map import RegionalMapWidget
from ph_economic_ai.ui.policy_reco import PolicyRecoWidget


class Stage4ReportPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._responses: list = []
        self._consensus: dict = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        bar = QFrame()
        bar.setStyleSheet('background:#FFFFFF;border-bottom:1px solid #EAECF0;')
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(20, 10, 20, 10)

        self._status_lbl = QLabel('Simulation complete')
        self._status_lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')
        self._detail_lbl = QLabel('')
        self._detail_lbl.setStyleSheet('font-size:9px;color:#9EA3AE;')

        export_btn = QPushButton('Export PDF')
        export_btn.setStyleSheet(
            'QPushButton{background:#1C1E26;color:#FFFFFF;border-radius:8px;'
            'padding:6px 14px;font-size:10px;font-weight:600;border:none;}'
            'QPushButton:hover{background:#374151;}'
        )
        export_btn.clicked.connect(self._on_export)

        bar_layout.addWidget(self._status_lbl)
        bar_layout.addWidget(self._detail_lbl)
        bar_layout.addStretch()
        bar_layout.addWidget(export_btn)
        root.addWidget(bar)

        # BSP alert banner — shown when projected CPI breaches target
        self._bsp_banner = BSPAlertBanner()
        root.addWidget(self._bsp_banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body.setStyleSheet('background:#F7F8FA;')
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 20, 20, 20)
        body_layout.setSpacing(14)

        # Top row: two side-by-side columns
        top_row = QHBoxLayout()
        top_row.setSpacing(14)
        self._left = QVBoxLayout()
        self._right = QVBoxLayout()
        top_row.addLayout(self._left, stretch=1)
        top_row.addLayout(self._right, stretch=1)
        body_layout.addLayout(top_row)

        # Full-width policy recommendations below the columns
        self._reco_widget = PolicyRecoWidget()
        body_layout.addWidget(self._reco_widget)

        # Full-width regional map at the bottom
        self._map_widget = RegionalMapWidget()
        body_layout.addWidget(self._map_widget)

        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        self._placeholder = QLabel('Run a simulation to see the report.')
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet('font-size:14px;color:#9EA3AE;')
        self._left.addWidget(self._placeholder)

        # Causal chain widget — added to right column, populated via set_chain()
        self._chain_widget = CausalChainWidget()
        self._chain_widget.setMinimumHeight(320)

    def populate(self, responses: list, consensus: dict,
                 regressor, df, cv_rmse: float, scenario: dict):
        self._responses = responses
        self._consensus = consensus

        for layout in (self._left, self._right):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        self._build_left(consensus, responses)
        self._build_right(regressor, df, cv_rmse, scenario, consensus)

        rounds = max((r.round_num for r in responses), default=0)
        self._detail_lbl.setText(
            f'{len(set(r.agent_name for r in responses))} agents · '
            f'{rounds} rounds · {len(responses)} responses'
        )

    def populate_swarm(
        self,
        master_verdict,   # MasterVerdict
        regressor,
        df,
        cv_rmse: float,
        scenario: dict,
    ):
        """Populate the report from a MasterVerdict (swarm mode)."""
        from ph_economic_ai.engine.swarm import MasterVerdict as _MV  # noqa: F401

        # Clear existing content
        for layout in (self._left, self._right):
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # Build a consensus dict for reuse in _build_right
        est = master_verdict.final_estimate
        valid_estimates = [
            v.estimate for v in master_verdict.regional_verdicts if v.estimate is not None
        ]
        low = min(valid_estimates) if valid_estimates else None
        high = max(valid_estimates) if valid_estimates else None
        consensus = {
            'weighted_avg': est,
            'low': low,
            'high': high,
            'confidence_pct': master_verdict.confidence_pct,
            'verdicts': [],
        }
        self._consensus = consensus

        self._build_swarm_left(master_verdict, consensus)
        self._build_right(regressor, df, cv_rmse, scenario, consensus)

        n_regions = len(master_verdict.regional_verdicts)
        dissent = len(master_verdict.dissenting_regions)
        self._detail_lbl.setText(
            f'{n_regions} regional verdicts · {dissent} dissenting · swarm mode'
        )

    def _build_swarm_left(self, master_verdict, consensus: dict):
        """Left column for swarm mode: master consensus box + regional verdicts table."""
        card, cl = self._card('Swarm Consensus')

        avg = consensus.get('weighted_avg')
        conf = consensus.get('confidence_pct', 0)
        low = consensus.get('low')
        high = consensus.get('high')

        consensus_frame = QFrame()
        consensus_frame.setStyleSheet(
            'background:#F7F8FA;border-radius:10px;border:1px solid #EAECF0;'
        )
        cf_layout = QVBoxLayout(consensus_frame)
        cf_layout.setContentsMargins(12, 10, 12, 10)

        val_str = f'+₱{avg:.2f}/L' if avg is not None else 'No consensus'
        val_lbl = QLabel(val_str)
        val_lbl.setStyleSheet('font-size:24px;font-weight:700;color:#1C1E26;')

        sub_lbl = QLabel(f'Master judge estimate · {conf}% confidence')
        sub_lbl.setStyleSheet('font-size:9px;color:#6B7280;')

        range_row = QHBoxLayout()
        for label, value in [('Low', low), ('High', high), ('Confidence', f'{conf}%')]:
            col = QVBoxLayout()
            col.addWidget(self._muted(label))
            if isinstance(value, float) and value is not None:
                v_str = f'+₱{value:.2f}'
            else:
                v_str = str(value) if value is not None else '—'
            bold = QLabel(v_str)
            bold.setStyleSheet('font-size:11px;font-weight:600;color:#1C1E26;')
            col.addWidget(bold)
            range_row.addLayout(col)

        cf_layout.addWidget(val_lbl)
        cf_layout.addWidget(sub_lbl)
        cf_layout.addLayout(range_row)
        cl.addWidget(consensus_frame)

        # Dissenting regions
        if master_verdict.dissenting_regions:
            dissent_lbl = QLabel('Dissenting regions: ' + ', '.join(master_verdict.dissenting_regions))
            dissent_lbl.setWordWrap(True)
            dissent_lbl.setStyleSheet('font-size:9px;color:#EF4444;')
            cl.addWidget(dissent_lbl)

        # Regional verdicts table
        rv_card, rvcl = self._card('Regional Verdicts')
        for rv in master_verdict.regional_verdicts:
            rvf = QFrame()
            rvf.setStyleSheet('background:#F7F8FA;border-radius:8px;border:1px solid #EAECF0;')
            rvfl = QVBoxLayout(rvf)
            rvfl.setContentsMargins(10, 8, 10, 8)
            rvfl.setSpacing(3)

            head_row = QHBoxLayout()
            pair_str = ' & '.join(rv.region_pair)
            name_lbl = QLabel(pair_str[:50])
            name_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')
            est_str = f'+₱{rv.estimate:.2f}/L' if rv.estimate is not None else '—'
            est_lbl = QLabel(est_str)
            est_lbl.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
            head_row.addWidget(name_lbl)
            head_row.addStretch()
            head_row.addWidget(est_lbl)
            rvfl.addLayout(head_row)

            conf_lbl = QLabel(f'Confidence: {rv.confidence:.0%}')
            conf_lbl.setStyleSheet('font-size:8px;color:#9EA3AE;')
            rvfl.addWidget(conf_lbl)

            rvcl.addWidget(rvf)

        self._left.addWidget(card)
        self._left.addWidget(rv_card)
        self._left.addStretch()

    def _card(self, title: str):
        frame = QFrame()
        frame.setStyleSheet(
            'QFrame{background:#FFFFFF;border:1px solid #EAECF0;border-radius:12px;}'
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        lbl = QLabel(title)
        lbl.setStyleSheet('font-size:11px;font-weight:700;color:#1C1E26;')
        layout.addWidget(lbl)
        return frame, layout

    def _muted(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet('font-size:8px;color:#9EA3AE;text-transform:uppercase;')
        return lbl

    def _build_left(self, consensus: dict, responses: list):
        card, cl = self._card('Debate Summary')

        avg = consensus.get('weighted_avg')
        conf = consensus.get('confidence_pct', 0)
        low = consensus.get('low')
        high = consensus.get('high')

        consensus_frame = QFrame()
        consensus_frame.setStyleSheet(
            'background:#F7F8FA;border-radius:10px;border:1px solid #EAECF0;'
        )
        cf_layout = QVBoxLayout(consensus_frame)
        cf_layout.setContentsMargins(12, 10, 12, 10)

        val_lbl = QLabel(f'+₱{avg:.2f}/L' if avg is not None else 'No consensus')
        val_lbl.setStyleSheet('font-size:24px;font-weight:700;color:#1C1E26;')

        sub_lbl = QLabel(f'Weighted average · {conf}% agreement')
        sub_lbl.setStyleSheet('font-size:9px;color:#6B7280;')

        range_row = QHBoxLayout()
        for label, value in [('Low', low), ('High', high), ('Confidence', f'{conf}%')]:
            col = QVBoxLayout()
            col.addWidget(self._muted(label))
            if isinstance(value, float) and value is not None:
                v_str = f'+₱{value:.2f}'
            else:
                v_str = str(value) if value is not None else '—'
            bold = QLabel(v_str)
            bold.setStyleSheet('font-size:11px;font-weight:600;color:#1C1E26;')
            col.addWidget(bold)
            range_row.addLayout(col)

        cf_layout.addWidget(val_lbl)
        cf_layout.addWidget(sub_lbl)
        cf_layout.addLayout(range_row)
        cl.addWidget(consensus_frame)

        final_round = max((r.round_num for r in responses), default=1)
        final = [r for r in responses if r.round_num == final_round]
        for resp in final:
            vf = QFrame()
            vf.setStyleSheet('background:#F7F8FA;border-radius:8px;border:1px solid #EAECF0;')
            vfl = QVBoxLayout(vf)
            vfl.setContentsMargins(10, 8, 10, 8)
            vfl.setSpacing(4)

            head_row = QHBoxLayout()
            name_lbl = QLabel(resp.agent_name)
            name_lbl.setStyleSheet('font-size:10px;font-weight:600;color:#1C1E26;')
            est = f'+₱{resp.price_estimate:.2f}' if resp.price_estimate is not None else '—'
            est_lbl = QLabel(est)
            est_lbl.setStyleSheet('font-size:10px;font-weight:700;color:#1C1E26;')
            head_row.addWidget(name_lbl)
            head_row.addStretch()
            head_row.addWidget(est_lbl)
            vfl.addLayout(head_row)

            stmt_lbl = QLabel(resp.statement[:300])
            stmt_lbl.setWordWrap(True)
            stmt_lbl.setStyleSheet('font-size:9px;color:#374151;')
            vfl.addWidget(stmt_lbl)
            cl.addWidget(vf)

        self._left.addWidget(card)
        self._left.addStretch()

    def _build_right(self, regressor, df, cv_rmse, scenario, consensus):
        card, cl = self._card('Final Outputs')
        from ph_economic_ai.ui import honest_surface as _hs
        _report = _hs.load_validated()
        _exp = self._muted('Exploratory forecasts — not validated. Backtest shows no '
                           'method beats naive persistence for these (see Methodology & Accuracy).')
        _exp.setWordWrap(True)
        cl.addWidget(_exp)

        avg = consensus.get('weighted_avg') or 0.0
        X, y, feature_cols, df_feat = build_features(df)

        # Use live price from scenario when available (auto-fetched); fall back to df last row
        model_current = float(df_feat.iloc[-1]['gas_price'])
        live_current = float(scenario.get('current_price', model_current))
        price_offset = live_current - model_current
        week_est = avg / 4.0  # rough weekly estimate (monthly ÷ 4)

        # Run ML forecast and offset into live-price domain
        forecast_prices: np.ndarray | None = None
        try:
            features = self._make_features(df_feat, feature_cols, scenario)
            raw = ml.forecast(regressor, features)
            if raw is not None and len(raw) > 0:
                forecast_prices = np.array(raw) + price_offset
        except Exception:
            pass

        ml_3m = float(forecast_prices[2]) if forecast_prices is not None and len(forecast_prices) >= 3 else None
        ml_6m = float(forecast_prices[5]) if forecast_prices is not None and len(forecast_prices) >= 6 else None

        metrics = [
            ('Next week (AI est.)',  f'{week_est:+.2f} ₱/L'),
            ('Next month (AI est.)', f'{avg:+.2f} ₱/L'),
            ('3-month (ML)',         f'₱{ml_3m:.2f}/L' if ml_3m is not None else '—'),
            ('6-month (ML)',         f'₱{ml_6m:.2f}/L' if ml_6m is not None else '—'),
        ]
        grid = QHBoxLayout()
        for label, value in metrics:
            mf = QFrame()
            mf.setStyleSheet('background:#F7F8FA;border:1px solid #EAECF0;border-radius:9px;')
            mf_layout = QVBoxLayout(mf)
            mf_layout.setContentsMargins(10, 8, 10, 8)
            mf_layout.addWidget(self._muted(label))
            v_lbl = QLabel(value)
            v_lbl.setStyleSheet('font-size:16px;font-weight:700;color:#1C1E26;')
            mf_layout.addWidget(v_lbl)
            grid.addWidget(mf)
        cl.addLayout(grid)

        # 6-month forecast chart with actual future month names
        if forecast_prices is not None and len(forecast_prices) > 0:
            try:
                now = datetime.now()
                n = len(forecast_prices)
                month_labels = [
                    calendar.month_abbr[(now.month - 1 + i) % 12 + 1]
                    for i in range(1, n + 1)
                ]
                fig = Figure(figsize=(5, 2.2), facecolor='#F7F8FA')
                ax = fig.add_subplot(111)
                xs = list(range(1, n + 1))
                ax.plot(xs, forecast_prices, color='#1C1E26', linewidth=2)
                _band = _hs.conformal_halfwidth(_report) or cv_rmse
                ax.fill_between(xs, forecast_prices - _band, forecast_prices + _band,
                                alpha=0.15, color='#1C1E26')
                ax.set_facecolor('#F7F8FA')
                ax.set_xticks(xs)
                ax.set_xticklabels(month_labels, fontsize=7)
                ax.tick_params(axis='y', labelsize=7)
                fig.tight_layout(pad=1.0)
                canvas = FigureCanvasQTAgg(fig)
                canvas.setFixedHeight(200)
                cl.addWidget(canvas)
                _bcap = self._muted('90% calibrated interval (conformal)'
                                    if _hs.conformal_halfwidth(_report) is not None
                                    else '±cross-val RMSE (uncalibrated)')
                _bcap.setWordWrap(True)
                cl.addWidget(_bcap)
            except Exception:
                pass

        # Feature importances
        try:
            fi = ml.get_feature_importances(regressor, feature_cols)
            if fi:
                fi_fig = Figure(figsize=(5, 2.2), facecolor='#F7F8FA')
                fi_ax = fi_fig.add_subplot(111)
                names = list(fi.keys())[:6]
                vals = [fi[k] for k in names]
                fi_ax.barh(names, vals, color='#1C1E26')
                fi_ax.set_facecolor('#F7F8FA')
                fi_ax.tick_params(labelsize=7)
                fi_fig.tight_layout(pad=1.0)
                fi_canvas = FigureCanvasQTAgg(fi_fig)
                fi_canvas.setFixedHeight(200)
                cl.addWidget(fi_canvas)
        except Exception:
            pass

        self._right.addWidget(card)
        try:
            acc_card, acc_l = self._card('Validated accuracy')
            for _line in _hs.validated_summary_lines(_report):
                _ql = QLabel(_line)
                _ql.setWordWrap(True)
                _ql.setStyleSheet('font-size:12px;color:#475467;')
                acc_l.addWidget(_ql)
            self._right.addWidget(acc_card)
        except Exception:
            pass
        # Causal chain panel below the ML outputs
        self._right.addWidget(self._chain_widget)
        self._right.addStretch()

    def set_chain(self, steps: list):
        """Populate causal chain panel with CausalChainStep list."""
        self._chain_widget.set_chain(steps)

    def set_bsp_alert(self, alert: dict):
        """Show or update the BSP inflation alert banner and chain alert strip."""
        self._bsp_banner.set_alert(alert)
        self._chain_widget.set_alert(alert)

    def set_regional_estimates(self, estimates: dict) -> None:
        """Push per-region price-change estimates to the map widget."""
        self._map_widget.set_estimates(estimates)

    def set_policy_recos(self, recos: list) -> None:
        """Populate the policy recommendation cards."""
        self._reco_widget.set_recos(recos)

    def _make_features(self, df_feat, feature_cols, scenario):
        """Build a 1-D feature vector from the last row of df_feat, applying scenario shocks."""
        last = df_feat.iloc[-1]
        features = []
        for col in feature_cols:
            if col == 'prev_gas_price':
                features.append(float(last['gas_price']))
            elif col == 'oil_price':
                features.append(float(last[col]) * (1 + scenario.get('oil_pct', 0) / 100))
            elif col == 'usd_php':
                features.append(float(last[col]) * (1 + scenario.get('usd_pct', 0) / 100))
            else:
                features.append(float(last[col]))
        return np.array(features)

    def _on_export(self):
        if not self._responses and not self._consensus.get('weighted_avg'):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, 'Export Report',
            str(Path.home() / 'Downloads' / 'simulation_report.pdf'),
            'PDF Files (*.pdf)'
        )
        if not path:
            return
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [
                Paragraph('PH Economic Pressure Simulation Report', styles['Title']),
                Spacer(1, 12),
            ]
            avg = self._consensus.get('weighted_avg')
            if avg is not None:
                story.append(Paragraph(f'Consensus: +{avg:.2f}/L', styles['Heading2']))
            for resp in self._responses:
                story.append(Paragraph(
                    f"Round {resp.round_num} - {resp.agent_name}: {resp.statement[:500]}",
                    styles['Normal']
                ))
                story.append(Spacer(1, 6))
            doc.build(story)
        except Exception:
            pass

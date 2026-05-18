import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout,
                              QStackedWidget, QLabel, QVBoxLayout)
from PyQt6.QtCore import Qt

from ph_economic_ai.ui.sidebar import SidebarWidget
from ph_economic_ai.ui.dashboard import DashboardPage
from ph_economic_ai.ui.pressure import PressureGaugePage
from ph_economic_ai.ui.agent_graph import AgentGraphPage

from ph_economic_ai.utils.preprocessing import build_features, compute_index, pressure_band
from ph_economic_ai.utils.explanation import generate as generate_explanation
from ph_economic_ai import model as ml


class _SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lyt = QVBoxLayout(self)
        lyt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl = QLabel('Settings — coming soon')
        lbl.setStyleSheet('font-size:16px; color:#AAAAAA;')
        lyt.addWidget(lbl)


class MainWindow(QMainWindow):
    def __init__(self, df, regressor, parent=None):
        super().__init__(parent)
        self._df = df
        self._regressor = regressor
        self._oil_shock_active = False

        X, y, _, self._df_feat = build_features(df)
        self._X = X
        self._last_features = np.array([
            self._df_feat.iloc[-1]['oil_price'],
            self._df_feat.iloc[-1]['usd_php'],
            self._df_feat.iloc[-1]['demand_index'],
            self._df_feat.iloc[-1]['gas_price'],
        ])

        self.setWindowTitle('Philippine Economic Pressure AI')
        self.setMinimumSize(1100, 680)
        self.setStyleSheet('background:#FFFFFF;')

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = SidebarWidget()
        self._sidebar.page_changed.connect(self._on_page_changed)
        root.addWidget(self._sidebar)

        self._stack = QStackedWidget()
        self._dashboard = DashboardPage()
        self._dashboard.recalculate_requested.connect(self._on_recalculate)
        self._dashboard.oil_shock_requested.connect(self._on_oil_shock)
        self._pressure_page = PressureGaugePage()
        self._agent_page = AgentGraphPage()
        self._settings_page = _SettingsPage()

        for page in (self._dashboard, self._pressure_page,
                     self._agent_page, self._settings_page):
            self._stack.addWidget(page)

        root.addWidget(self._stack, stretch=1)

        # Initial render
        self._refresh()

    def _on_page_changed(self, idx: int):
        self._stack.setCurrentIndex(idx)

    def _on_recalculate(self):
        self._oil_shock_active = False
        self._last_features = np.array([
            self._df_feat.iloc[-1]['oil_price'],
            self._df_feat.iloc[-1]['usd_php'],
            self._df_feat.iloc[-1]['demand_index'],
            self._df_feat.iloc[-1]['gas_price'],
        ])
        self._refresh()

    def _on_oil_shock(self):
        self._oil_shock_active = True
        features = self._last_features.copy()
        features[0] *= 1.10  # oil price +10%
        self._last_features = features
        self._refresh()

    def _build_result(self) -> dict:
        predicted_price, confidence, pred_std = ml.predict(
            self._regressor, self._last_features
        )
        current_price = float(self._df_feat.iloc[-1]['gas_price'])

        if predicted_price > current_price + 0.5:
            trend = 'Rising'
        elif predicted_price < current_price - 0.5:
            trend = 'Falling'
        else:
            trend = 'Stable'

        pressure_index, oil_delta, usd_delta, demand_norm = compute_index(
            float(self._last_features[0]),
            float(self._last_features[1]),
            float(self._last_features[2]),
            self._df,
        )
        band = pressure_band(pressure_index)

        explanation = generate_explanation(
            oil_delta, usd_delta, demand_norm,
            pressure_index, current_price, predicted_price,
        )

        scenarios = ml.simulate_scenarios(
            self._regressor, self._last_features, predicted_price
        )

        train_means, train_stds = ml.get_training_predictions(
            self._regressor, self._X
        )

        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'trend': trend,
            'confidence': confidence,
            'pressure_index': pressure_index,
            'pressure_band': band,
            'oil_delta': oil_delta,
            'usd_delta': usd_delta,
            'demand_norm': demand_norm,
            'explanation': explanation,
            'scenarios': scenarios,
            'pred_std': pred_std,
            'train_means': train_means,
            'train_stds': train_stds,
            'df': self._df_feat,
        }

    def _refresh(self):
        result = self._build_result()
        self._dashboard.refresh(result)
        self._pressure_page.refresh(result)

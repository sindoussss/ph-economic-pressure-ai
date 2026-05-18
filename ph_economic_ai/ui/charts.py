import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PriceChart(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure(figsize=(5, 2.6), dpi=96)
        self.figure.patch.set_facecolor('#FFFFFF')
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self._style_axes()

    def _style_axes(self):
        self.ax.set_facecolor('#FAFAFA')
        for spine in ('top', 'right'):
            self.ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            self.ax.spines[spine].set_color('#EAEAEA')
        self.ax.tick_params(colors='#AAAAAA', labelsize=8)
        self.ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f'₱{x:.0f}')
        )
        self.ax.grid(axis='y', color='#F0F0F0', linewidth=0.6, zorder=0)
        self.figure.tight_layout(pad=1.2)

    def update_data(self, dates: list, actuals: np.ndarray, train_means: np.ndarray,
                    train_stds: np.ndarray, predicted_price: float, pred_std: float):
        self.ax.clear()
        self._style_axes()

        n = len(actuals)
        x_actual = list(range(n))
        x_pred = list(range(n)) + [n]
        pred_line = list(train_means) + [predicted_price]
        upper = list(train_means + train_stds) + [predicted_price + pred_std]
        lower = list(train_means - train_stds) + [predicted_price - pred_std]

        # Confidence band
        self.ax.fill_between(x_pred, lower, upper, color='#C8DEF5', alpha=0.45, zorder=1)

        # Actual line
        self.ax.plot(x_actual, actuals, color='#999999', linewidth=1.8,
                     label='Actual', zorder=2)

        # Predicted line
        self.ax.plot(x_pred, pred_line, color='#4A90E2', linewidth=2.2,
                     label='Predicted', zorder=3)

        # Forecast divider
        self.ax.axvline(x=n - 1, color='#DDDDDD', linewidth=1.0,
                        linestyle='--', zorder=2)
        self.ax.text(n - 0.5, self.ax.get_ylim()[1], '→ Forecast',
                     fontsize=7, color='#BBBBBB', va='top')

        # Endpoint markers
        self.ax.plot(n - 1, actuals[-1], 'o', color='#FFFFFF',
                     markeredgecolor='#999999', markersize=5, zorder=4)
        self.ax.plot(n, predicted_price, 'o', color='#4A90E2', markersize=5, zorder=4)

        # X-axis labels (every 12 months)
        tick_pos = list(range(0, n, 12))
        tick_labels = [dates[i] for i in tick_pos if i < len(dates)]
        self.ax.set_xticks(tick_pos[:len(tick_labels)])
        self.ax.set_xticklabels(tick_labels, fontsize=8, color='#AAAAAA')

        # Legend
        self.ax.legend(fontsize=8, loc='upper left',
                       framealpha=0.7, edgecolor='#EAEAEA')

        self.figure.tight_layout(pad=1.2)
        self.draw()

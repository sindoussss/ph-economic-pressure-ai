import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from ph_economic_ai.data import generate_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # ── Startup pipeline ──────────────────────────────────────────────────────
    df = generate_dataset()
    X, y, _, _ = build_features(df)
    regressor = ml.train(X, y)

    # ── Launch ────────────────────────────────────────────────────────────────
    window = MainWindow(df=df, regressor=regressor)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

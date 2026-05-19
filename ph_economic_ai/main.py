import sys
from PyQt6.QtWidgets import QApplication, QMessageBox

from ph_economic_ai.data import fetch_dataset
from ph_economic_ai.utils.preprocessing import build_features
from ph_economic_ai import model as ml
from ph_economic_ai.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        df, data_source = fetch_dataset()
    except RuntimeError as e:
        QMessageBox.critical(None, 'Data Error', str(e))
        sys.exit(1)

    X, y, _, _ = build_features(df)
    regressor = ml.train(X, y)

    window = MainWindow(df=df, regressor=regressor, data_source=data_source)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

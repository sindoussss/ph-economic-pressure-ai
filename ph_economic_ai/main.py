import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMessageBox

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ph_economic_ai import model as ml
from ph_economic_ai.data import fetch_dataset
from ph_economic_ai.engine.store import AgentTrustStore
from ph_economic_ai.ui.main_window import SimMainWindow
from ph_economic_ai.utils.preprocessing import (
    build_electricity_features,
    build_features,
    build_food_features,
)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    try:
        df, data_source = fetch_dataset()
    except RuntimeError as e:
        QMessageBox.critical(None, 'Data Error', str(e))
        sys.exit(1)

    # Gas model.
    X_gas, y_gas, _, _ = build_features(df)
    gas_regressor = ml.train(X_gas, y_gas)
    cv_rmse = ml.cross_val_rmse(X_gas, y_gas)

    # Food and electricity models use observed historical gas prices as the
    # training proxy for gas predictions.
    gas_pred_train = df['gas_price'].values
    regressors: dict = {'gas': gas_regressor}

    try:
        X_food, y_food, _, _ = build_food_features(df, gas_pred_train)
        if len(X_food) > 0:
            regressors['food'] = ml.train_sector(X_food, y_food)
    except (ValueError, KeyError):
        pass

    try:
        X_elec, y_elec, _, _ = build_electricity_features(df, gas_pred_train)
        if len(X_elec) > 0:
            regressors['electricity'] = ml.train_sector(X_elec, y_elec)
    except (ValueError, KeyError):
        pass

    store = AgentTrustStore()

    window = SimMainWindow(
        df=df,
        regressor=gas_regressor,
        regressors=regressors,
        data_source=data_source,
        cv_rmse=cv_rmse,
        store=store,
    )
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

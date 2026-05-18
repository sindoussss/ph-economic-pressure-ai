import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    """Train on first 80% of rows (time-ordered). Returns fitted regressor."""
    split = int(len(X) * 0.8)
    regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    regressor.fit(X[:split], y[:split])
    return regressor


def predict(regressor: RandomForestRegressor, last_features: np.ndarray) -> tuple:
    """
    Predict next price from a 1-D feature vector.
    Returns (predicted_price, confidence_0_100, pred_std).
    """
    X = last_features.reshape(1, -1)
    tree_preds = np.array([t.predict(X)[0] for t in regressor.estimators_])
    predicted_price = float(tree_preds.mean())
    pred_std = float(tree_preds.std())
    confidence = float(np.clip(100.0 - (pred_std / max(predicted_price, 1.0) * 100.0), 0.0, 100.0))
    return predicted_price, confidence, pred_std


def get_training_predictions(regressor: RandomForestRegressor, X: np.ndarray) -> tuple:
    """Return (means, stds) arrays over all training rows — used for chart confidence band."""
    tree_matrix = np.array([t.predict(X) for t in regressor.estimators_])
    return tree_matrix.mean(axis=0), tree_matrix.std(axis=0)


def simulate_scenarios(regressor: RandomForestRegressor, last_features: np.ndarray,
                       baseline_price: float) -> dict:
    """
    Perturb last_features for each scenario and return price deltas vs baseline.
    last_features layout: [oil_price, usd_php, demand_index, prev_gas_price]
    """
    def _delta(features):
        p, _, _ = predict(regressor, features)
        return p - baseline_price

    oil_f = last_features.copy(); oil_f[0] *= 1.05
    usd_f = last_features.copy(); usd_f[1] *= 1.02
    dem_f = last_features.copy(); dem_f[2] = max(0.0, dem_f[2] - 10.0)

    return {
        'oil_shock': _delta(oil_f),
        'usd_shock': _delta(usd_f),
        'demand_drop': _delta(dem_f),
    }

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score


def _compute_feature_importances(regressor, n_features: int) -> np.ndarray:
    """Gain-based feature importances from HGB internal tree nodes, normalized to sum=1."""
    importances = np.zeros(n_features)
    try:
        for tree_list in regressor._predictors:
            for tree in tree_list:
                for node in tree.nodes:
                    if not node['is_leaf']:
                        feat_idx = int(node['feature_idx'])
                        if 0 <= feat_idx < n_features:
                            importances[feat_idx] += node['gain']
    except AttributeError:
        return importances  # graceful fallback if sklearn internals change
    total = importances.sum()
    if total > 0:
        importances /= total
    return importances


def train(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    """Train on all rows (time-ordered). Returns fitted regressor."""
    regressor = HistGradientBoostingRegressor(
        random_state=42, min_samples_leaf=5, max_leaf_nodes=15
    )
    regressor.fit(X, y)
    # Attach gain-based feature importances (HGB doesn't expose them natively)
    regressor.feature_importances_ = _compute_feature_importances(regressor, X.shape[1])
    return regressor


def train_sector(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    """Train a sector-specific model. Identical to train(); exists for naming clarity."""
    return train(X, y)


def predict_interval(regressor, last_features: np.ndarray, qhat: float) -> tuple:
    """Predict next price with a conformal band. Returns (point, low, high).

    qhat is the conformal half-width for the desired level (see
    benchmark.conformal / accuracy_report.json). qhat=0 -> degenerate band.
    """
    point = float(regressor.predict(last_features.reshape(1, -1))[0])
    return point, point - qhat, point + qhat


def load_conformal_widths(report_path=None) -> dict:
    """Load {level_str: qhat} from the frozen accuracy_report.json.

    Returns {} if the report has not been generated yet (caller decides UX).
    """
    from ph_economic_ai.benchmark.report import load_report, REPORT_PATH
    path = report_path or REPORT_PATH
    try:
        return load_report(path).get('conformal_widths', {})
    except FileNotFoundError:
        return {}


def predict(regressor, last_features: np.ndarray) -> tuple:
    """Backward-compatible point forecast with a REAL 90% band derived from the
    frozen conformal report. Returns (predicted_price, low_90, high_90).

    Replaces the former hardcoded (price, 90.0, 0.0). If no report exists yet,
    the band collapses to the point (low==high==point) and callers should show
    'uncalibrated' until the benchmark has been run.
    """
    widths = load_conformal_widths()
    qhat90 = float(widths.get('0.9', 0.0))
    return predict_interval(regressor, last_features, qhat90)


def get_training_predictions(regressor: HistGradientBoostingRegressor, X: np.ndarray) -> tuple:
    """Return (means, stds) for all rows. stds are zeros — HGB has no per-tree variance."""
    means = regressor.predict(X)
    stds = np.zeros(len(X))
    return means, stds


def get_feature_importances(model: HistGradientBoostingRegressor,
                            feature_names: list) -> dict:
    """Return feature name → importance (0–1), sorted descending. Sums to 1."""
    importances = model.feature_importances_
    return dict(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True))


def cross_val_rmse(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """Walk-forward CV on a fresh HGB. Returns mean RMSE (positive, PHP/liter)."""
    from sklearn.model_selection import TimeSeriesSplit
    model = HistGradientBoostingRegressor(
        random_state=42, min_samples_leaf=5, max_leaf_nodes=15
    )
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=tscv)
    return float(-scores.mean())


def forecast(regressor: HistGradientBoostingRegressor, last_features: np.ndarray,
             n_months: int = 6) -> np.ndarray:
    """
    Roll n_months forward from last_features using flat projection.
    The last element of last_features is prev_gas_price — updated each step.
    Returns array of shape (n_months,) with predicted prices.
    """
    prices = []
    features = last_features.copy()
    for _ in range(n_months):
        price, _, _ = predict(regressor, features)
        prices.append(price)
        features[-1] = price  # prev_gas_price is always last (see build_features)
    return np.array(prices)


def simulate_scenarios(regressor: HistGradientBoostingRegressor,
                       last_features: np.ndarray, baseline_price: float) -> dict:
    """
    Perturb last_features for 3 scenarios and return price deltas vs baseline.
    Feature layout: [oil_price(0), usd_php(1), demand_index(2), ..., prev_gas_price(-1)]
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

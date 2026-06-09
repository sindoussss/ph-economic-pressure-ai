"""Validate the RBOB-derived gas proxy against the World Bank gold series.

Answers the 'is your data real?' objection: high correlation + small bias means
the proxy tracks reality and is acceptable as a live input feature.
"""
import numpy as np
import pandas as pd


def proxy_vs_gold(proxy: pd.Series, gold: pd.Series) -> dict:
    """Align proxy and gold on shared 'YYYY-MM' dates; report r, bias, MAE, n."""
    joined = pd.concat([proxy.rename('proxy'), gold.rename('gold')], axis=1).dropna()
    n = len(joined)
    if n < 2:
        return {'pearson_r': float('nan'), 'bias_mean': float('nan'),
                'mae': float('nan'), 'n': n}
    p = joined['proxy'].to_numpy()
    g = joined['gold'].to_numpy()
    r = float(np.corrcoef(p, g)[0, 1])
    return {
        'pearson_r': round(r, 4),
        'bias_mean': round(float(np.mean(p - g)), 4),
        'mae': round(float(np.mean(np.abs(p - g))), 4),
        'n': n,
    }

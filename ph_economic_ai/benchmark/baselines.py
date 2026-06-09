"""Naive next-step forecasters. Each takes the training history (1-D array of
prices up to and including time t) and returns the prediction for t+1."""
import numpy as np


def random_walk_next(history: np.ndarray) -> float:
    """No-change forecast: next = last observed value."""
    return float(history[-1])


def seasonal_naive_next(history: np.ndarray, season: int = 12) -> float:
    """Next = value one full season ago. Falls back to random walk if history
    is shorter than the season."""
    if len(history) > season:
        return float(history[-season])
    return random_walk_next(history)

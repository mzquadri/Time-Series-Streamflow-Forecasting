"""Baselines, and the reason this series needs careful ones.

The generated streamflow is close to a random walk. Its lag-one autocorrelation is
0.9987, because the generator carries 85% of yesterday's value into today and the
level drifts upward throughout the record. Two consequences follow, and both
decide how a score should be read.

Predicting yesterday's value is very strong. Any model that cannot beat it has
learned nothing useful about this series, however high its R2 looks.

A day-of-year average is very weak, and not because seasonality is unimportant.
The level rises from about 70 in the first year to about 236 in the last, so an
average over the training years is biased far below the test years. Reporting that
failure as evidence that a model is good would be a strawman, so the same
baseline is also reported with the training trend extrapolated, which is the fair
version of it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE, R2 and MAPE. R2 and NSE are the same quantity."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1.0 - float((err**2).sum()) / ss_tot if ss_tot else float("nan")
    nonzero = actual != 0
    mape = (float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100)
            if nonzero.any() else float("nan"))
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": r2,
        "nse": r2,
        "mape_percent": mape,
        "n": len(actual),
    }


def persistence(series: np.ndarray, test_index: np.ndarray) -> np.ndarray:
    """Yesterday's value. The baseline that matters on an autocorrelated series."""
    return series[test_index - 1]


def drift(series: np.ndarray, train_end: int, test_index: np.ndarray) -> np.ndarray:
    """Yesterday's value plus the average daily change seen in training.

    A random walk with drift. It costs one number more than persistence and is the
    honest thing to compare against when the level is trending.
    """
    step = float(np.diff(series[:train_end]).mean())
    return series[test_index - 1] + step


def climatology(day_of_year: np.ndarray, series: np.ndarray, train_end: int,
                test_index: np.ndarray) -> np.ndarray:
    """Mean value for each calendar day, taken over the training years."""
    table = pd.Series(series[:train_end]).groupby(day_of_year[:train_end]).mean()
    overall = float(series[:train_end].mean())
    return pd.Series(day_of_year[test_index]).map(table).fillna(overall).to_numpy()


def climatology_with_trend(day_of_year: np.ndarray, series: np.ndarray,
                           train_end: int, test_index: np.ndarray) -> np.ndarray:
    """Day-of-year mean after removing, then re-adding, a fitted linear trend.

    The plain day-of-year mean fails badly here purely because the level drifts.
    Removing the trend before averaging and adding it back at forecast time
    separates "seasonality does not help" from "the level moved", which are very
    different conclusions.
    """
    t_train = np.arange(train_end, dtype=float)
    slope, intercept = np.polyfit(t_train, series[:train_end], 1)
    detrended = series[:train_end] - (slope * t_train + intercept)

    table = pd.Series(detrended).groupby(day_of_year[:train_end]).mean()
    seasonal = pd.Series(day_of_year[test_index]).map(table).fillna(0.0).to_numpy()
    return seasonal + slope * test_index.astype(float) + intercept

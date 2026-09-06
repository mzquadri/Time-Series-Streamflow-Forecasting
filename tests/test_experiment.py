"""Tests for the parts that decide whether a reported score means anything.

Two of these exist because of defects that were present. The comparison put a
monthly R2 beside a daily one and declared a winner, and the only baseline offered
was one guaranteed to fail on a trending series.

The rest guard what a forecasting benchmark has to hold: no feature sees its own
target or anything later, the split is chronological, and every method in a table
is scored on identical rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines import (  # noqa: E402
    climatology,
    climatology_with_trend,
    drift,
    metrics,
    persistence,
)
from experiment import TARGET, lag_features  # noqa: E402
from generate_data import generate_streamflow  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return generate_streamflow(n_years=6, seed=42)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------

def test_the_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate_streamflow(n_years=4, seed=42),
                                  generate_streamflow(n_years=4, seed=42))


def test_a_different_seed_gives_different_data():
    a = generate_streamflow(n_years=4, seed=42)
    b = generate_streamflow(n_years=4, seed=7)

    assert not a[TARGET].equals(b[TARGET])


def test_the_series_is_strongly_autocorrelated(data):
    """The property that makes persistence the baseline worth beating."""
    y = data[TARGET].to_numpy()

    assert np.corrcoef(y[:-1], y[1:])[0, 1] > 0.95


def test_the_level_drifts_upward(data):
    """The property that makes a plain day-of-year mean fail."""
    y = data[TARGET].to_numpy()

    assert y[-365:].mean() > y[:365].mean() * 1.2


def test_streamflow_never_goes_below_the_floor(data):
    assert (data[TARGET] >= 5.0).all()


# --------------------------------------------------------------------------
# features: no row may see its own target or the future
# --------------------------------------------------------------------------

def test_no_feature_leaks_the_target_or_the_future(data):
    feat = lag_features(data)
    target = feat[TARGET].to_numpy()

    for column in feat.columns:
        if column in ("date", TARGET):
            continue
        values = feat[column].to_numpy(dtype=float)
        # A feature equal to its own row's target would be leakage. Correlation
        # is allowed to be high, since lag1 nearly is the target, but never exact.
        assert not np.allclose(values, target), f"{column} equals the target"


def test_lag1_is_yesterdays_value_not_todays(data):
    feat = lag_features(data)
    merged = data.set_index("date")[TARGET]

    row = feat.iloc[100]
    yesterday = merged.loc[row["date"] - pd.Timedelta(days=1)]
    assert row["lag1"] == pytest.approx(yesterday)


def test_rolling_features_exclude_the_current_day(data):
    feat = lag_features(data)
    series = data.set_index("date")[TARGET]

    row = feat.iloc[200]
    window_end = row["date"] - pd.Timedelta(days=1)
    expected = series.loc[:window_end].iloc[-7:].mean()
    assert row["roll_mean_7"] == pytest.approx(expected)


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------

def test_persistence_returns_the_previous_value():
    y = np.arange(50, dtype=float)
    index = np.array([30, 31, 32])

    assert np.allclose(persistence(y, index), [29.0, 30.0, 31.0])


def test_drift_adds_the_average_training_step():
    y = np.arange(0, 100, 2, dtype=float)  # a constant step of 2
    index = np.array([40, 41])

    assert np.allclose(drift(y, 30, index), y[index - 1] + 2.0)


def test_climatology_averages_the_calendar_day():
    doy = np.tile(np.arange(1, 366), 3)
    y = np.tile(np.arange(1, 366, dtype=float), 3)

    assert np.allclose(climatology(doy, y, 730, np.array([730, 731])), [1.0, 2.0])


def test_climatology_with_trend_recovers_a_pure_trend():
    """On a straight line with no seasonality, it should track the line."""
    n = 800
    doy = (np.arange(n) % 365) + 1
    y = 10.0 + 0.5 * np.arange(n, dtype=float)
    index = np.array([700, 750])

    predicted = climatology_with_trend(doy, y, 600, index)
    assert np.allclose(predicted, y[index], atol=1e-6)


def test_plain_climatology_fails_on_that_same_trend():
    """The contrast the benchmark reports, shown directly."""
    n = 800
    doy = (np.arange(n) % 365) + 1
    y = 10.0 + 0.5 * np.arange(n, dtype=float)
    index = np.array([700, 750])

    plain = climatology(doy, y, 600, index)
    assert np.abs(plain - y[index]).mean() > 50


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def test_a_perfect_prediction_scores_perfectly():
    y = np.array([3.0, 9.0, 4.0])
    m = metrics(y, y)

    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_predicting_the_mean_scores_zero_r2():
    y = np.array([3.0, 9.0, 4.0])

    assert metrics(y, np.full_like(y, y.mean()))["r2"] == pytest.approx(0.0)


def test_r2_and_nse_are_the_same_number():
    y = np.array([3.0, 9.0, 4.0])
    m = metrics(y, np.array([3.5, 8.0, 4.5]))

    assert m["r2"] == m["nse"]


def test_metrics_record_how_many_rows_were_scored():
    """A table is only comparable if every row covers the same days."""
    y = np.arange(20, dtype=float)

    assert metrics(y, y)["n"] == 20

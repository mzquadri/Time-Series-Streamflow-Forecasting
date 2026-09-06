"""Run the comparison honestly: same rows, same granularity, real baselines.

    python src/experiment.py

The previous version put a monthly SARIMAX and a daily XGBoost in one table and
declared a winner. Those are different targets with different variance, so the
two R2 values were never comparable. Daily and monthly are now reported as
separate tables, each with its own baselines, and no claim is made across them.

Writes results/benchmark.json.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from baselines import (
    climatology,
    climatology_with_trend,
    drift,
    metrics,
    persistence,
)
from generate_data import generate_streamflow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "benchmark.json")

TARGET = "streamflow_m3s"
TEST_YEARS = 2
#: Extra generated series used only to check the finding is not seed-specific.
SEEDS = (42, 7, 123, 2024, 99)
LAGS = [1, 2, 3, 7, 14, 30, 365]
WINDOWS = [7, 14, 30]


def lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar, lagged target and lagged precipitation features.

    Every feature is shifted by at least one day, and rolling statistics are taken
    after a shift, so no row can see its own target or anything later.
    """
    out = df.copy()
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365)
    out["month_sin"] = np.sin(2 * np.pi * out["date"].dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["date"].dt.month / 12)
    # A plain time index, so a tree model can express the drift at all. Without it
    # the trees can only extrapolate flat, which on a trending series is fatal.
    out["t"] = np.arange(len(out), dtype=float)

    for lag in LAGS:
        out[f"lag{lag}"] = out[TARGET].shift(lag)
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = out[TARGET].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[TARGET].shift(1).rolling(w).std()
    for lag in [1, 2, 3, 7]:
        out[f"precip_lag{lag}"] = out["precipitation_mm"].shift(lag)
    out["precip_roll_7"] = out["precipitation_mm"].shift(1).rolling(7).sum()
    return out.dropna().reset_index(drop=True)


def main() -> int:
    df = generate_streamflow(n_years=15, seed=42)
    y = df[TARGET].to_numpy()
    doy = df["date"].dt.dayofyear.to_numpy()
    split_date = df["date"].max() - pd.Timedelta(days=365 * TEST_YEARS)
    is_test = (df["date"] > split_date).to_numpy()
    test_index = np.flatnonzero(is_test)
    train_end = int(test_index[0])

    lag1 = float(np.corrcoef(y[:-1], y[1:])[0, 1])
    print(f"  {len(df)} days, train {train_end}, test {len(test_index)}")
    print(f"  lag-1 autocorrelation {lag1:.4f}")
    print(f"  level: first year {y[:365].mean():.1f}, last year {y[-365:].mean():.1f}\n")

    # Models that read the engineered features. Scored on the same daily rows as
    # the baselines above, minus the warm-up the longest lag needs.
    feat = lag_features(df)
    feat_test = (feat["date"] > split_date).to_numpy()
    columns = [c for c in feat.columns if c not in ("date", TARGET)]
    x_train = feat.loc[~feat_test, columns]
    y_train = feat.loc[~feat_test, TARGET]
    x_test = feat.loc[feat_test, columns]
    y_test = feat.loc[feat_test, TARGET].to_numpy()

    models = {
        "ridge_on_features": Ridge(alpha=1.0).fit(x_train, y_train).predict(x_test),
        # tree_method="exact" because the default histogram builder gives a
        # different model for a different thread count, so the published RMSE
        # depended on the machine that produced it. Exact is thread-independent.
        "xgboost": XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, n_jobs=1, tree_method="exact",
        ).fit(x_train, y_train).predict(x_test),
    }
    # The baselines are recomputed on the model's rows so every daily number in
    # the table is scored on an identical set of days.
    model_index = test_index[-len(y_test):]
    daily_on_model_rows = {
        "persistence": persistence(y, model_index),
        "random_walk_with_drift": drift(y, train_end, model_index),
        "day_of_year_mean": climatology(doy, y, train_end, model_index),
        "day_of_year_mean_plus_trend": climatology_with_trend(doy, y, train_end, model_index),
    }
    assert np.allclose(y[model_index], y_test), "baseline and model rows disagree"

    daily_scores = {k: metrics(y_test, v) for k, v in daily_on_model_rows.items()}
    daily_scores.update({k: metrics(y_test, v) for k, v in models.items()})

    print("  daily, all scored on the same rows, ranked by RMSE:")
    for name in sorted(daily_scores, key=lambda k: daily_scores[k]["rmse"]):
        m = daily_scores[name]
        print(f"    {name:30} RMSE {m['rmse']:8.3f}  MAE {m['mae']:7.3f}  R2 {m['r2']:9.4f}")

    # Monthly, kept separate. A monthly R2 is not comparable with a daily one.
    monthly = df.set_index("date")[TARGET].resample("MS").mean()
    m_train = monthly[monthly.index <= split_date]
    m_test = monthly[monthly.index > split_date]

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit = SARIMAX(m_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                  enforce_stationarity=False, enforce_invertibility=False).fit(
                      disp=False, maxiter=200)
    sarimax_pred = fit.forecast(steps=len(m_test)).clip(lower=5).to_numpy()
    m_actual = m_test.to_numpy()
    m_index = np.arange(len(monthly))[monthly.index > split_date]
    m_values = monthly.to_numpy()
    m_train_end = int(m_index[0])

    monthly_scores = {
        "sarimax": metrics(m_actual, sarimax_pred),
        "persistence": metrics(m_actual, m_values[m_index - 1]),
        "random_walk_with_drift": metrics(
            m_actual, m_values[m_index - 1] + float(np.diff(m_values[:m_train_end]).mean())),
    }
    print(f"\n  monthly ({len(m_actual)} points), a separate problem from the daily table:")
    for name in sorted(monthly_scores, key=lambda k: monthly_scores[k]["rmse"]):
        m = monthly_scores[name]
        print(f"    {name:30} RMSE {m['rmse']:8.3f}  MAE {m['mae']:7.3f}  R2 {m['r2']:9.4f}")

    # Why a tree does badly here: the target is almost yesterday's value, so a
    # method's error is dominated by how closely it can reproduce lag1. A tree
    # approximates that identity with piecewise constants.
    lag1_test = feat.loc[feat_test, "lag1"].to_numpy()
    deviation = {
        "persistence": 0.0,
        **{name: float(np.abs(pred - lag1_test).mean()) for name, pred in models.items()},
    }
    daily_change_std = float(np.diff(y).std())

    # One seed is one measurement. The central claim is repeated across several
    # generated series so it is not a property of seed 42.
    print("\n  repeating the comparison on other seeds:")
    robustness = []
    for other in SEEDS:
        alt = generate_streamflow(n_years=15, seed=other)
        alt_y = alt[TARGET].to_numpy()
        alt_split = alt["date"].max() - pd.Timedelta(days=365 * TEST_YEARS)
        alt_feat = lag_features(alt)
        alt_te = (alt_feat["date"] > alt_split).to_numpy()
        alt_cols = [c for c in alt_feat.columns if c not in ("date", TARGET)]
        alt_ytest = alt_feat.loc[alt_te, TARGET].to_numpy()
        alt_index = np.flatnonzero(
            (alt["date"] > alt_split).to_numpy())[-len(alt_ytest):]
        scores = {
            "persistence": metrics(alt_ytest, persistence(alt_y, alt_index))["rmse"],
            "ridge_on_features": metrics(alt_ytest, Ridge(alpha=1.0).fit(
                alt_feat.loc[~alt_te, alt_cols], alt_feat.loc[~alt_te, TARGET]).predict(
                    alt_feat.loc[alt_te, alt_cols]))["rmse"],
            "xgboost": metrics(alt_ytest, XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, random_state=42, n_jobs=1, tree_method="exact",
            ).fit(alt_feat.loc[~alt_te, alt_cols],
                  alt_feat.loc[~alt_te, TARGET]).predict(
                      alt_feat.loc[alt_te, alt_cols]))["rmse"],
        }
        scores["xgboost_beats_persistence"] = scores["xgboost"] < scores["persistence"]
        scores["seed"] = other
        robustness.append({k: (round(v, 3) if isinstance(v, float) else v)
                           for k, v in scores.items()})
        print(f"    seed {other:<5} persistence {scores['persistence']:6.3f}  "
              f"xgboost {scores['xgboost']:6.3f}  ridge {scores['ridge_on_features']:6.3f}")
    n_beat = sum(r["xgboost_beats_persistence"] for r in robustness)
    print(f"    XGBoost beats persistence on {n_beat} of {len(robustness)} seeds")

    best_daily = min(daily_scores, key=lambda k: daily_scores[k]["rmse"])
    payload = {
        "environment": {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "data": {
            "days": len(df), "years": 15, "seed": 42, "target": TARGET,
            "lag1_autocorrelation": round(lag1, 4),
            "first_year_mean": round(float(y[:365].mean()), 2),
            "last_year_mean": round(float(y[-365:].mean()), 2),
        },
        "split": {
            "strategy": "chronological", "test_years": TEST_YEARS,
            "train_days": train_end, "test_days": len(test_index),
            "scored_daily_rows": len(y_test),
            "note": "models lose the first 365 rows to the longest lag; "
                    "baselines are rescored on the same rows",
        },
        "daily": {k: {m: round(v, 4) for m, v in s.items()} for k, s in daily_scores.items()},
        "monthly": {k: {m: round(v, 4) for m, v in s.items()} for k, s in monthly_scores.items()},
        "mechanism": {
            "daily_change_std": round(daily_change_std, 3),
            "mean_absolute_deviation_from_lag1": {k: round(v, 3) for k, v in deviation.items()},
            "note": "the useful daily correction is about the size of daily_change_std; "
                    "a method whose deviation from lag1 exceeds it adds more error "
                    "than signal",
        },
        "robustness": {
            "seeds": robustness,
            "xgboost_beats_persistence_count": n_beat,
            "seeds_tested": len(robustness),
        },
        "best_daily": best_daily,
        "xgboost_beats_persistence": bool(
            daily_scores["xgboost"]["rmse"] < daily_scores["persistence"]["rmse"]),
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\n  best daily method: {best_daily}")
    print(f"  wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

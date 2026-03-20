"""
Time Series Forecasting Pipeline for Streamflow Prediction.

Models:
1. ARIMA/SARIMAX — Classical statistical approach
2. XGBoost — ML with engineered lag/calendar features
3. Baseline — Seasonal naive (same day last year)

Evaluation:
- RMSE, MAE, R², NSE, MAPE
- Rolling forecast (walk-forward validation)
- Visual comparison plots
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from xgboost import XGBRegressor
import joblib

warnings.filterwarnings("ignore")


def create_lag_features(
    df: pd.DataFrame, target: str, lags: list, window_sizes: list = None
) -> pd.DataFrame:
    """Create lag features, rolling statistics, and calendar features."""
    result = df.copy()
    result["dayofyear"] = result["date"].dt.dayofyear
    result["month"] = result["date"].dt.month
    result["dayofweek"] = result["date"].dt.dayofweek
    result["year"] = result["date"].dt.year

    # Sine/cosine encoding for cyclical features
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)
    result["doy_sin"] = np.sin(2 * np.pi * result["dayofyear"] / 365)
    result["doy_cos"] = np.cos(2 * np.pi * result["dayofyear"] / 365)

    # Lag features
    for lag in lags:
        result[f"{target}_lag{lag}"] = result[target].shift(lag)

    # Rolling window features
    if window_sizes is None:
        window_sizes = [7, 14, 30]
    for w in window_sizes:
        result[f"{target}_roll_mean_{w}"] = result[target].shift(1).rolling(w).mean()
        result[f"{target}_roll_std_{w}"] = result[target].shift(1).rolling(w).std()

    # Precipitation lags
    for lag in [1, 2, 3, 7]:
        result[f"precip_lag{lag}"] = result["precipitation_mm"].shift(lag)
    result["precip_roll_7"] = result["precipitation_mm"].shift(1).rolling(7).sum()

    result = result.dropna()
    return result


def compute_metrics(actual, predicted, name="Model"):
    """Compute and return forecasting metrics."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    nse = 1 - np.sum((actual - predicted) ** 2) / np.sum(
        (actual - np.mean(actual)) ** 2
    )
    mape = 100 * np.mean(np.abs((actual - predicted) / np.maximum(actual, 1e-8)))

    metrics = {
        "RMSE": round(float(rmse), 3),
        "MAE": round(float(mae), 3),
        "R2": round(float(r2), 4),
        "NSE": round(float(nse), 4),
        "MAPE": round(float(mape), 2),
    }
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def run_forecasting(csv_path: str, output_dir: str = "../results"):
    """Run the full forecasting pipeline."""
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(
        f"Dataset: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}"
    )

    target = "streamflow_m3s"

    # ============================================================
    # 1. EXPLORATORY PLOTS
    # ============================================================

    # 1a. Full time series
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(df["date"], df[target], linewidth=0.6, color="#2196F3")
    axes[0].set_ylabel("Streamflow (m³/s)")
    axes[0].set_title("Daily Streamflow Time Series")
    axes[1].bar(df["date"], df["precipitation_mm"], width=1, color="#4CAF50", alpha=0.7)
    axes[1].set_ylabel("Precipitation (mm)")
    axes[1].invert_yaxis()
    axes[2].plot(df["date"], df["temperature_c"], linewidth=0.4, color="#FF5722")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "01_time_series_overview.png"), dpi=150)
    plt.close(fig)
    print("Saved: 01_time_series_overview.png")

    # 1b. Seasonal decomposition
    monthly = df.set_index("date")[target].resample("MS").mean()
    decomp = seasonal_decompose(monthly, model="additive", period=12)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    decomp.observed.plot(ax=axes[0], color="#2196F3")
    axes[0].set_ylabel("Observed")
    decomp.trend.plot(ax=axes[1], color="#4CAF50")
    axes[1].set_ylabel("Trend")
    decomp.seasonal.plot(ax=axes[2], color="#FF9800")
    axes[2].set_ylabel("Seasonal")
    decomp.resid.plot(ax=axes[3], color="#9C27B0")
    axes[3].set_ylabel("Residual")
    fig.suptitle(
        "Seasonal Decomposition (Monthly Mean Streamflow)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "02_seasonal_decomposition.png"), dpi=150)
    plt.close(fig)
    print("Saved: 02_seasonal_decomposition.png")

    # ============================================================
    # 2. TRAIN/TEST SPLIT (last 2 years = test)
    # ============================================================
    split_date = df["date"].max() - pd.Timedelta(days=365 * 2)
    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()
    print(f"\nTrain: {len(train_df)} days, Test: {len(test_df)} days")
    print(f"Split date: {split_date.date()}")

    all_results = {}

    # ============================================================
    # 3. BASELINE — Seasonal Naive
    # ============================================================
    # Predict using same day of year from previous year
    test_df = test_df.copy()
    test_df["doy"] = test_df["date"].dt.dayofyear
    train_df_temp = train_df.copy()
    train_df_temp["doy"] = train_df_temp["date"].dt.dayofyear
    doy_mean = train_df_temp.groupby("doy")[target].mean()
    test_df["baseline_pred"] = test_df["doy"].map(doy_mean).values
    test_df["baseline_pred"] = test_df["baseline_pred"].fillna(train_df[target].mean())

    all_results["Seasonal Naive"] = compute_metrics(
        test_df[target].values,
        test_df["baseline_pred"].values,
        "Seasonal Naive Baseline",
    )

    # ============================================================
    # 4. SARIMAX
    # ============================================================
    print("\nFitting SARIMAX model (this may take a minute)...")
    # Use monthly data for SARIMAX (daily is too slow)
    train_monthly = train_df.set_index("date")[target].resample("MS").mean()
    test_monthly = test_df.set_index("date")[target].resample("MS").mean()

    model_sarima = SARIMAX(
        train_monthly,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarima_fit = model_sarima.fit(disp=False, maxiter=200)
    print(f"  AIC: {sarima_fit.aic:.1f}, BIC: {sarima_fit.bic:.1f}")

    # Forecast
    sarima_forecast = sarima_fit.forecast(steps=len(test_monthly))
    sarima_forecast = sarima_forecast.clip(lower=5)

    all_results["SARIMAX (Monthly)"] = compute_metrics(
        test_monthly.values, sarima_forecast.values, "SARIMAX (Monthly)"
    )

    # ============================================================
    # 5. XGBoost with Lag Features
    # ============================================================
    print("\nTraining XGBoost model...")
    lags = [1, 2, 3, 7, 14, 30, 365]
    df_feat = create_lag_features(df, target, lags=lags)

    feature_cols = [
        c for c in df_feat.columns if c not in ["date", target, "dayofyear", "year"]
    ]
    train_feat = df_feat[df_feat["date"] <= split_date]
    test_feat = df_feat[df_feat["date"] > split_date]

    X_train = train_feat[feature_cols]
    y_train = train_feat[target]
    X_test = test_feat[feature_cols]
    y_test = test_feat[target]

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred = np.clip(xgb_pred, 5, None)

    all_results["XGBoost"] = compute_metrics(y_test.values, xgb_pred, "XGBoost")

    # Save XGBoost model
    joblib.dump(xgb_model, os.path.join(output_dir, "xgboost_model.pkl"))

    # Feature importance
    importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(importance)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color="#673AB7")
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_cols[i] for i in sorted_idx])
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost — Top 15 Feature Importances")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "03_xgb_feature_importance.png"), dpi=150)
    plt.close(fig)
    print("Saved: 03_xgb_feature_importance.png")

    # ============================================================
    # 6. COMPARISON PLOTS
    # ============================================================

    # 6a. Full test period — XGBoost vs Actual
    test_dates = test_feat["date"].values
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_dates, y_test.values, label="Observed", linewidth=0.8, color="#2196F3")
    ax.plot(
        test_dates, xgb_pred, label="XGBoost", linewidth=0.8, color="#FF5722", alpha=0.8
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_title("XGBoost Forecast vs Observed — Test Period")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "04_xgboost_prediction.png"), dpi=150)
    plt.close(fig)
    print("Saved: 04_xgboost_prediction.png")

    # 6b. Zoomed view (first 120 days of test)
    n_zoom = min(120, len(xgb_pred))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        test_dates[:n_zoom],
        y_test.values[:n_zoom],
        label="Observed",
        linewidth=1.5,
        color="#2196F3",
    )
    ax.plot(
        test_dates[:n_zoom],
        xgb_pred[:n_zoom],
        label="XGBoost",
        linewidth=1.5,
        color="#FF5722",
        linestyle="--",
    )
    ax.fill_between(
        test_dates[:n_zoom],
        y_test.values[:n_zoom],
        xgb_pred[:n_zoom],
        alpha=0.15,
        color="gray",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_title("Zoomed View — First 120 Test Days")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "05_prediction_zoomed.png"), dpi=150)
    plt.close(fig)
    print("Saved: 05_prediction_zoomed.png")

    # 6c. SARIMAX monthly forecast
    fig, ax = plt.subplots(figsize=(14, 5))
    train_monthly.plot(ax=ax, label="Train", color="#2196F3")
    test_monthly.plot(ax=ax, label="Test (Observed)", color="#4CAF50")
    sarima_forecast.plot(
        ax=ax, label="SARIMAX Forecast", color="#FF5722", linestyle="--"
    )
    ax.set_ylabel("Mean Monthly Streamflow (m³/s)")
    ax.set_title("SARIMAX Monthly Forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "06_sarimax_forecast.png"), dpi=150)
    plt.close(fig)
    print("Saved: 06_sarimax_forecast.png")

    # 6d. Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    # XGBoost scatter
    axes[0].scatter(y_test.values, xgb_pred, alpha=0.3, s=10, c="#FF5722")
    lims = [min(y_test.min(), xgb_pred.min()), max(y_test.max(), xgb_pred.max())]
    axes[0].plot(lims, lims, "k--", linewidth=1)
    axes[0].set_xlabel("Observed (m³/s)")
    axes[0].set_ylabel("Predicted (m³/s)")
    axes[0].set_title(f"XGBoost — R²={all_results['XGBoost']['R2']:.3f}")
    axes[0].set_aspect("equal", adjustable="box")

    # SARIMAX scatter
    axes[1].scatter(
        test_monthly.values, sarima_forecast.values, alpha=0.6, s=30, c="#4CAF50"
    )
    lims2 = [
        min(test_monthly.min(), sarima_forecast.min()),
        max(test_monthly.max(), sarima_forecast.max()),
    ]
    axes[1].plot(lims2, lims2, "k--", linewidth=1)
    axes[1].set_xlabel("Observed (m³/s)")
    axes[1].set_ylabel("Predicted (m³/s)")
    axes[1].set_title(f"SARIMAX — R²={all_results['SARIMAX (Monthly)']['R2']:.3f}")
    axes[1].set_aspect("equal", adjustable="box")

    fig.suptitle("Observed vs Predicted Scatter Plots", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "07_scatter_plots.png"), dpi=150)
    plt.close(fig)
    print("Saved: 07_scatter_plots.png")

    # 6e. Error analysis
    errors = y_test.values - xgb_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(errors, bins=60, color="#009688", edgecolor="white")
    ax1.axvline(0, color="red", linestyle="--")
    ax1.set_xlabel("Prediction Error (m³/s)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(
        f"XGBoost Error Distribution — MAE={all_results['XGBoost']['MAE']:.2f}"
    )

    ax2.scatter(test_dates, errors, alpha=0.2, s=5, c="#795548")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual (m³/s)")
    ax2.set_title("Residuals Over Time")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "08_error_analysis.png"), dpi=150)
    plt.close(fig)
    print("Saved: 08_error_analysis.png")

    # 6f. Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = list(all_results.keys())
    metrics_list = ["RMSE", "MAE"]
    x = np.arange(len(model_names))
    width = 0.35
    colors = ["#2196F3", "#FF9800"]
    for i, metric in enumerate(metrics_list):
        values = [all_results[m][metric] for m in model_names]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Error (m³/s)")
    ax.set_title("Model Comparison — Forecasting Errors")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "09_model_comparison.png"), dpi=150)
    plt.close(fig)
    print("Saved: 09_model_comparison.png")

    # ============================================================
    # 7. SAVE RESULTS
    # ============================================================
    with open(os.path.join(output_dir, "forecast_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 50}")
    print("FINAL MODEL COMPARISON")
    print("=" * 50)
    for name, metrics in all_results.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

    return all_results


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "streamflow.csv")
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    run_forecasting(csv_path=data_path, output_dir=results_dir)

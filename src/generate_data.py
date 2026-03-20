"""
Generate synthetic daily streamflow data for time series forecasting.

Creates a 15-year daily streamflow time series with:
- Strong seasonal pattern (spring snowmelt, summer low flow)
- Trend component (slight increase over time)
- Autocorrelation structure
- Precipitation-driven event peaks
- Realistic noise
"""

import numpy as np
import pandas as pd
import os


def generate_streamflow(n_years: int = 15, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily streamflow with seasonal and event patterns."""
    rng = np.random.RandomState(seed)
    n_days = n_years * 365
    dates = pd.date_range(start="2008-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values.astype(float)

    # Seasonal baseflow
    seasonal = 50 + 30 * np.sin(2 * np.pi * (doy - 90) / 365)

    # Long-term trend (slight increase)
    trend = 0.005 * np.arange(n_days)

    # Precipitation events
    precip = np.zeros(n_days)
    rain_prob = 0.3 + 0.15 * np.sin(2 * np.pi * (doy - 30) / 365)
    rain_occur = rng.binomial(1, np.clip(rain_prob, 0.1, 0.6))
    rain_amount = rng.exponential(8, n_days) * rain_occur
    # Extreme events
    extreme = rng.binomial(1, 0.004, n_days).astype(bool)
    rain_amount[extreme] *= rng.uniform(4, 10, extreme.sum())
    precip = np.maximum(rain_amount, 0)

    # Streamflow with memory
    streamflow = np.zeros(n_days)
    streamflow[0] = 50.0
    for t in range(1, n_days):
        # Autoregressive component
        ar = 0.85 * streamflow[t - 1]
        # Seasonal baseflow pull
        base_pull = 0.15 * seasonal[t]
        # Rainfall response (lagged)
        rain_response = 0.0
        for lag, w in enumerate([0.3, 0.25, 0.2, 0.15, 0.1]):
            idx = t - lag
            if idx >= 0:
                rain_response += w * precip[idx] * 0.8
        streamflow[t] = ar + base_pull + rain_response + trend[t]
        streamflow[t] += rng.normal(0, 2.0)
        streamflow[t] = max(streamflow[t], 5.0)

    # Temperature (predictor)
    temperature = (
        10 + 15 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 3, n_days)
    )

    df = pd.DataFrame(
        {
            "date": dates,
            "streamflow_m3s": np.round(streamflow, 2),
            "precipitation_mm": np.round(precip, 2),
            "temperature_c": np.round(temperature, 2),
        }
    )

    return df


if __name__ == "__main__":
    df = generate_streamflow(n_years=15)
    save_path = os.path.join(os.path.dirname(__file__), "..", "data", "streamflow.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\n{df.describe().round(2)}")

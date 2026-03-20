# Time Series Streamflow Forecasting

Daily streamflow forecasting comparing statistical (SARIMAX), machine learning (XGBoost), and baseline (seasonal naive) approaches. Features lag engineering, seasonal decomposition, and comprehensive evaluation using hydrology-standard metrics.

## Results

| Model | RMSE (m³/s) | MAE (m³/s) | R² | NSE | MAPE (%) |
|-------|-------------|------------|-----|-----|----------|
| Seasonal Naive | 90.30 | 89.94 | -11.69 | -11.69 | 39.09 |
| SARIMAX (Monthly) | 13.54 | 11.90 | 0.721 | 0.721 | 5.22 |
| **XGBoost** | **3.71** | **2.87** | **0.979** | **0.979** | **1.20** |

### Time Series Overview

![Overview](results/figures/01_time_series_overview.png)

### Seasonal Decomposition

![Decomposition](results/figures/02_seasonal_decomposition.png)

### XGBoost Feature Importance

![Feature Importance](results/figures/03_xgb_feature_importance.png)

### XGBoost Forecast vs Observed

![XGBoost Prediction](results/figures/04_xgboost_prediction.png)

### Zoomed View (First 120 Test Days)

![Zoomed](results/figures/05_prediction_zoomed.png)

### SARIMAX Monthly Forecast

![SARIMAX](results/figures/06_sarimax_forecast.png)

### Scatter Plots

![Scatter](results/figures/07_scatter_plots.png)

### Error Analysis

![Error](results/figures/08_error_analysis.png)

### Model Comparison

![Comparison](results/figures/09_model_comparison.png)

## Methodology

### Data
- 15-year synthetic daily streamflow time series (2008-2022)
- Features: precipitation (mm), temperature (°C)
- Seasonal snowmelt pattern, precipitation-driven events, autoregressive structure

### Feature Engineering (XGBoost)
- **Lag features**: 1, 2, 3, 7, 14, 30, 365-day lags
- **Rolling statistics**: 7, 14, 30-day rolling mean and standard deviation
- **Calendar features**: Cyclical month/day-of-year encoding (sin/cos)
- **Precipitation**: Lag and 7-day cumulative sum

### Models
1. **Seasonal Naive**: Same-day-of-year average from training set
2. **SARIMAX(1,1,1)(1,1,1,12)**: Applied to monthly means
3. **XGBoost**: 300 trees, depth 6, learning rate 0.05, with lag features

### Evaluation
- Train: first 13 years, Test: last 2 years (730 days)
- Metrics: RMSE, MAE, R², Nash-Sutcliffe Efficiency (NSE), MAPE

## Project Structure

```
.
├── src/
│   ├── generate_data.py   # Synthetic streamflow data generation
│   └── forecast.py        # Full forecasting pipeline
├── data/
│   └── streamflow.csv     (generated, not tracked)
├── results/
│   ├── figures/           # 9 visualization plots
│   ├── forecast_results.json
│   └── xgboost_model.pkl
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/forecast.py
```

## Tech Stack

- **XGBoost** — Gradient boosted trees for daily forecasting
- **statsmodels** — SARIMAX time series modeling
- **scikit-learn** — Evaluation metrics
- **pandas / NumPy** — Data manipulation and feature engineering
- **matplotlib** — Visualization

## References

- Kratzert, F. et al. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *HESS*.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition.
- Nash, J.E. & Sutcliffe, J.V. (1970). River flow forecasting through conceptual models. *Journal of Hydrology*.

## License

MIT

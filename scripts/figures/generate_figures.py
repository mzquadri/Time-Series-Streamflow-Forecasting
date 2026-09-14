"""Figures for this benchmark, drawn from the generated series and the recorded run.

Four figures, each answering one question:

    01  what kind of series this is, and why that decides everything after
    02  which method actually wins on the same rows
    03  why gradient boosting loses here
    04  why the original daily against monthly comparison could not be read

    python scripts/figures/generate_figures.py

Output: docs/figures/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import portfolio_style as ps  # noqa: E402

from generate_data import generate_streamflow  # noqa: E402

OUT = ROOT / "docs" / "figures"
BENCH = ROOT / "results" / "benchmark.json"

NICE = {
    "persistence": "Persistence",
    "random_walk_with_drift": "Random walk with drift",
    "day_of_year_mean": "Day-of-year mean",
    "day_of_year_mean_plus_trend": "Day-of-year mean plus trend",
    "ridge_on_features": "Ridge on features",
    "xgboost": "XGBoost",
    "sarimax": "SARIMAX",
}
COLOUR = {
    "ridge_on_features": ps.GREEN,
    "xgboost": ps.RED,
    "persistence": ps.BLUE,
    "random_walk_with_drift": ps.BLUE_SOFT,
    "day_of_year_mean_plus_trend": ps.AMBER,
    "day_of_year_mean": ps.SLATE_SOFT,
    "sarimax": ps.GREEN,
}


def load() -> dict:
    if not BENCH.exists():
        raise SystemExit("results/benchmark.json is missing. Run: python src/experiment.py")
    return json.loads(BENCH.read_text(encoding="utf-8"))


def findings(bench: dict) -> dict:
    """The conclusions the figures are drawn under, in a platform-stable form.

    Every one of these has a margin far wider than the difference between two
    machines. What is not included is the ordering of persistence against the
    drift baseline, which differ by 0.0002 in RMSE and by a single constant in
    their predictions, so their relative rank is not a claim worth pinning.
    """
    return {
        "best_daily": bench["best_daily"],
        "xgboost_beats_persistence": bench["xgboost_beats_persistence"],
        "xgboost_beats_persistence_count":
            bench["robustness"]["xgboost_beats_persistence_count"],
        "seeds_tested": bench["robustness"]["seeds_tested"],
        "ridge_still_beats_persistence":
            bench["same_day_weather"]["ridge_still_beats_persistence"],
        "xgboost_still_loses_to_persistence":
            bench["same_day_weather"]["xgboost_still_loses_to_persistence"],
    }


def fig_series(bench):
    """The two properties that decide every result: drift, and near-perfect memory."""
    df = generate_streamflow(n_years=15, seed=42)
    y = df["streamflow_m3s"].to_numpy()
    dates = pd.to_datetime(df["date"]).to_numpy()
    split = len(y) - 730

    fig = plt.figure(figsize=(13.0, 7.2))
    axL = fig.add_axes([0.070, 0.245, 0.545, 0.480])
    axR = fig.add_axes([0.710, 0.245, 0.240, 0.480])

    axL.plot(dates, y, color=ps.BLUE, lw=0.7, zorder=3)
    axL.axvspan(dates[split], dates[-1], color=ps.AMBER_SOFT, alpha=0.35, zorder=1)
    axL.text(dates[split + 40], y.max() * 0.97, "held out", fontsize=9.6,
             color=ps.AMBER, va="top")
    # The fitted straight line, to show how far the level travels.
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t[:split], y[:split], 1)
    axL.plot(dates, slope * t + intercept, color=ps.INK, lw=1.3, ls="--", zorder=4)
    ps.clean(axL)
    axL.set_ylabel("streamflow (m3/s)", fontsize=10.4)
    axL.set_xlabel("15 years of daily values", fontsize=10.4)

    lags = np.arange(1, 31)
    centred = y - y.mean()
    denom = float((centred**2).sum())
    acf = [float((centred[:-k] * centred[k:]).sum() / denom) for k in lags]
    axR.bar(lags, acf, color=ps.BLUE, width=0.7, zorder=3)
    axR.set_ylim(0, 1.05)
    ps.clean(axR)
    axR.set_xlabel("lag (days)", fontsize=10.4)
    axR.set_ylabel("autocorrelation", fontsize=10.4)
    axR.text(0, 1.05, f"lag 1 is {bench['data']['lag1_autocorrelation']:.4f}",
             transform=axR.transAxes, fontsize=10.4, color=ps.INK, fontweight="600",
             va="bottom")

    first, last = bench["data"]["first_year_mean"], bench["data"]["last_year_mean"]
    ps.title_block(
        fig, "A series that barely moves from one day to the next",
        "Two properties decide every result that follows: the level drifts a long "
        "way, and today is almost\nyesterday.", y=0.955, size=20)
    ps.footnote(fig, [
        f"The mean rises from {first:.0f} to {last:.0f} across the record, so any "
        f"method that assumes a stable level fails on the held-out years for that "
        f"reason alone, not because seasonality is unhelpful.",
        f"Lag-one autocorrelation is {bench['data']['lag1_autocorrelation']:.4f}. "
        f"Predicting yesterday's value is therefore a strong baseline, and a model "
        f"that cannot beat it has learned nothing about this series.",
        "Source: the seed 42 generator, results/benchmark.json."], y=0.100)
    ps.save(fig, OUT, "01_series_character", findings=findings(bench))


def fig_daily(bench):
    """Every daily method on identical rows."""
    daily = bench["daily"]
    order = sorted(daily, key=lambda k: daily[k]["rmse"])

    fig = plt.figure(figsize=(13.0, 7.4))
    ax = fig.add_axes([0.290, 0.250, 0.620, 0.470])

    ypos = np.arange(len(order))[::-1]
    for i, name in enumerate(order):
        value = daily[name]["rmse"]
        # The failing baseline is 30 times the others; clipping keeps the rest readable
        # and the true number is printed on the bar.
        shown = min(value, 12.0)
        ax.barh(ypos[i], shown, color=COLOUR.get(name, ps.SLATE), height=0.58,
                alpha=0.9, zorder=3)
        ax.text(-0.012, ypos[i], NICE.get(name, name), ha="right", va="center",
                fontsize=10.4, color=ps.INK, transform=ax.get_yaxis_transform())
        label = f"{value:.3f}" + ("  (bar clipped)" if value > 12.0 else "")
        ax.text(shown + 0.15, ypos[i], label, va="center", fontsize=9.8,
                color=COLOUR.get(name, ps.SLATE), fontweight="600")
    ax.set_yticks([])
    ax.set_xlim(0, 15.5)
    ps.clean(ax, left=False, grid_axis="x")
    ax.set_xlabel("RMSE on the held-out days (m3/s), lower is better", fontsize=10.4)

    best = bench["best_daily"]
    ps.title_block(
        fig, "The gradient booster loses to predicting yesterday",
        f"All six scored on the same {bench['split']['scored_daily_rows']} held-out "
        f"days. The two model rows read engineered lag and calendar\nfeatures; the "
        f"four baselines read almost nothing.", y=0.955, size=20)
    ps.footnote(fig, [
        f"XGBoost reaches {daily['xgboost']['rmse']:.3f} against "
        f"{daily['persistence']['rmse']:.3f} for repeating yesterday's value. "
        f"{NICE[best]} wins at {daily[best]['rmse']:.3f}, and the next figure "
        f"explains why the linear model beats the tree ensemble.",
        f"The day-of-year mean scores {daily['day_of_year_mean']['rmse']:.1f} only "
        f"because the level drifts. Given the training trend it reaches "
        f"{daily['day_of_year_mean_plus_trend']['rmse']:.2f}, so its failure is about "
        f"the drift, not about seasonality.",
        "Source: results/benchmark.json."], y=0.100)
    ps.save(fig, OUT, "02_daily_comparison", findings=findings(bench))


def fig_mechanism(bench):
    """Why the tree loses: it cannot reproduce a near-identity mapping."""
    mech = bench["mechanism"]
    deviation = mech["mean_absolute_deviation_from_lag1"]
    daily = bench["daily"]
    names = ["persistence", "ridge_on_features", "xgboost"]

    fig = plt.figure(figsize=(13.0, 7.0))
    axL = fig.add_axes([0.075, 0.255, 0.370, 0.465])
    axR = fig.add_axes([0.575, 0.255, 0.370, 0.465])

    x = np.arange(len(names))
    for i, name in enumerate(names):
        axL.bar(i, deviation[name], color=COLOUR.get(name, ps.SLATE), width=0.55,
                alpha=0.9, zorder=3)
        axL.text(i, deviation[name] + 0.06, f"{deviation[name]:.2f}", ha="center",
                 fontsize=10.0, color=COLOUR.get(name, ps.SLATE), fontweight="600")
    change = mech["daily_change_std"]
    axL.axhline(change, color=ps.INK, lw=1.4, ls="--", zorder=5)
    axL.text(len(names) - 0.55, change + 0.07,
             f"size of one day's change ({change:.2f})", fontsize=9.4, color=ps.INK,
             ha="right")
    axL.set_xticks(x)
    axL.set_xticklabels([NICE[n].replace(" on ", "\non ") for n in names], fontsize=9.6)
    axL.set_ylim(0, max(max(deviation.values()), change) * 1.30)
    ps.clean(axL)
    axL.set_ylabel("mean distance from yesterday's value", fontsize=10.4)
    axL.text(0, 1.06, "how far each method moves from lag 1", transform=axL.transAxes,
             fontsize=10.4, color=ps.INK, fontweight="600", va="bottom")

    for i, name in enumerate(names):
        axR.bar(i, daily[name]["rmse"], color=COLOUR.get(name, ps.SLATE), width=0.55,
                alpha=0.9, zorder=3)
        axR.text(i, daily[name]["rmse"] + 0.06, f"{daily[name]['rmse']:.3f}",
                 ha="center", fontsize=10.0, color=COLOUR.get(name, ps.SLATE),
                 fontweight="600")
    axR.set_xticks(x)
    axR.set_xticklabels([NICE[n].replace(" on ", "\non ") for n in names], fontsize=9.6)
    axR.set_ylim(0, max(daily[n]["rmse"] for n in names) * 1.25)
    ps.clean(axR)
    axR.set_ylabel("RMSE (m3/s)", fontsize=10.4)
    axR.text(0, 1.06, "and what that costs", transform=axR.transAxes, fontsize=10.4,
             color=ps.INK, fontweight="600", va="bottom")

    ps.title_block(
        fig, "A tree cannot sit still",
        "Tomorrow is yesterday plus a small correction. The useful signal is about "
        "the size of one day's change, so\nhow precisely a method can reproduce "
        "yesterday decides how much room it has left.", y=0.955, size=20)
    ps.footnote(fig, [
        "A tree ensemble predicts piecewise constants, so it cannot reproduce the "
        "near-identity mapping exactly.",
        f"It lands {deviation['xgboost']:.2f} away from yesterday on average, close "
        f"to the whole daily change of {change:.2f}, so most of what it adds is error.",
        f"Ridge represents the identity directly and spends its "
        f"{deviation['ridge_on_features']:.2f} of movement on the seasonal and "
        f"rainfall corrections that are useful.",
        "That is why it is the only method here that improves on persistence. "
        "Source: results/benchmark.json."], y=0.145)
    ps.save(fig, OUT, "03_why_the_tree_loses", findings=findings(bench))


def fig_granularity(bench):
    """The comparison that could not be read, and the two that can."""
    daily, monthly = bench["daily"], bench["monthly"]

    fig = plt.figure(figsize=(13.0, 7.0))
    axL = fig.add_axes([0.085, 0.255, 0.355, 0.460])
    axR = fig.add_axes([0.585, 0.255, 0.355, 0.460])

    for ax, scores, title, subtitle in (
        (axL, {k: daily[k] for k in ("xgboost", "ridge_on_features", "persistence")},
         "Daily", f"{bench['split']['scored_daily_rows']} days"),
        (axR, monthly, "Monthly", f"{monthly['sarimax']['n']} months"),
    ):
        names = sorted(scores, key=lambda k: -scores[k]["r2"])
        for i, name in enumerate(names):
            ax.bar(i, scores[name]["r2"], color=COLOUR.get(name, ps.SLATE), width=0.55,
                   alpha=0.9, zorder=3)
            ax.text(i, scores[name]["r2"] + 0.012, f"{scores[name]['r2']:.3f}",
                    ha="center", fontsize=9.8, color=COLOUR.get(name, ps.SLATE),
                    fontweight="600")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([NICE.get(n, n).replace(" on ", "\non ").replace(" with ", "\nwith ")
                            for n in names], fontsize=9.2)
        ax.set_ylim(0, 1.10)
        ps.clean(ax)
        ax.set_ylabel("R2", fontsize=10.4)
        ax.text(0, 1.06, f"{title}, {subtitle}", transform=ax.transAxes, fontsize=10.6,
                color=ps.INK, fontweight="600", va="bottom")

    ps.title_block(
        fig, "Two different problems, not one comparison",
        "R2 measures variance explained relative to whatever series it is computed "
        "on. Averaging to months removes\nmost of the day-to-day variation, so the "
        "two panels are not on a common scale.", y=0.955, size=20)
    ps.footnote(fig, [
        f"The earlier version of this repository put SARIMAX at "
        f"{monthly['sarimax']['r2']:.3f} on monthly values beside XGBoost at "
        f"{daily['xgboost']['r2']:.3f} on daily values and declared a winner. Those "
        f"numbers describe different targets.",
        f"Read separately, both panels say something. SARIMAX genuinely beats a "
        f"monthly random walk ({monthly['random_walk_with_drift']['r2']:.3f}), and on "
        f"daily values the ridge model genuinely beats persistence.",
        "Source: results/benchmark.json."], y=0.100)
    ps.save(fig, OUT, "04_daily_against_monthly", findings=findings(bench))


def main() -> int:
    ps.apply()
    bench = load()
    print()
    fig_series(bench)
    fig_daily(bench)
    fig_mechanism(bench)
    fig_granularity(bench)
    print(f"\n  figures written to {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

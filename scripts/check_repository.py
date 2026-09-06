"""Verify the artifacts, and that the README still agrees with the recorded run.

The daily XGBoost score is not reproducible across machines. The same seed, the
same pinned version and the same data give 3.12 here and 3.42 on the CI runner,
because the tree builder's floating point behaviour depends on the platform.
Chasing a bit-identical gradient booster across operating systems is not a good
use of a check, so numbers are compared within a tolerance and the finding the
repository actually reports is compared exactly.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "README.md",
    "requirements.txt",
    "src/generate_data.py",
    "src/baselines.py",
    "src/experiment.py",
    "results/benchmark.json",
    "docs/figures/01_series_character.png",
    "docs/figures/02_daily_comparison.png",
    "docs/figures/03_why_the_tree_loses.png",
    "docs/figures/04_daily_against_monthly.png",
)

#: Relative tolerance for a published number. Wide enough for platform noise in
#: the tree ensemble, far tighter than any difference the README draws a
#: conclusion from.
TOLERANCE = 0.10

LABELS = {
    "persistence": "Persistence",
    "random_walk_with_drift": "Random walk with drift",
    "day_of_year_mean": "Day-of-year mean",
    "day_of_year_mean_plus_trend": "Day-of-year mean plus trend",
    "ridge_on_features": "Ridge on features",
    "xgboost": "XGBoost",
    "sarimax": "SARIMAX",
}


def close(a: float, b: float) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= TOLERANCE


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    bench = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
    readme = re.sub(r"\s+", " ", (ROOT / "README.md").read_text(encoding="utf-8"))
    problems = []

    # Both tables carry rows with the same names, so each is searched only inside
    # its own block. Searching the whole document matched the daily row while
    # checking the monthly one.
    sections = {}
    for group, heading in (("daily", "| Method | RMSE (m3/s) | MAE (m3/s) | R2 |"),
                           ("monthly", "| Monthly, 24 points | RMSE | MAE | R2 |")):
        start = readme.find(heading)
        if start < 0:
            problems.append(f"the {group} results table is missing from the README")
            sections[group] = ""
            continue
        rest = readme[start + len(heading):]
        # Stop at the next heading, so a table cannot absorb the one after it.
        sections[group] = rest[:rest.find("##")] if "##" in rest else rest

    # Every table row must be present and within tolerance of the recorded run.
    for group in ("daily", "monthly"):
        for key, scores in bench[group].items():
            label = re.escape(LABELS[key])
            row = re.search(
                rf"\| \*?\*?{label}\*?\*? \| \*?\*?([\d.-]+)\*?\*? \| "
                rf"\*?\*?([\d.-]+)\*?\*? \| \*?\*?([\d.-]+)\*?\*? \|", sections[group])
            if row is None:
                problems.append(f"{group} row for {LABELS[key]} is missing from the README")
                continue
            for name, stated, actual in zip(
                ("RMSE", "MAE", "R2"),
                (float(row.group(1)), float(row.group(2)), float(row.group(3))),
                (scores["rmse"], scores["mae"], scores["r2"]), strict=True,
            ):
                if not close(stated, actual):
                    problems.append(
                        f"{group} {LABELS[key]} {name}: README says {stated}, "
                        f"the run gives {actual}")

    # The finding itself is checked exactly, not within a tolerance.
    beats = bench["robustness"]["xgboost_beats_persistence_count"]
    if beats and "loses to predicting yesterday" in readme:
        problems.append(f"XGBoost now beats persistence on {beats} seed(s), "
                        "but the README still says it loses")
    if beats == 0 and "all five" not in readme:
        problems.append("the README no longer states that XGBoost loses on every seed")

    if problems:
        raise SystemExit("README disagrees with results/benchmark.json:\n"
                         + "\n".join(problems))

    print(f"Repository check passed: {len(REQUIRED)} artifacts present, README "
          f"agrees with results/benchmark.json to within {TOLERANCE:.0%}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

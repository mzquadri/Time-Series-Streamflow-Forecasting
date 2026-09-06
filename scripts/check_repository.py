"""Verify the artifacts this experiment advertises, and that the README matches.

Cheap by design: it confirms the tracked files exist and that benchmark.json still
agrees with the headline numbers in the README. It does not regenerate data or
refit models, so it stays usable as a quick check.
"""

import json
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


def main() -> int:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))

    bench = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    problems = []
    for name in ("xgboost", "persistence", "ridge_on_features"):
        value = f"{bench['daily'][name]['rmse']:.3f}"
        if value not in readme:
            problems.append(f"README does not state the daily {name} RMSE of {value}")

    # The finding this repository exists to report. If a future run reverses it,
    # the README must not keep claiming it.
    claim = "loses to predicting yesterday"
    beats = bench["xgboost_beats_persistence"]
    if beats and claim in readme:
        problems.append("benchmark now shows XGBoost beating persistence, "
                        "but the README still says it loses")

    if problems:
        raise SystemExit("README disagrees with results/benchmark.json:\n"
                         + "\n".join(problems))

    print(f"Repository check passed: {len(REQUIRED)} artifacts present, "
          f"README matches results/benchmark.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the versioned artifacts advertised by this synthetic experiment."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "requirements.txt",
    "src/generate_data.py",
    "src/forecast.py",
    "results/forecast_results.json",
    "results/xgboost_model.pkl",
    "results/figures/04_xgboost_prediction.png",
)


def main() -> None:
    missing = [path for path in REQUIRED if not (ROOT / path).is_file()]
    if missing:
        raise SystemExit("Missing required artifacts:\n" + "\n".join(missing))
    print(f"Repository check passed: {len(REQUIRED)} required artifacts available.")


if __name__ == "__main__":
    main()

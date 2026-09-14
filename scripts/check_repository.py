"""Verify the artifacts, and that the README still agrees with the recorded run.

The daily XGBoost score is not reproducible across machines. The same seed, the
same pinned version and the same data give 3.12 here and 3.416 on the CI runner,
because the tree builder's floating point behaviour depends on the platform.
Chasing a bit-identical gradient booster across operating systems is not a good
use of a check, so that one number is compared within a wide tolerance and the
finding the repository actually reports is compared exactly.

The wide tolerance used to apply to every model, which was more than the evidence
supported. Reading the numbers the Linux runner printed against the ones this
README states: ridge 2.003, persistence 2.757, day-of-year mean 90.299, SARIMAX
4.679 and both monthly rows, 13.562 and 13.608, all reproduce exactly. XGBoost is
the only model that moves. A 10% band on the other six would have accepted
persistence drifting from 2.757 to 3.03 without a word, so they are held tight
and the allowance is spent only where it was measured to be needed.
"""

import json
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

#: Written by scripts/figures/portfolio_style.py into every figure it saves.
FINDINGS_KEY = "RunFindings"

PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])
NUL = bytes([0x00])

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

#: Relative tolerance for the gradient booster, whose score depends on the
#: platform. Wide enough for the measured 3.12 against 3.416, far tighter than
#: any difference the README draws a conclusion from.
TREE_TOLERANCE = 0.10

#: Relative tolerance for everything else. These reproduce exactly across Windows
#: and the Linux runner, so this is not an allowance for expected movement: it is
#: a guard against the last printed digit, and anything larger is a real change
#: that belongs in the README.
TOLERANCE = 0.005

#: The models that get the wide band, and why. A model is added here only after
#: its instability has been observed, never on the assumption that it might be
#: unstable.
PLATFORM_DEPENDENT = {"xgboost": "the tree builder's floating point ordering"}

LABELS = {
    "persistence": "Persistence",
    "random_walk_with_drift": "Random walk with drift",
    "day_of_year_mean": "Day-of-year mean",
    "day_of_year_mean_plus_trend": "Day-of-year mean plus trend",
    "ridge_on_features": "Ridge on features",
    "xgboost": "XGBoost",
    "sarimax": "SARIMAX",
}


def close(a: float, b: float, tolerance: float) -> bool:
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= tolerance



def png_text(path: Path) -> dict[str, str]:
    """The tEXt entries of a PNG, read without a third-party imaging library."""
    raw = path.read_bytes()
    if raw[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        raise SystemExit(f"{path.name} is not a PNG")
    entries, offset = {}, len(PNG_SIGNATURE)
    while offset + 8 <= len(raw):
        length = struct.unpack(">I", raw[offset:offset + 4])[0]
        kind = raw[offset + 4:offset + 8]
        if kind == b"tEXt":
            key, _, value = raw[offset + 8:offset + 8 + length].partition(NUL)
            entries[key.decode("latin-1")] = value.decode("latin-1")
        elif kind == b"IEND":
            break
        offset += 12 + length
    return entries


def expected_findings(bench: dict) -> dict:
    """The conclusions a figure drawn from this run would have been drawn under."""
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


def figure_problems(bench: dict) -> list[str]:
    """Each committed figure against the findings the recorded run supports.

    The figures were checked for existence and nothing else, and continuous
    integration regenerates them after the step that reads the run, so a figure
    drawn under a conclusion the run no longer supports would have survived.

    Values are not compared and neither are pixels. The XGBoost score moves
    between machines, so a figure drawn here would disagree with a run made on
    the CI runner over a difference this repository already documents as not
    meaningful. What is compared is the set of conclusions, every one of which
    has a margin far wider than the gap between two platforms.
    """
    problems: list[str] = []
    figures = sorted(FIGURES.glob("*.png"))
    if not figures:
        return ["no figures in docs/figures"]

    expected = expected_findings(bench)
    renderers = set()
    for figure in figures:
        text = png_text(figure)
        renderers.add(text.get("Software", "unrecorded"))
        if FINDINGS_KEY not in text:
            problems.append(f"{figure.name} records no findings; rerun "
                            f"scripts/figures/generate_figures.py")
            continue
        stamped = json.loads(text[FINDINGS_KEY])
        moved = sorted(k for k in expected if stamped.get(k) != expected[k])
        if moved:
            problems.append(f"{figure.name} was drawn when {', '.join(moved)} "
                            f"differed; rerun scripts/figures/generate_figures.py")
    if len(renderers) > 1:
        problems.append(f"the figures were not rendered together: {sorted(renderers)}")
    if not problems:
        print(f"Figure check passed: {len(figures)} figures carry the findings "
              f"this run supports.")
    return problems


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
                tolerance = (TREE_TOLERANCE if key in PLATFORM_DEPENDENT
                             else TOLERANCE)
                if not close(stated, actual, tolerance):
                    reason = PLATFORM_DEPENDENT.get(key)
                    note = (f" (allowed {tolerance:.0%} because of {reason})"
                            if reason else "")
                    problems.append(
                        f"{group} {LABELS[key]} {name}: README says {stated}, "
                        f"the run gives {actual}{note}")

    # The ablation. The ridge number reproduces across machines; the tree's does
    # not, which is why only one of the two is published.
    same_day = bench["same_day_weather"]
    # Only ridge's ablated score is published. The tree's moves 12% between
    # machines, wider than the 10% its full-feature row is allowed, so quoting it
    # would mean widening a tolerance to fit a number rather than because the
    # spread was measured. The conclusion is checked instead, just below.
    for key, label, tolerance in (
        ("ridge_on_features_rmse", "ridge without same-day weather", TOLERANCE),
    ):
        stated = re.search(rf"{re.escape(label)} scores \*\*([\d.]+)\*\*", readme)
        if stated is None:
            problems.append(f"the README does not state what {label} scores")
        elif not close(float(stated.group(1)), same_day[key], tolerance):
            problems.append(f"{label}: README says {stated.group(1)}, the run "
                            f"gives {same_day[key]}")
    if not same_day["ridge_still_beats_persistence"]:
        problems.append("ridge no longer beats persistence without same-day "
                        "weather, but the README says the conclusion survives")
    if not same_day["xgboost_still_loses_to_persistence"]:
        problems.append("XGBoost now beats persistence without same-day weather, "
                        "but the README says the conclusion survives")

    problems += figure_problems(bench)

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
          f"agrees with results/benchmark.json to within {TOLERANCE:.1%}, and to "
          f"{TREE_TOLERANCE:.0%} for {', '.join(sorted(PLATFORM_DEPENDENT))}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

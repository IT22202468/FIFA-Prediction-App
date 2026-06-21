"""
Update match results
====================
Logs completed World Cup 2026 scores into the dataset, then re-runs
weights.py and model.py so predictions stay current.

Two modes
---------
1. Interactive  — walks through every fixture whose date has passed and asks
                  for the score.  Skip any match by pressing Enter with no input.

       python src/update_results.py

2. Single match — supply the teams and score on the command line:

       python src/update_results.py \\
           --home "France" --away "Senegal" \\
           --home-score 2 --away-score 1

   Optional extras (default values come from fixtures_2026.csv):
       --city "Foxborough"  --country "United States"
       --tournament "FIFA World Cup"
       --no-refit           # skip re-running weights + model

What it does
------------
  1. Reads fixtures_2026.csv  to find the matching fixture row
  2. Appends the completed row to results_clean.csv
  3. Removes it from fixtures_2026.csv
  4. Re-runs  weights.py (2016+ filter)  →  results_weighted_2016.csv
  5. Re-runs  model.py                   →  dc_params_2016.json / dc_ratings_2016.csv
"""

import argparse
import pathlib
import subprocess
import sys
from datetime import date

import pandas as pd

ROOT        = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
SRC_DIR     = ROOT / "src"
FIXTURES    = DATA_DIR / "fixtures_2026.csv"
RESULTS     = DATA_DIR / "results_clean.csv"
WEIGHTED    = DATA_DIR / "results_weighted_2016.csv"


# ---------------------------------------------------------------------------
# Core update logic
# ---------------------------------------------------------------------------

def _load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _is_duplicate(results: pd.DataFrame, row: pd.Series) -> bool:
    """Return True if this fixture is already in results_clean."""
    mask = (
        (results["home_team"] == row["home_team"]) &
        (results["away_team"] == row["away_team"]) &
        (results["date"].dt.date == pd.Timestamp(row["date"]).date())
    )
    return mask.any()


def record_result(
    home_team:   str,
    away_team:   str,
    home_score:  int,
    away_score:  int,
    city:        str | None = None,
    country:     str | None = None,
    tournament:  str | None = None,
) -> bool:
    """
    Move a completed fixture from fixtures_2026.csv into results_clean.csv.

    Returns True on success, False if the fixture wasn't found.
    """
    fixtures = _load_fixtures()
    results  = _load_results()

    # Find the fixture row (match by team names, case-insensitive)
    mask = (
        fixtures["home_team"].str.lower() == home_team.lower()
    ) & (
        fixtures["away_team"].str.lower() == away_team.lower()
    )
    matches = fixtures[mask]

    if matches.empty:
        print(f"  ✗  Fixture not found: {home_team} vs {away_team}")
        print(f"     (check spelling — run --list-fixtures to see all pending)")
        return False

    fixture = matches.iloc[0].copy()

    if _is_duplicate(results, fixture):
        print(f"  ⚠  Already recorded: {home_team} {home_score}–{away_score} {away_team}")
        return False

    # Fill in the score and any overridden fields
    fixture["home_score"] = home_score
    fixture["away_score"] = away_score
    if city:       fixture["city"]       = city
    if country:    fixture["country"]    = country
    if tournament: fixture["tournament"] = tournament

    # Append to results_clean.csv
    new_row = pd.DataFrame([fixture])
    new_row["date"] = new_row["date"].dt.strftime("%Y-%m-%d")
    new_row.to_csv(RESULTS, mode="a", header=False, index=False)

    # Remove from fixtures_2026.csv
    remaining = fixtures[~mask].copy()
    remaining["date"] = remaining["date"].dt.strftime("%Y-%m-%d")
    remaining.to_csv(FIXTURES, index=False)

    print(f"  ✓  Recorded: {home_team} {home_score}–{away_score} {away_team}  "
          f"({fixture['date'].strftime('%Y-%m-%d')})")
    return True


# ---------------------------------------------------------------------------
# Re-fit pipeline
# ---------------------------------------------------------------------------

def refit(start_date: str = "2016-01-01") -> None:
    """Re-run weights.py then model.py with the 2016+ filter."""
    print("\nRe-running Phase 1 (weights) …")
    subprocess.run(
        [sys.executable, str(SRC_DIR / "weights.py"), "912.5", start_date],
        check=True,
    )

    print("\nRe-running Phase 2 (model) …")
    subprocess.run(
        [sys.executable, str(SRC_DIR / "model.py"), str(WEIGHTED)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode(refit_after: bool = True) -> None:
    """
    Walk through all fixtures whose date has already passed (≤ today)
    and prompt for the result of each one.
    """
    today    = date.today()
    fixtures = _load_fixtures()
    past     = fixtures[fixtures["date"].dt.date <= today].sort_values("date")

    if past.empty:
        print("No past-due fixtures found — nothing to update.")
        return

    print(f"Found {len(past)} fixture(s) on or before {today}.\n")
    print("Enter score as  home away  (e.g.  2 1)  or press Enter to skip.\n")

    updated = 0
    for _, row in past.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        d    = row["date"].strftime("%Y-%m-%d")
        prompt = f"  {d}  {home} vs {away}  →  score: "

        raw = input(prompt).strip()
        if not raw:
            print("     skipped")
            continue

        parts = raw.split()
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            print("     invalid input (expected two numbers, e.g. '2 1') — skipped")
            continue

        hs, as_ = int(parts[0]), int(parts[1])
        ok = record_result(home, away, hs, as_)
        if ok:
            updated += 1

    print(f"\n{updated} result(s) recorded.")

    if updated > 0 and refit_after:
        refit()


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def list_fixtures() -> None:
    today    = date.today()
    fixtures = _load_fixtures().sort_values("date")

    past    = fixtures[fixtures["date"].dt.date <= today]
    future  = fixtures[fixtures["date"].dt.date >  today]

    if not past.empty:
        print(f"\n── Past-due (awaiting scores) ──────────────────────────")
        for _, r in past.iterrows():
            print(f"  {r['date'].strftime('%Y-%m-%d')}  {r['home_team']:<25} vs  {r['away_team']}")

    if not future.empty:
        print(f"\n── Upcoming ────────────────────────────────────────────")
        for _, r in future.iterrows():
            print(f"  {r['date'].strftime('%Y-%m-%d')}  {r['home_team']:<25} vs  {r['away_team']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Update World Cup 2026 results and re-fit the DC model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode — enter scores for all past-due fixtures
  python src/update_results.py

  # Single result
  python src/update_results.py --home "France" --away "Senegal" --home-score 2 --away-score 1

  # Single result, skip re-fitting
  python src/update_results.py --home "Iraq" --away "Norway" --home-score 0 --away-score 1 --no-refit

  # See all pending fixtures
  python src/update_results.py --list-fixtures
        """,
    )

    parser.add_argument("--home",        type=str,  help="Home team name")
    parser.add_argument("--away",        type=str,  help="Away team name")
    parser.add_argument("--home-score",  type=int,  help="Home goals scored")
    parser.add_argument("--away-score",  type=int,  help="Away goals scored")
    parser.add_argument("--city",        type=str,  default=None,
                        help="City (uses value from fixtures file if omitted)")
    parser.add_argument("--country",     type=str,  default=None,
                        help="Country (uses value from fixtures file if omitted)")
    parser.add_argument("--tournament",  type=str,  default=None,
                        help="Tournament name (default: FIFA World Cup)")
    parser.add_argument("--no-refit",    action="store_true",
                        help="Skip re-running weights + model after update")
    parser.add_argument("--list-fixtures", action="store_true",
                        help="Print all pending fixtures and exit")

    args = parser.parse_args()

    if args.list_fixtures:
        list_fixtures()
        return

    # Single-result CLI mode
    if args.home and args.away:
        if args.home_score is None or args.away_score is None:
            parser.error("--home-score and --away-score are required with --home/--away")

        ok = record_result(
            home_team  = args.home,
            away_team  = args.away,
            home_score = args.home_score,
            away_score = args.away_score,
            city       = args.city,
            country    = args.country,
            tournament = args.tournament,
        )
        if ok and not args.no_refit:
            refit()
        return

    # Interactive mode (no flags given)
    interactive_mode(refit_after=not args.no_refit)


if __name__ == "__main__":
    main()

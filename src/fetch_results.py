"""
Auto-fetch World Cup 2026 results from football-data.org
=========================================================
Pulls completed matches from the API, reconciles team names with the
dataset, records any new results, and re-fits the model automatically.

Setup (one-time)
----------------
1. Get a free API key at  https://www.football-data.org/client/register
2. Store it in one of these places (checked in order):
     a) Environment variable:   export FOOTBALL_DATA_API_KEY="your_key_here"
     b) File:                   FIFA_Predictor_App/.api_key
     c) Pass it on the CLI:     python src/fetch_results.py --api-key "your_key"

Usage
-----
    # Fetch new results, record them, re-fit model
    python src/fetch_results.py

    # Preview only — don't write anything
    python src/fetch_results.py --dry-run

    # Skip re-fitting after recording
    python src/fetch_results.py --no-refit
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
from datetime import date

import pandas as pd
import requests

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SRC_DIR  = ROOT / "src"
FIXTURES = DATA_DIR / "fixtures_2026.csv"
RESULTS  = DATA_DIR / "results_clean.csv"
WEIGHTED = DATA_DIR / "results_weighted_2016.csv"

API_BASE = "https://api.football-data.org/v4"
WC_CODE  = "WC"   # football-data.org competition code for FIFA World Cup

# ---------------------------------------------------------------------------
# Team name mapping: API name → our dataset name
# football-data.org uses slightly different names for some nations.
# ---------------------------------------------------------------------------
API_TO_DATASET = {
    # Common differences
    "USA":                          "United States",
    "United States of America":     "United States",
    "Korea Republic":               "South Korea",
    "IR Iran":                      "Iran",
    "Côte d'Ivoire":                "Côte d'Ivoire",
    "Cote d'Ivoire":                "Côte d'Ivoire",
    "Ivory Coast":                  "Côte d'Ivoire",
    "Bosnia-Herzegovina":           "Bosnia and Herzegovina",
    "Bosnia & Herzegovina":         "Bosnia and Herzegovina",
    "DR Congo":                     "DR Congo",
    "Congo DR":                     "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Cape Verde":                   "Cabo Verde",
    "Curacao":                      "Curaçao",
    "Czechia":                      "Czech Republic",
    "Czech Republic":               "Czech Republic",
    "New Zealand":                  "New Zealand",
    "Saudi Arabia":                 "Saudi Arabia",
    "South Africa":                 "South Africa",
    "South Korea":                  "South Korea",
}


def _normalise_name(api_name: str) -> str:
    """Map an API team name to our dataset's canonical name."""
    return API_TO_DATASET.get(api_name, api_name)


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------

def _load_api_key(cli_key: str | None = None) -> str:
    """
    Resolve API key, checked in this order:
      1. --api-key CLI argument
      2. FOOTBALL_DATA_API_KEY environment variable
      3. .env file in the project root  (FOOTBALL_DATA_API_KEY=your_key)
      4. .api_key file in the project root  (just the key, nothing else)
    """
    if cli_key:
        return cli_key

    # 2. Environment variable (only reliable if exported in the same session)
    env = os.environ.get("FOOTBALL_DATA_API_KEY")
    if env:
        return env

    # 3. .env file — parsed manually, no python-dotenv needed
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key_name, _, val = line.partition("=")
            if key_name.strip() == "FOOTBALL_DATA_API_KEY":
                val = val.strip().strip('"').strip("'")
                if val:
                    return val

    # 4. Plain .api_key file
    key_file = ROOT / ".api_key"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key

    raise RuntimeError(
        "No API key found. Get a free key at https://www.football-data.org/client/register\n\n"
        "Then save it using ONE of these methods:\n\n"
        "  Recommended — add to a .env file in the project folder:\n"
        "    echo 'FOOTBALL_DATA_API_KEY=your_key_here' >> .env\n\n"
        "  Or save to a plain .api_key file:\n"
        "    echo 'your_key_here' > .api_key\n\n"
        "  Or pass it directly:\n"
        "    python src/fetch_results.py --api-key 'your_key_here'"
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def fetch_finished_matches(api_key: str) -> list[dict]:
    """
    Fetch all FINISHED matches in the 2026 FIFA World Cup from football-data.org.
    Returns a list of dicts with keys: date, home_team, away_team, home_score, away_score.
    """
    url     = f"{API_BASE}/competitions/{WC_CODE}/matches"
    headers = {"X-Auth-Token": api_key}
    params  = {"status": "FINISHED"}

    resp = requests.get(url, headers=headers, params=params, timeout=15)

    if resp.status_code == 403:
        raise RuntimeError(
            "API returned 403 Forbidden — check your API key and that your "
            "free-tier subscription covers the World Cup competition."
        )
    if resp.status_code == 404:
        raise RuntimeError(
            "Competition 'WC' not found — it may not be available in your "
            "subscription tier yet.  Check https://www.football-data.org/coverage"
        )
    resp.raise_for_status()

    data    = resp.json()
    matches = data.get("matches", [])

    results = []
    for m in matches:
        score = m.get("score", {}).get("fullTime", {})
        home_goals = score.get("home")
        away_goals = score.get("away")

        if home_goals is None or away_goals is None:
            continue   # result not yet available

        results.append({
            "date":       m["utcDate"][:10],          # YYYY-MM-DD
            "home_team":  _normalise_name(m["homeTeam"]["name"]),
            "away_team":  _normalise_name(m["awayTeam"]["name"]),
            "home_score": int(home_goals),
            "away_score": int(away_goals),
        })

    return results


# ---------------------------------------------------------------------------
# Reconcile with fixtures_2026.csv
# ---------------------------------------------------------------------------

def _load_fixtures() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _already_recorded(results: pd.DataFrame, home: str, away: str, match_date: str) -> bool:
    mask = (
        (results["home_team"] == home) &
        (results["away_team"] == away) &
        (results["date"].dt.strftime("%Y-%m-%d") == match_date)
    )
    return mask.any()


def _in_fixtures(fixtures: pd.DataFrame, home: str, away: str) -> pd.Series | None:
    mask = (
        (fixtures["home_team"].str.lower() == home.lower()) &
        (fixtures["away_team"].str.lower() == away.lower())
    )
    rows = fixtures[mask]
    return rows.iloc[0] if not rows.empty else None


# ---------------------------------------------------------------------------
# Main sync logic
# ---------------------------------------------------------------------------

HOST_NATIONS = {"United States", "Mexico", "Canada"}


def _build_fallback_row(home: str, away: str, d: str,
                        hs: int, as_: int) -> dict:
    """
    Build a results_clean row for a WC match that isn't in fixtures_2026.csv.
    These are matches that were already played before the dataset was assembled.
    Neutral = False only when a host nation plays at home; True otherwise.
    """
    return {
        "date":       d,
        "home_team":  home,
        "away_team":  away,
        "home_score": hs,
        "away_score": as_,
        "tournament": "FIFA World Cup",
        "city":       "",
        "country":    "",
        "neutral":    False if home in HOST_NATIONS else True,
    }


def sync(api_key: str, dry_run: bool = False, refit: bool = True) -> int:
    """
    Fetch finished matches, record any that are new, optionally re-fit.

    Handles two cases:
      A) Match is in fixtures_2026.csv  → move it to results_clean (normal flow)
      B) Match is not in fixtures_2026 but also not in results_clean
         → it's a gap in the original dataset; append it directly to results_clean
    """
    print("Fetching finished matches from football-data.org …")
    api_results = fetch_finished_matches(api_key)
    print(f"  API returned {len(api_results)} finished match(es).\n")

    fixtures = _load_fixtures()
    results  = _load_results()

    new_count = 0

    for m in api_results:
        home = m["home_team"]
        away = m["away_team"]
        hs   = m["home_score"]
        as_  = m["away_score"]
        d    = m["date"]

        tag = f"{home} {hs}–{as_} {away}  ({d})"

        # Already recorded — nothing to do
        if _already_recorded(results, home, away, d):
            continue

        fixture = _in_fixtures(fixtures, home, away)

        if fixture is not None:
            # Case A: move from fixtures_2026 → results_clean
            if dry_run:
                print(f"  [dry-run] Would record: {tag}")
                new_count += 1
                continue

            new_row = fixture.copy()
            new_row["home_score"] = hs
            new_row["away_score"] = as_
            new_row["date"]       = d

            pd.DataFrame([new_row]).to_csv(RESULTS, mode="a", header=False, index=False)

            updated_fixtures = fixtures[
                ~(
                    (fixtures["home_team"].str.lower() == home.lower()) &
                    (fixtures["away_team"].str.lower() == away.lower())
                )
            ].copy()
            updated_fixtures["date"] = updated_fixtures["date"].dt.strftime("%Y-%m-%d")
            updated_fixtures.to_csv(FIXTURES, index=False)
            fixtures = _load_fixtures()

        else:
            # Case B: dataset gap — match was played before dataset cutoff
            # and is absent from both files; append directly to results_clean
            if dry_run:
                print(f"  [dry-run] Would backfill gap: {tag}")
                new_count += 1
                continue

            new_row = _build_fallback_row(home, away, d, hs, as_)
            pd.DataFrame([new_row]).to_csv(RESULTS, mode="a", header=False, index=False)
            print(f"  ✓  Backfilled gap: {tag}")

        # Reload results so _already_recorded stays accurate
        results = _load_results()

        if fixture is not None:
            print(f"  ✓  Recorded: {tag}")
        new_count += 1

    if new_count == 0:
        print("No new results to record — everything is up to date.")
        return 0

    if dry_run:
        print(f"\n[dry-run] {new_count} result(s) would be recorded. "
              "Re-run without --dry-run to apply.")
        return new_count

    print(f"\n{new_count} new result(s) recorded.")

    if refit:
        _refit()

    return new_count


def _refit() -> None:
    print("\nRe-running Phase 1 (weights) …")
    subprocess.run(
        [sys.executable, str(SRC_DIR / "weights.py"), "912.5", "2016-01-01"],
        check=True,
    )
    print("\nRe-running Phase 2 (model) …")
    subprocess.run(
        [sys.executable, str(SRC_DIR / "model.py"), str(WEIGHTED)],
        check=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Auto-fetch WC 2026 results from football-data.org and update the model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/fetch_results.py                       # fetch, record, re-fit
  python src/fetch_results.py --dry-run             # preview only
  python src/fetch_results.py --no-refit            # record but skip re-fit
  python src/fetch_results.py --api-key "abc123"    # pass key directly
        """,
    )
    parser.add_argument("--api-key",  type=str, default=None,
                        help="football-data.org API key (or set FOOTBALL_DATA_API_KEY env var)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Show what would be recorded without writing anything")
    parser.add_argument("--no-refit", action="store_true",
                        help="Record results but skip re-running weights + model")

    args = parser.parse_args()

    try:
        key = _load_api_key(args.api_key)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    try:
        sync(api_key=key, dry_run=args.dry_run, refit=not args.no_refit)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not reach football-data.org — check your internet connection.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

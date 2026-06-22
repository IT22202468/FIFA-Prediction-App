"""
Phase 0 — Clean & Prepare
=========================
Loads results.csv, standardizes team names, drops null-score rows (unplayed
matches), and writes results_clean.csv for model training.

Expected input format
---------------------
    date        : YYYY-MM-DD  (e.g. 2022-11-20)
    home_team   : team name
    away_team   : team name
    home_score  : integer (blank/NaN for unplayed matches)
    away_score  : integer (blank/NaN for unplayed matches)
    tournament  : string
    city        : string
    country     : string
    neutral     : TRUE / FALSE

Unplayed matches (null scores)
-------------------------------
These are NOT training data. Put them in data/fixtures.csv for simulation.
clean.py will drop them and print a count so you know how many were removed.

Outputs
-------
    data/results_clean.csv   — cleaned historical results (training data)
"""

import pathlib
import sys
import pandas as pd

ROOT    = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV  = DATA_DIR / "results.csv"

# ---------------------------------------------------------------------------
# Team name canonicalization
# ---------------------------------------------------------------------------
NAME_MAP = {
    "IR Iran":                         "Iran",
    "Korea Republic":                  "South Korea",
    "Korea DPR":                       "North Korea",
    "Kyrgyz Republic":                 "Kyrgyzstan",
    "São Tomé and Príncipe":           "Sao Tome and Principe",
    "St. Kitts and Nevis":             "Saint Kitts and Nevis",
    "St. Lucia":                       "Saint Lucia",
    "St. Vincent and the Grenadines":  "Saint Vincent and the Grenadines",
    "Antigua & Barbuda":               "Antigua and Barbuda",
    "Bosnia-Herzegovina":              "Bosnia and Herzegovina",
    "Cape Verde":                      "Cabo Verde",
    "Congo DR":                        "DR Congo",
    "Congo":                           "Republic of Congo",
    "Curacao":                         "Curaçao",
    "FYR Macedonia":                   "North Macedonia",
    "Ivory Coast":                     "Côte d'Ivoire",
    "Macau":                           "Macao",
    "Trinidad & Tobago":               "Trinidad and Tobago",
    "USA":                             "United States",
    "U.S. Virgin Islands":             "US Virgin Islands",
    "Brunei":                          "Brunei Darussalam",
}


def run():
    print("=" * 60)
    print("Phase 0 — Clean & Prepare")
    print("=" * 60)

    if not RAW_CSV.exists():
        sys.exit(f"Not found: {RAW_CSV}\nPlace your historical results file at data/results.csv")

    # ── Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(RAW_CSV, dtype=str)
    print(f"Loaded {len(df):,} rows from {RAW_CSV.name}")

    # ── Parse dates (YYYY-MM-DD expected) ─────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        print(f"  ⚠  {bad_dates} rows have unparseable dates and will be dropped")
        df = df.dropna(subset=["date"])
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # ── Parse scores ──────────────────────────────────────────────────────
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # ── Drop null-score rows (unplayed / future fixtures) ─────────────────
    null_mask = df["home_score"].isna() | df["away_score"].isna()
    n_null = null_mask.sum()
    if n_null:
        print(f"  Dropped {n_null} null-score row(s) — put unplayed matches in data/fixtures.csv")
    df = df[~null_mask].copy().reset_index(drop=True)

    # ── Standardize team names ────────────────────────────────────────────
    applied = {}
    for col in ("home_team", "away_team"):
        for old, new in NAME_MAP.items():
            if old == new:
                continue
            mask = df[col] == old
            if mask.any():
                df.loc[mask, col] = new
                applied[old] = new
    if applied:
        print(f"  Name fixes: {applied}")

    # ── Normalize neutral column ───────────────────────────────────────────
    df["neutral"] = (
        df["neutral"]
        .astype(str).str.strip().str.upper()
        .map({"TRUE": True, "FALSE": False, "1": True, "0": False})
        .fillna(False)
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out = DATA_DIR / "results_clean.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved {out.name} — {len(df):,} rows")
    print("=== Phase 0 complete ===")


if __name__ == "__main__":
    run()

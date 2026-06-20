"""
Phase 0 — Clean & prepare
=========================
1. Fix date-century bug (sequential reconstruction)
2. Split off the 56 null-score rows (simulation targets)
3. Standardize team name spellings
4. Confirm / fix neutral flag for 2026 host nations (USA, Mexico, Canada)

Outputs (written to data/):
    results_clean.csv      — full historical dataset with clean dates (no null scores)
    fixtures_2026.csv      — 56 unplayed WC 2026 matches (simulation targets)
    team_name_report.txt   — report of any name variants found
"""

import os
import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "results.csv"


# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
def load_raw(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    print(f"Loaded {len(df):,} rows from {path.name}")
    return df


# ---------------------------------------------------------------------------
# 2. Fix date-century bug
# ---------------------------------------------------------------------------
def reconstruct_dates(raw_dates: pd.Series) -> pd.Series:
    """
    Walk rows in order. For YYYY-MM-DD rows, parse directly.
    For M/D/YY rows, two rules determine the century:

      (a) At the YYYY-MM-DD → M/D/YY transition (prev_yy is None): advance
          century_base until the reconstructed date is no longer before the
          last YYYY-MM-DD date.  This handles the 1899→1900 jump where the
          last YYYY-MM-DD century base is 1800.

      (b) Within the M/D/YY block (prev_yy is set): only bump the century
          when the 2-digit year drops by ≥50 (genuine century rollover,
          e.g. 1999→2000).  This avoids false bumps from minor out-of-order
          dates within the same year (e.g. June 1 appearing after June 3).
    """
    dates = []
    prev_ts = None
    prev_yy = None          # 2-digit year of the most recent M/D/YY row
    century_base = 1900     # will be overwritten by first YYYY-MM-DD century

    for raw in raw_dates:
        raw = str(raw).strip()

        if "-" in raw:
            # Already YYYY-MM-DD
            ts = pd.Timestamp(raw)
            century_base = (ts.year // 100) * 100
            prev_yy = None  # signal that the next M/D/YY block starts fresh
        else:
            # M/D/YY  (e.g. "2/3/00")
            parts = raw.split("/")
            month, day, yy = int(parts[0]), int(parts[1]), int(parts[2])

            if prev_yy is None:
                # Rule (a): first M/D/YY row after YYYY-MM-DD block.
                # Advance century until the date is not before prev_ts.
                ts = pd.Timestamp(year=century_base + yy, month=month, day=day)
                while prev_ts is not None and ts < prev_ts:
                    century_base += 100
                    ts = pd.Timestamp(year=century_base + yy, month=month, day=day)
            else:
                # Rule (b): within the M/D/YY block — only bump on rollover.
                if (prev_yy - yy) >= 50:
                    century_base += 100
                ts = pd.Timestamp(year=century_base + yy, month=month, day=day)

            prev_yy = yy

        dates.append(ts)
        prev_ts = ts

    return pd.Series(dates, name="date")


def validate_dates(df: pd.DataFrame) -> None:
    """Quick sanity checks on the reconstructed dates."""
    assert df["date"].min().year == 1872, f"Unexpected earliest year: {df['date'].min().year}"
    assert df["date"].max().year == 2026, f"Unexpected latest year: {df['date'].max().year}"
    # Allow minor out-of-order rows (data entry artifacts), but flag them
    shifted = df["date"].shift(1)
    inversions = (df["date"] < shifted).sum()
    if inversions:
        print(f"  ⚠ {inversions} minor out-of-order row(s) in raw file (data artifact, not a bug)")

    # Decade-by-decade match counts — should dip during WWI (1914-18) and WWII (1939-45)
    decade_counts = (
        df.assign(decade=(df["date"].dt.year // 10) * 10)
        .groupby("decade")
        .size()
    )
    print("\nMatch counts by decade:")
    for decade, count in decade_counts.items():
        print(f"  {decade}s: {count:,}")


# ---------------------------------------------------------------------------
# 3. Split null-score rows (unplayed 2026 WC fixtures)
# ---------------------------------------------------------------------------
def split_fixtures(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask_null = df["home_score"].isna() | df["away_score"].isna()
    fixtures = df[mask_null].copy().reset_index(drop=True)
    historical = df[~mask_null].copy().reset_index(drop=True)
    print(f"\nSplit: {len(historical):,} historical rows | {len(fixtures)} simulation fixtures")
    assert len(fixtures) == 56, f"Expected 56 null-score fixtures, got {len(fixtures)}"
    return historical, fixtures


# ---------------------------------------------------------------------------
# 4. Standardize team name spellings
# ---------------------------------------------------------------------------
# Known variants to normalize → canonical name
NAME_MAP = {
    # Common historical variants
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Kyrgyz Republic": "Kyrgyzstan",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "Antigua & Barbuda": "Antigua and Barbuda",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Brunei": "Brunei Darussalam",
    "Cape Verde": "Cabo Verde",
    "Chinese Taipei": "Chinese Taipei",  # keep as-is (FIFA standard)
    "Congo DR": "DR Congo",
    "Congo": "Republic of Congo",
    "Curacao": "Curaçao",
    "Eswatini": "Eswatini",
    "FYR Macedonia": "North Macedonia",
    "Ivory Coast": "Côte d'Ivoire",
    "Macau": "Macao",
    "Northern Ireland": "Northern Ireland",  # keep (UK constituent country)
    "Palestine": "Palestine",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "United States": "United States",  # canonical
    "USA": "United States",
    "U.S. Virgin Islands": "US Virgin Islands",
}


def standardize_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply NAME_MAP and report any fixes made."""
    applied = {}
    for col in ("home_team", "away_team"):
        for old, new in NAME_MAP.items():
            if old == new:
                continue
            mask = df[col] == old
            if mask.any():
                df.loc[mask, col] = new
                applied[old] = new

    return df, applied


def team_name_report(df: pd.DataFrame, applied: dict, out_path: pathlib.Path) -> None:
    """Write a report of all unique team names and any normalizations applied."""
    all_teams = sorted(set(df["home_team"].unique()) | set(df["away_team"].unique()))
    lines = [
        "=== Team Name Report ===",
        f"Total unique teams: {len(all_teams)}",
        "",
        "--- Normalizations applied ---",
    ]
    if applied:
        for old, new in sorted(applied.items()):
            lines.append(f"  {old!r:40s} → {new!r}")
    else:
        lines.append("  (none)")

    lines += ["", "--- All unique team names ---"]
    lines += [f"  {t}" for t in all_teams]

    out_path.write_text("\n".join(lines))
    print(f"\nTeam name report → {out_path.name}  ({len(all_teams)} teams)")


# ---------------------------------------------------------------------------
# 5. Fix neutral flag for 2026 host nations
# ---------------------------------------------------------------------------
HOST_NATIONS = {"United States", "Mexico", "Canada"}


def fix_neutral_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    For 2026 World Cup matches where a host nation is the 'home_team',
    set neutral = FALSE so home-advantage is preserved in the model.

    All other 2026 WC matches remain TRUE (genuinely neutral venues).
    """
    df = df.copy()
    # Normalize the neutral column to boolean
    df["neutral"] = df["neutral"].str.upper().map({"TRUE": True, "FALSE": False})

    is_wc_2026 = (df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)
    host_is_home = df["home_team"].isin(HOST_NATIONS)

    before = df.loc[is_wc_2026, "neutral"].sum()
    df.loc[is_wc_2026 & host_is_home, "neutral"] = False
    after = df.loc[is_wc_2026, "neutral"].sum()

    changed = int(before - after)
    print(f"\nNeutral-flag fix: {changed} 2026 WC match(es) flipped FALSE for host-nation home games")
    affected = df.loc[is_wc_2026 & host_is_home, ["date", "home_team", "away_team", "neutral"]]
    if not affected.empty:
        print(affected.to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    print("=" * 60)
    print("Phase 0 — Clean & Prepare")
    print("=" * 60)

    # Load
    df = load_raw(RAW_CSV)

    # Fix dates
    print("\n[1/4] Reconstructing dates...")
    df["date"] = reconstruct_dates(df["date"])
    validate_dates(df)
    print("  ✓ Dates look correct")

    # Standardize team names (before split so it applies to fixtures too)
    print("\n[2/4] Standardizing team names...")
    df, applied_names = standardize_names(df)

    # Split fixtures
    print("\n[3/4] Splitting null-score fixtures...")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    historical, fixtures = split_fixtures(df)

    # Fix neutral flags (on historical + fixtures separately)
    print("\n[4/4] Fixing neutral flags for host-nation 2026 matches...")
    # The historical set won't have 2026 WC null-score rows; but apply to fixtures
    fixtures = fix_neutral_flags(fixtures)
    historical = fix_neutral_flags(historical)

    # Write outputs
    historical_out = DATA_DIR / "results_clean.csv"
    fixtures_out = DATA_DIR / "fixtures_2026.csv"
    report_out = DATA_DIR / "team_name_report.txt"

    historical.to_csv(historical_out, index=False)
    fixtures.to_csv(fixtures_out, index=False)
    team_name_report(df, applied_names, report_out)

    print(f"\n✓ Saved: {historical_out.name} ({len(historical):,} rows)")
    print(f"✓ Saved: {fixtures_out.name} ({len(fixtures)} rows)")
    print(f"✓ Saved: {report_out.name}")
    print("\n=== Phase 0 complete ===")


if __name__ == "__main__":
    run()

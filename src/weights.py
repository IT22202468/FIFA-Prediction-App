"""
Phase 1 — Build weights
=======================
Computes a per-match training weight for every row in results_clean.csv.

The combined weight is:
    w = time_decay(match_date) * tournament_importance(tournament)

Time decay
----------
    w_time = exp(-ξ · days_since_match)
    ξ = ln(2) / half_life_days        (half_life controls how fast old matches fade)

Default half-life = 912 days (~2.5 years).  This is a starting point; the exact
value should be tuned against the Phase 3 RPS backtest.

Tournament importance
---------------------
Four tiers, chosen to reflect lineup quality and match intensity:

    1.0  — Major championship finals (World Cup, continental titles, Confeds)
    0.65 — Qualifiers to major championships + secondary competitive (Nations Leagues)
    0.35 — Regional cups and sub-continental competitions
    0.15 — Friendlies and invitational tournaments

Outputs
-------
    data/results_weighted.csv   — results_clean.csv with added columns:
                                    days_since, time_weight, importance_weight, weight
"""

import pathlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Tournament importance tiers
# ---------------------------------------------------------------------------

# Tier 1 — Major championship tournaments (highest quality, full squads, high stakes)
_TIER_1 = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "Confederations Cup",
    "Oceania Nations Cup",
    "Olympic Games",
    "CONCACAF Championship",       # pre-2000 predecessor to Gold Cup
    "Pan American Championship",   # historical CONMEBOL
    "CCCF Championship",           # historical CONCACAF
    "British Home Championship",   # Home Nations competitive fixture 1884–1984
    "CONMEBOL–UEFA Cup of Champions",
}

# Tier 2 — Qualifiers to major championships + competitive Nations Leagues
_TIER_2 = {
    "FIFA World Cup qualification",
    "UEFA Euro qualification",
    "African Cup of Nations qualification",
    "AFC Asian Cup qualification",
    "Gold Cup qualification",
    "Copa América qualification",
    "CONCACAF Championship qualification",
    "Oceania Nations Cup qualification",
    "AFC Challenge Cup",
    "AFC Challenge Cup qualification",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "CONCACAF Nations League qualification",
    "AFF Championship",
    "AFF Championship qualification",
    "Arab Cup",
    "Arab Cup qualification",
    "WAFF Championship",
    "EAFF Championship",
    "EAFF Championship qualification",
    "SAFF Cup",
    "COSAFA Cup",
    "COSAFA Cup qualification",
    "CFU Caribbean Cup",
    "CFU Caribbean Cup qualification",
    "All-African Games",
    "Afro-Asian Games",
    "Asian Games",
    "Southeast Asian Games",
    "Southeast Asian Peninsular Games",
    "South Asian Games",
    "CAFA Nations Cup",
    "NAFC Championship",
    "NAFU Championship",
    "MSG Prime Minister's Cup",
    "Pacific Games",
    "Pacific Mini Games",
    "South Pacific Games",
    "South Pacific Mini Games",
    "Melanesia Cup",
    "UNIFFAC Cup",
    "UDEAC Cup",
    "Bolivarian Games",
    "Central American and Caribbean Games",
}

# Tier 3 — Regional / sub-continental cups (competitive but lower-stakes)
_TIER_3 = {
    "CECAFA Cup",
    "Gulf Cup",
    "Nordic Championship",
    "Balkan Cup",
    "Baltic Cup",
    "Island Games",
    "CONIFA World Football Cup",
    "CONIFA European Football Cup",
    "CONIFA Asia Cup",
    "CONIFA South America Football Cup",
    "CONIFA Africa Football Cup",
    "CONIFA World Football Cup qualification",
    "CONIFA World Cup qualification",
    "ELF Cup",
    "UNCAF Cup",
    "Merdeka Tournament",
    "King's Cup",
    "Kirin Cup",
    "Kirin Challenge Cup",
    "Merlion Cup",
    "Korea Cup",
    "Dynasty Cup",
    "Lunar New Year Cup",
    "Dragon Cup",
    "Great Wall Cup",
    "Vietnam Independence Cup",
    "Indonesia Tournament",
    "Nehru Cup",
    "Indian Ocean Island Games",
    "East Asian Games",
    "ASEAN Championship",
    "ASEAN Championship qualification",
    "Windward Islands Tournament",
    "West African Cup",
    "Amílcar Cabral Cup",
    "Nile Basin Tournament",
    "Muratti Vase",
    "Central European International Cup",
    "Inter-Allied Games",
    "Far Eastern Championship Games",
    "GaNEFo",
    "Viva World Cup",
    "FIFI Wild Cup",
    "Superclásico de las Américas",
    "Copa Chevallier Boutell",
    "Copa Lipton",
    "Copa Newton",
    "Copa Roca",
    "Copa Artigas",
    "Copa Paz del Chaco",
    "Copa Ramón Castilla",
    "Copa Félix Bogado",
    "Copa Bernardo O'Higgins",
    "Copa del Pacífico",
    "Copa Premio Honor Uruguayo",
    "Copa Premio Honor Argentino",
    "Copa Rio Branco",
    "Copa Carlos Dittborn",
    "Copa Juan Pinto Durán",
    "Copa Oswaldo Cruz",
    "Copa Confraternidad",
    "Trans-Tasman Cup",
    "Intercontinental Cup",
    "Brazilian Independence Cup",
    "Brazil Independence Cup",
    "Mundialito",
    "Tournoi de France",
    "Rous Cup",
    "Dunhill Cup",
    "Marlboro Cup",
    "USA Cup",
    "CONCACAF Series",
    "Peace Cup",
    "Tri-Nations Series",
    "Tri Nation Tournament",
    "Tri-Nations Cup",
    "Three Nations Cup",
    "Four Nations Tournament",
    "Four Nations' Cup",
    "Unity Cup",
    "Atlantic Cup",
    "Atlantic Heritage Cup",
    "Canadian Shield",
    "Caribbean Cup",
    "Millennium Cup",
    "Soccer Ashes",
    "ABCS Tournament",
    "Palestine Cup",
    "Palestine International Championship",
    "Jordan International Tournament",
    "Cyprus International Tournament",
    "Malta International Tournament",
    "Tournament Burkina Faso",
    "Simba Tournament",
    "Mapinduzi Cup",
    "Mahinda Rajapaksa Cup",
    "Navruz Cup",
    "FIFA Series",
    "FIFA 75th Anniversary Cup",
    "African Friendship Games",
    "Zambian Independence Tournament",
    "Dakar Tournament",
    "Scania 100 Tournament",
    "Diamond Jubilee International Football Tournament",
    "Morocco, Capital of African Football",
    "Nations Cup",
    "Al Ain International Cup",
    "OSN Cup",
    "Joe Robbie Cup",
    "Guangzhou International Friendship Tournament",
    "Beijing International Friendship Tournament",
    "Corsica Cup",
    "Tynwald Hill Tournament",
    "Benedikt Fontana Cup",
    "Évence Coppée Trophy",
    "Real Madrid 75th Anniversary Cup",
    "Hungarian Heritage Cup",
    "Hungary Heritage Cup",
    "Copa de Honor Municipalidad de Lima",
    "Inter Games",
    "Mukuru 4 Nations",
    "Outrigger Challenge Cup",
    "World Unity Cup",
    "ConIFA Challenger Cup",
    "International Tournament of Peoples, Cultures and Tribes",
    "The Other Final",
    "South Asian Super Cup",
    "TIFOCO Tournament",
    "Open International Championship",
    "Marianas Cup",
    "Two Nations Cup",
    "Matthew Cup",
    "Matthews Cup",
    "Niamh Challenge Cup",
    "Phillip Seaga Cup",
    "SKN Football Festival",
    "Philippine Peace Cup",
    "Mauritius Four Nations Cup",
    "Prime Minister's Cup",
    "United Arab Emirates Friendship Tournament",
    "VFF Cup",
    "Coupe de l'Outre-Mer",
    "Kuneitra Cup",
    "Cup of Ancient Civilizations",
    "Tri-Nations",
    "Superclásico",
    "Copa Carlos Dittborn",
    "EAFF Championship",
    "AFC Solidarity Cup",
    "Malaysian Premier Cup",
    "Lusambo Cup",
    "Celebration Cup",
    "China Cup",
    "Turkey Cup",
    "Mao Cup",
    "Slovakia Cup",
    "Nations Cup",
}

# Tier 4 — Friendlies (default; everything not in Tiers 1–3)
_TIER_4_KEYWORDS = {"friendly", "invitational", "tour", "test match"}

# Importance scores per tier
IMPORTANCE = {
    1: 1.00,
    2: 0.65,
    3: 0.35,
    4: 0.15,
}


def get_importance_tier(tournament: str) -> int:
    """Return importance tier (1–4) for a given tournament name."""
    if tournament in _TIER_1:
        return 1
    if tournament in _TIER_2:
        return 2
    if tournament in _TIER_3:
        return 3
    # Catch any unrecognized tournaments that sound like friendlies
    t_lower = tournament.lower()
    if any(kw in t_lower for kw in _TIER_4_KEYWORDS):
        return 4
    # Unknown tournament — treat as low-stakes (tier 3) rather than friendly
    return 3


def importance_weight(tournament: str) -> float:
    """Return importance weight (0–1) for a given tournament name."""
    return IMPORTANCE[get_importance_tier(tournament)]


# ---------------------------------------------------------------------------
# Time-decay weight
# ---------------------------------------------------------------------------

def time_decay_weight(
    dates: pd.Series,
    reference_date: pd.Timestamp,
    half_life_days: float = 912.5,   # ~2.5 years; tune in Phase 3
) -> pd.Series:
    """
    Exponential time-decay weight.

        w = exp(-ξ · days_since_match)
        ξ = ln(2) / half_life_days

    Parameters
    ----------
    dates : pd.Series of Timestamps
    reference_date : the "today" anchor (typically the last match in training set)
    half_life_days : matches this many days ago have weight 0.5

    Returns
    -------
    pd.Series of float weights in (0, 1]
    """
    xi = np.log(2) / half_life_days
    days_since = (reference_date - dates).dt.days.astype(float)
    return np.exp(-xi * days_since)


# ---------------------------------------------------------------------------
# Combined weight
# ---------------------------------------------------------------------------

def build_weights(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    half_life_days: float = 912.5,
) -> pd.DataFrame:
    """
    Add weight columns to the cleaned historical DataFrame.

    Parameters
    ----------
    df : output of Phase 0 (results_clean.csv) — must have 'date' and 'tournament'
    reference_date : anchor for time decay (default: most recent match in df)
    half_life_days : half-life for exponential decay (tune via Phase 3 backtest)

    Returns
    -------
    df with added columns:
        days_since        — calendar days before reference_date
        time_weight       — exponential decay weight
        importance_weight — tournament tier weight
        weight            — combined = time_weight * importance_weight
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if reference_date is None:
        reference_date = df["date"].max()

    df["days_since"] = (reference_date - df["date"]).dt.days.astype(int)
    df["time_weight"] = time_decay_weight(df["date"], reference_date, half_life_days)
    df["importance_weight"] = df["tournament"].map(importance_weight)
    df["weight"] = df["time_weight"] * df["importance_weight"]

    return df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(df: pd.DataFrame) -> None:
    """Print a summary of the weight distribution."""
    print("\n=== Weight diagnostics ===")
    ref = df["date"].max()
    half_life = np.log(2) / (-np.log(df.loc[df["days_since"] == 1, "time_weight"].values[0])) if (df["days_since"] == 1).any() else "N/A"
    print(f"Reference date (time-decay anchor): {ref.date()}")
    print(f"Half-life: {half_life:.1f} days (~{half_life/365.25:.2f} years)" if isinstance(half_life, float) else f"Half-life: {half_life}")

    print("\n--- Importance weight by tier ---")
    tier_summary = (
        df.assign(tier=df["tournament"].map(get_importance_tier))
        .groupby("tier")
        .agg(
            matches=("weight", "count"),
            example_tournaments=("tournament", lambda x: ", ".join(x.value_counts().head(3).index)),
        )
        .rename(index={1: "Tier 1 (1.00)", 2: "Tier 2 (0.65)", 3: "Tier 3 (0.35)", 4: "Tier 4 / Friendly (0.15)"})
    )
    print(tier_summary.to_string())

    print("\n--- Combined weight distribution ---")
    print(df["weight"].describe().round(4))

    print("\n--- Effective sample size (sum of weights by decade) ---")
    decade_w = (
        df.assign(decade=(df["date"].dt.year // 10) * 10)
        .groupby("decade")["weight"]
        .sum()
        .round(1)
    )
    for decade, w in decade_w.items():
        print(f"  {decade}s: {w:,.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(half_life_days: float = 912.5, start_date: str | None = None):
    """
    Parameters
    ----------
    half_life_days : exponential decay half-life (days)
    start_date     : if given (e.g. '2016-01-01'), only matches on or after
                     this date are included.  Output file gets a date suffix
                     so the full-history file is never overwritten.
    """
    print("=" * 60)
    print("Phase 1 — Build Weights")
    print("=" * 60)

    src = DATA_DIR / "results_clean.csv"
    df = pd.read_csv(src)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} rows from {src.name}")

    if start_date:
        cutoff = pd.Timestamp(start_date)
        before = len(df)
        df = df[df["date"] >= cutoff].copy()
        print(f"Filtered to {start_date}+: {len(df):,} rows (dropped {before - len(df):,})")

    df = build_weights(df, half_life_days=half_life_days)
    print_diagnostics(df)

    # Choose output filename: suffix with start year when filtered
    if start_date:
        suffix = f"_{start_date[:4]}"
    else:
        suffix = ""
    out = DATA_DIR / f"results_weighted{suffix}.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ Saved: {out.name} ({len(df):,} rows)")
    print("\n=== Phase 1 complete ===")

    return df


if __name__ == "__main__":
    import sys
    half_life  = float(sys.argv[1]) if len(sys.argv) > 1 else 912.5
    start_date = sys.argv[2]        if len(sys.argv) > 2 else None
    run(half_life_days=half_life, start_date=start_date)

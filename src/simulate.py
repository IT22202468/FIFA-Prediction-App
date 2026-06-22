"""
Generic Group-Stage / League Simulator
=======================================
Simulates a tournament group stage or a league competition N times using the
fitted Dixon-Coles model and reports per-team standings probabilities.

Input: data/fixtures.csv
  Required columns : home_team, away_team
  Optional columns : group    — group letter/name; enables group-stage mode
                     neutral  — TRUE/FALSE; defaults to True (group) / False (league)

Group-stage mode  (fixtures.csv has a 'group' column)
  • Teams are partitioned into groups.
  • Each group's standings are resolved with FIFA tiebreak rules:
      points → GD → GF → H2H points → H2H GD → H2H GF → random draw
  • Output: P(1st) … P(Nth), avg_pts, avg_gd, avg_gf per team.

League mode  (no 'group' column)
  • All teams share one round-robin table.
  • Output: P(finish in each position), avg_pts, avg_gd, avg_gf per team.

Usage
-----
  python src/simulate.py                          # 10 000 sims, random seed
  python src/simulate.py --sims 50000 --seed 42
  python src/simulate.py --out data/results.json  # also save to JSON
  python src/simulate.py --list-teams             # show all teams the model knows

fixtures.csv examples
---------------------
  Group stage:
      home_team,away_team,group
      Brazil,Morocco,C
      Haiti,Scotland,C
      ...

  League:
      home_team,away_team,neutral
      Arsenal,Chelsea,False
      ...
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
from predict import load_params, _dc_joint_pmf

MAX_GOALS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fixtures(path=None):
    if path is None:
        path = DATA_DIR / "fixtures.csv"
    path = pathlib.Path(path)
    if not path.exists():
        sys.exit(
            f"fixtures.csv not found at {path}.\n"
            "Create data/fixtures.csv with columns: home_team, away_team [, group] [, neutral]"
        )
    df = pd.read_csv(path, dtype=str)
    for col in ("home_team", "away_team"):
        if col not in df.columns:
            sys.exit(f"fixtures.csv is missing required column: '{col}'")
    return df


def _parse_neutral(val, default):
    return str(val).strip().upper() in ("TRUE", "1", "YES") if pd.notna(val) else default


def preprocess(fixtures, mode):
    """Add neutral_bool and group_label columns; return a list of rows for speed."""
    df = fixtures.copy()
    default_neutral = (mode == "group")

    if "neutral" in df.columns:
        df["neutral_bool"] = df["neutral"].apply(lambda v: _parse_neutral(v, default_neutral))
    else:
        df["neutral_bool"] = default_neutral

    if mode == "group":
        df["group_label"] = df["group"].astype(str).str.strip()
    else:
        df["group_label"] = "LEAGUE"

    return df[["home_team", "away_team", "group_label", "neutral_bool"]].values.tolist()


# ---------------------------------------------------------------------------
# Group / league structure detection
# ---------------------------------------------------------------------------

def detect_mode(fixtures):
    return "group" if "group" in fixtures.columns else "league"


def get_groups(fixtures):
    """Return {group_name: sorted_list_of_teams} from fixtures 'group' column."""
    groups = defaultdict(set)
    for _, row in fixtures.iterrows():
        g = str(row["group"]).strip()
        groups[g].add(row["home_team"])
        groups[g].add(row["away_team"])
    return {g: sorted(teams) for g, teams in sorted(groups.items())}


def get_league_teams(fixtures):
    return sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))


# ---------------------------------------------------------------------------
# PMF precomputation
# ---------------------------------------------------------------------------

def precompute_pmfs(rows, params):
    """
    Compute and cache the flattened joint-PMF for every unique
    (home_team, away_team, neutral_bool) triple.
    """
    atk = params["attack"]
    dfn = params["defense"]
    mu  = params["home_advantage"]
    rho = params["rho"]

    def safe_a(t): return atk.get(t, 0.0)
    def safe_d(t): return dfn.get(t, 0.0)

    pmfs    = {}
    unknown = set()

    for h, a, _g, neutral in rows:
        key = (h, a, neutral)
        if key in pmfs:
            continue
        for t in (h, a):
            if t not in atk:
                unknown.add(t)
        lh = np.exp((0.0 if neutral else mu) + safe_a(h) - safe_d(a))
        la = np.exp(safe_a(a) - safe_d(h))
        pmfs[key] = _dc_joint_pmf(lh, la, rho, MAX_GOALS).flatten()

    if unknown:
        print(f"  ⚠  Teams not in model (using 0 attack/defense): {sorted(unknown)}")

    return pmfs


# ---------------------------------------------------------------------------
# Standings (FIFA tiebreak rules)
# ---------------------------------------------------------------------------

def _compute_standings(teams, match_results, rng):
    """
    Sort teams from 1st to last.
    Tiebreak: pts → GD → GF → H2H pts → H2H GD → H2H GF → random draw.
    match_results: list of (home, home_goals, away, away_goals).
    """
    pts = defaultdict(int)
    gd  = defaultdict(int)
    gf  = defaultdict(int)

    for h, hg, a, ag in match_results:
        gf[h] += hg;  gf[a] += ag
        gd[h] += hg - ag;  gd[a] += ag - hg
        if   hg > ag: pts[h] += 3
        elif ag > hg: pts[a] += 3
        else:         pts[h] += 1;  pts[a] += 1

    def h2h(tied):
        tied_set = set(tied)
        hp = defaultdict(int); hd = defaultdict(int); hf = defaultdict(int)
        for h, hg, a, ag in match_results:
            if h in tied_set and a in tied_set:
                hf[h] += hg;  hf[a] += ag
                hd[h] += hg - ag;  hd[a] += ag - hg
                if   hg > ag: hp[h] += 3
                elif ag > hg: hp[a] += 3
                else:         hp[h] += 1;  hp[a] += 1
        return hp, hd, hf

    sorted_teams = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))

    result = []
    i = 0
    while i < len(sorted_teams):
        j = i + 1
        while (j < len(sorted_teams)
               and pts[sorted_teams[j]] == pts[sorted_teams[i]]
               and gd[sorted_teams[j]]  == gd[sorted_teams[i]]
               and gf[sorted_teams[j]]  == gf[sorted_teams[i]]):
            j += 1
        tied = sorted_teams[i:j]
        if len(tied) > 1:
            hp, hd, hf = h2h(tied)
            tied.sort(key=lambda t: (
                -hp[t], -hd[t], -hf[t],
                int(rng.integers(0, 10**9))
            ))
        result.extend(tied)
        i = j

    return result


# ---------------------------------------------------------------------------
# Single simulation pass
# ---------------------------------------------------------------------------

def _simulate_once(rows, pmfs, groups_or_teams, mode, rng):
    """
    Simulate every fixture once.
    Returns:
      standings : {group_label: [team_1st, team_2nd, ...]}
      stats     : {team: {pts, gd, gf}}
    """
    bucket = defaultdict(list)   # group_label → [(h, hg, a, ag)]
    N1 = MAX_GOALS + 1

    for h, a, g, neutral in rows:
        key = (h, a, neutral)
        flat = pmfs.get(key)
        if flat is None:
            continue
        idx = rng.choice(len(flat), p=flat)
        hg, ag = divmod(idx, N1)
        bucket[g].append((h, hg, a, ag))

    standings = {}
    stats     = {}

    if mode == "group":
        for g, teams in groups_or_teams.items():
            results = bucket.get(g, [])
            standings[g] = _compute_standings(teams, results, rng)
            p = defaultdict(int); d = defaultdict(int); f = defaultdict(int)
            for h, hg, a, ag in results:
                f[h] += hg;  f[a] += ag
                d[h] += hg - ag;  d[a] += ag - hg
                if   hg > ag: p[h] += 3
                elif ag > hg: p[a] += 3
                else:         p[h] += 1;  p[a] += 1
            for t in teams:
                stats[t] = {"pts": p[t], "gd": d[t], "gf": f[t]}
    else:
        results = bucket.get("LEAGUE", [])
        standings["LEAGUE"] = _compute_standings(groups_or_teams, results, rng)
        p = defaultdict(int); d = defaultdict(int); f = defaultdict(int)
        for h, hg, a, ag in results:
            f[h] += hg;  f[a] += ag
            d[h] += hg - ag;  d[a] += ag - hg
            if   hg > ag: p[h] += 3
            elif ag > hg: p[a] += 3
            else:         p[h] += 1;  p[a] += 1
        for t in groups_or_teams:
            stats[t] = {"pts": p[t], "gd": d[t], "gf": f[t]}

    return standings, stats


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(n_sims=10_000, seed=None, params_path=None, fixtures_path=None):
    """
    Run n_sims Monte-Carlo simulations.

    Returns
    -------
    results : dict
        Group mode  → {group_name: {team: {"P1": float, ..., "avg_pts", "avg_gd", "avg_gf"}}}
        League mode → {"LEAGUE":   {team: {"P1": float, ..., "avg_pts", "avg_gd", "avg_gf"}}}
    mode : "group" | "league"
    """
    if params_path is None:
        params_path = DATA_DIR / "dc_params.json"

    params   = load_params(params_path)
    fixtures = load_fixtures(fixtures_path)
    mode     = detect_mode(fixtures)

    if mode == "group":
        groups_or_teams = get_groups(fixtures)
        print(f"Mode: GROUP STAGE — {len(groups_or_teams)} group(s), "
              f"{sum(len(v) for v in groups_or_teams.values())} teams")
        for g, teams in groups_or_teams.items():
            print(f"  Group {g}: {', '.join(teams)}")
    else:
        groups_or_teams = get_league_teams(fixtures)
        print(f"Mode: LEAGUE — {len(groups_or_teams)} teams, {len(fixtures)} fixtures")

    rows = preprocess(fixtures, mode)
    pmfs = precompute_pmfs(rows, params)
    rng  = np.random.default_rng(seed)

    pos_counts = defaultdict(lambda: defaultdict(int))
    stat_sums  = defaultdict(lambda: {"pts": 0.0, "gd": 0.0, "gf": 0.0})
    group_of   = {}

    if mode == "group":
        for g, teams in groups_or_teams.items():
            for t in teams:
                group_of[t] = g
    else:
        for t in groups_or_teams:
            group_of[t] = "LEAGUE"

    print(f"\nRunning {n_sims:,} simulations...", end="", flush=True)

    for i in range(n_sims):
        if (i + 1) % 10_000 == 0:
            print(f" {(i+1)//1_000}k", end="", flush=True)

        standings, stats = _simulate_once(rows, pmfs, groups_or_teams, mode, rng)

        for _g, sorted_teams in standings.items():
            for pos, team in enumerate(sorted_teams, start=1):
                pos_counts[team][pos] += 1

        for team, s in stats.items():
            stat_sums[team]["pts"] += s["pts"]
            stat_sums[team]["gd"]  += s["gd"]
            stat_sums[team]["gf"]  += s["gf"]

    print(" done.")

    if mode == "group":
        n_positions = max(len(t) for t in groups_or_teams.values())
    else:
        n_positions = len(groups_or_teams)

    all_teams = set(fixtures["home_team"]) | set(fixtures["away_team"])
    results   = {}

    for team in sorted(all_teams):
        if sum(pos_counts[team].values()) == 0:
            continue
        grp = group_of.get(team, "LEAGUE")
        if grp not in results:
            results[grp] = {}
        entry = {f"P{p}": round(pos_counts[team].get(p, 0) / n_sims, 4)
                 for p in range(1, n_positions + 1)}
        entry["avg_pts"] = round(stat_sums[team]["pts"] / n_sims, 2)
        entry["avg_gd"]  = round(stat_sums[team]["gd"]  / n_sims, 2)
        entry["avg_gf"]  = round(stat_sums[team]["gf"]  / n_sims, 2)
        results[grp][team] = entry

    return results, mode


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _pos_label(p):
    return f"{p}st" if p == 1 else f"{p}nd" if p == 2 else f"{p}rd" if p == 3 else f"{p}th"


def print_group_results(results):
    """Print per-group standings tables sorted by P(1st)."""
    for g_name, group_data in sorted(results.items()):
        teams = sorted(group_data, key=lambda t: -group_data[t].get("P1", 0))
        if not teams:
            continue
        n_pos = sum(1 for k in group_data[teams[0]] if k.startswith("P"))
        hdr   = "  ".join(f"{_pos_label(p):>6}" for p in range(1, n_pos + 1))

        print(f"\n{'─' * 72}")
        print(f"  Group {g_name}")
        print(f"{'─' * 72}")
        print(f"  {'Team':<26} {hdr}  {'Pts':>5}  {'GD':>5}  {'GF':>5}")
        print(f"  {'─'*26} " + "  ".join(["─"*6]*n_pos) + f"  {'─'*5}  {'─'*5}  {'─'*5}")

        for team in teams:
            d    = group_data[team]
            vals = "  ".join(f"{d.get(f'P{p}', 0)*100:5.1f}%" for p in range(1, n_pos + 1))
            print(f"  {team:<26} {vals}  {d['avg_pts']:>5.1f}  "
                  f"{d['avg_gd']:>+5.1f}  {d['avg_gf']:>5.1f}")


def print_league_results(results):
    """Print a league table sorted by average points."""
    grp_data = results.get("LEAGUE", {})
    if not grp_data:
        return
    teams = sorted(grp_data, key=lambda t: -grp_data[t]["avg_pts"])
    n_pos = sum(1 for k in grp_data[teams[0]] if k.startswith("P"))

    print(f"\n{'─' * 72}")
    print(f"  League Table  ({n_pos} positions)")
    print(f"{'─' * 72}")
    print(f"  {'#':>3}  {'Team':<26}  {'P(1st)':>7}  {'AvgPts':>6}  {'AvgGD':>6}  {'AvgGF':>6}")
    print(f"  {'─'*3}  {'─'*26}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*6}")

    for rank, team in enumerate(teams, start=1):
        d  = grp_data[team]
        p1 = d.get("P1", 0) * 100
        print(f"  {rank:>3}. {team:<26}  {p1:6.1f}%  {d['avg_pts']:>6.1f}  "
              f"{d['avg_gd']:>+6.1f}  {d['avg_gf']:>6.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Simulate a group stage or league from fixtures.csv"
    )
    ap.add_argument("--sims",       type=int,  default=10_000,
                    help="Number of simulations (default: 10 000)")
    ap.add_argument("--seed",       type=int,  default=None,
                    help="RNG seed for reproducibility")
    ap.add_argument("--params",     type=str,  default=None,
                    help="Path to dc_params.json (default: data/dc_params.json)")
    ap.add_argument("--fixtures",   type=str,  default=None,
                    help="Path to fixtures.csv (default: data/fixtures.csv)")
    ap.add_argument("--out",        type=str,  default=None,
                    help="Save results to a JSON file")
    ap.add_argument("--list-teams", action="store_true",
                    help="List all teams the model knows and exit")
    args = ap.parse_args()

    params_path = pathlib.Path(args.params) if args.params else DATA_DIR / "dc_params.json"

    if args.list_teams:
        params = load_params(params_path)
        teams  = sorted(params["attack"])
        print(f"{len(teams)} teams in model:")
        for t in teams:
            print(f"  {t}")
        return

    results, mode = run(
        n_sims        = args.sims,
        seed          = args.seed,
        params_path   = params_path,
        fixtures_path = args.fixtures,
    )

    if mode == "group":
        print_group_results(results)
    else:
        print_league_results(results)

    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()

"""
FIFA World Cup 2022 — Full Tournament Simulator
================================================
Simulates the complete Qatar 2022 World Cup N times using the fitted
Dixon-Coles model and reports per-team probabilities for every round.

Format
------
  32 teams  •  8 groups of 4  •  Top 2 per group → Round of 16
  R16 → QF → SF → 3rd-place match → Final

Groups (Qatar 2022 draw)
------------------------
  A: Qatar, Ecuador, Senegal, Netherlands
  B: England, Iran, United States, Wales
  C: Argentina, Saudi Arabia, Mexico, Poland
  D: France, Australia, Denmark, Tunisia
  E: Spain, Costa Rica, Germany, Japan
  F: Belgium, Canada, Morocco, Croatia
  G: Brazil, Serbia, Switzerland, Cameroon
  H: Portugal, Ghana, Uruguay, South Korea

Round of 16 bracket
--------------------
  M49: 1A vs 2B     M50: 1C vs 2D     M51: 1E vs 2F     M52: 1G vs 2H
  M53: 1B vs 2A     M54: 1D vs 2C     M55: 1F vs 2E     M56: 1H vs 2G

QF:   M57: W49/W50   M58: W51/W52   M59: W53/W54   M60: W55/W56
SF:   M61: W57/W58   M62: W59/W60
3rd:  M63: L61/L62
Final: M64: W61/W62

Usage
-----
  python src/simulate_2022.py
  python src/simulate_2022.py --sims 50000 --seed 42
  python src/simulate_2022.py --out data/sim_2022.json
  python src/simulate_2022.py --list-teams
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
# Tournament structure
# ---------------------------------------------------------------------------

GROUPS = {
    "A": ["Ecuador", "Netherlands", "Qatar", "Senegal"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Mexico", "Poland", "Saudi Arabia"],
    "D": ["Australia", "Denmark", "France", "Tunisia"],
    "E": ["Costa Rica", "Germany", "Japan", "Spain"],
    "F": ["Belgium", "Canada", "Croatia", "Morocco"],
    "G": ["Brazil", "Cameroon", "Serbia", "Switzerland"],
    "H": ["Ghana", "Portugal", "South Korea", "Uruguay"],
}

# Names that appear in the model under different spellings
TEAM_ALIASES = {
    "Korea Republic":  "South Korea",
    "IR Iran":         "Iran",
    "USA":             "United States",
}

# R16: (match_id, slot1, slot2)  — slots are "1X" or "2X"
R16 = [
    (49, "1A", "2B"),
    (50, "1C", "2D"),
    (51, "1E", "2F"),
    (52, "1G", "2H"),
    (53, "1B", "2A"),
    (54, "1D", "2C"),
    (55, "1F", "2E"),
    (56, "1H", "2G"),
]

# QF/SF/Final: (match_id, feed_match_1, feed_match_2)
QF    = [(57, 49, 50), (58, 51, 52), (59, 53, 54), (60, 55, 56)]
SF    = [(61, 57, 58), (62, 59, 60)]
THIRD = (63, 61, 62)
FINAL = (64, 61, 62)

ROUND_LABELS = {
    "R16":     set(m for m, _, _ in R16),
    "QF":      set(m for m, _, _ in QF),
    "SF":      set(m for m, _, _ in SF),
    "3rd":     {THIRD[0]},
    "Final":   {FINAL[0]},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_aliases(name: str) -> str:
    return TEAM_ALIASES.get(name, name)


def _all_teams():
    return [_apply_aliases(t) for teams in GROUPS.values() for t in teams]


def _group_fixtures():
    """Return list of (home, away, group) for every group match."""
    fixtures = []
    for g, teams in GROUPS.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                fixtures.append((_apply_aliases(teams[i]),
                                 _apply_aliases(teams[j]),
                                 g))
    return fixtures


# ---------------------------------------------------------------------------
# PMF precomputation
# ---------------------------------------------------------------------------

def _precompute_pmfs(fixtures, params):
    atk = params["attack"]
    dfn = params["defense"]
    rho = params["rho"]

    def sa(t): return atk.get(t, 0.0)
    def sd(t): return dfn.get(t, 0.0)

    pmfs = {}
    unknown = set()
    for h, a, _ in fixtures:
        for t in (h, a):
            if t not in atk:
                unknown.add(t)
        key = (h, a)
        if key not in pmfs:
            lh = np.exp(sa(h) - sd(a))   # group matches: all neutral
            la = np.exp(sa(a) - sd(h))
            pmfs[key] = _dc_joint_pmf(lh, la, rho, MAX_GOALS).flatten()
    if unknown:
        print(f"  ⚠  Teams not in model (using 0): {sorted(unknown)}")
    return pmfs


# ---------------------------------------------------------------------------
# Group-stage standings (FIFA tiebreak)
# ---------------------------------------------------------------------------

def _standings(teams, results, rng):
    """Sort teams 1st→4th using FIFA tiebreak rules."""
    pts = defaultdict(int); gd = defaultdict(int); gf = defaultdict(int)
    for h, hg, a, ag in results:
        gf[h] += hg; gf[a] += ag
        gd[h] += hg - ag; gd[a] += ag - hg
        if   hg > ag: pts[h] += 3
        elif ag > hg: pts[a] += 3
        else:         pts[h] += 1; pts[a] += 1

    def h2h(tied):
        s = set(tied)
        hp = defaultdict(int); hd = defaultdict(int); hf = defaultdict(int)
        for h, hg, a, ag in results:
            if h in s and a in s:
                hf[h] += hg; hf[a] += ag
                hd[h] += hg - ag; hd[a] += ag - hg
                if   hg > ag: hp[h] += 3
                elif ag > hg: hp[a] += 3
                else:         hp[h] += 1; hp[a] += 1
        return hp, hd, hf

    order = sorted(teams, key=lambda t: (-pts[t], -gd[t], -gf[t]))
    out = []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and \
              pts[order[j]] == pts[order[i]] and \
              gd[order[j]]  == gd[order[i]]  and \
              gf[order[j]]  == gf[order[i]]:
            j += 1
        tied = order[i:j]
        if len(tied) > 1:
            hp, hd, hf = h2h(tied)
            tied.sort(key=lambda t: (-hp[t], -hd[t], -hf[t],
                                     int(rng.integers(0, 10**9))))
        out.extend(tied)
        i = j
    return out, {t: (pts[t], gd[t], gf[t]) for t in teams}


# ---------------------------------------------------------------------------
# Knockout match (extra time + penalties)
# ---------------------------------------------------------------------------

def _sim_ko(t1, t2, params, rng):
    atk = params["attack"]
    dfn = params["defense"]
    rho = params["rho"]

    def sa(t): return atk.get(t, 0.0)
    def sd(t): return dfn.get(t, 0.0)

    # 90 min (neutral)
    lh = np.exp(sa(t1) - sd(t2))
    la = np.exp(sa(t2) - sd(t1))
    flat = _dc_joint_pmf(lh, la, rho, MAX_GOALS).flatten()
    idx  = rng.choice(len(flat), p=flat)
    g1, g2 = divmod(idx, MAX_GOALS + 1)
    if g1 != g2:
        return (t1 if g1 > g2 else t2)

    # Extra time (30 min ≈ ⅓ of 90 min)
    flat_et = _dc_joint_pmf(lh / 3, la / 3, rho, MAX_GOALS).flatten()
    idx     = rng.choice(len(flat_et), p=flat_et)
    g1, g2  = divmod(idx, MAX_GOALS + 1)
    if g1 != g2:
        return (t1 if g1 > g2 else t2)

    # Penalties — weighted coin flip by attack rating
    s1 = np.exp(sa(t1)); s2 = np.exp(sa(t2))
    return t1 if rng.random() < s1 / (s1 + s2) else t2


# ---------------------------------------------------------------------------
# Single full-tournament simulation
# ---------------------------------------------------------------------------

def _simulate_once(fixtures, pmfs, params, rng):
    N1 = MAX_GOALS + 1

    # ── Group stage ──────────────────────────────────────────────────────
    bucket = defaultdict(list)   # group → [(h, hg, a, ag)]
    for h, a, g in fixtures:
        flat = pmfs[(h, a)]
        idx  = rng.choice(len(flat), p=flat)
        hg, ag = divmod(idx, N1)
        bucket[g].append((h, hg, a, ag))

    # Resolve standings
    group_winner = {}   # "1A" / "2A" … → team name
    for g, teams in GROUPS.items():
        sorted_teams, _ = _standings(
            [_apply_aliases(t) for t in teams], bucket[g], rng
        )
        group_winner[f"1{g}"] = sorted_teams[0]
        group_winner[f"2{g}"] = sorted_teams[1]

    # ── Knockout ─────────────────────────────────────────────────────────
    winner = {}   # match_id → winning team
    loser  = {}   # match_id → losing team
    reached = defaultdict(str)  # team → deepest round reached

    def _play(match_id, t1, t2):
        w = _sim_ko(t1, t2, params, rng)
        l = t2 if w == t1 else t1
        winner[match_id] = w
        loser[match_id]  = l
        return w, l

    # R16
    for mid, s1, s2 in R16:
        t1 = group_winner[s1]
        t2 = group_winner[s2]
        w, l = _play(mid, t1, t2)
        reached[w] = "R16"; reached[l] = "R16"

    # QF
    for mid, f1, f2 in QF:
        w, l = _play(mid, winner[f1], winner[f2])
        reached[w] = "QF"; reached[l] = "QF"

    # SF
    sf_losers = []
    for mid, f1, f2 in SF:
        w, l = _play(mid, winner[f1], winner[f2])
        reached[w] = "SF"; reached[l] = "SF"
        sf_losers.append(l)

    # 3rd-place
    _play(THIRD[0], sf_losers[0], sf_losers[1])

    # Final
    w, l = _play(FINAL[0], winner[SF[0][0]], winner[SF[1][0]])
    reached[w] = "Champion"; reached[l] = "Final"

    return reached


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run(n_sims=10_000, seed=None, params_path=None):
    if params_path is None:
        params_path = DATA_DIR / "dc_params.json"

    params   = load_params(params_path)
    fixtures = _group_fixtures()
    pmfs     = _precompute_pmfs(fixtures, params)
    rng      = np.random.default_rng(seed)

    all_teams = _all_teams()
    rounds    = ["R16", "QF", "SF", "Final", "Champion"]
    counts    = defaultdict(lambda: defaultdict(int))  # team → round → count

    print(f"FIFA World Cup 2022 — {n_sims:,} simulations")
    print(f"Running...", end="", flush=True)

    for i in range(n_sims):
        if (i + 1) % 10_000 == 0:
            print(f" {(i+1)//1_000}k", end="", flush=True)
        reached = _simulate_once(fixtures, pmfs, params, rng)
        for team, deepest in reached.items():
            depth = rounds.index(deepest)
            for r in rounds[:depth + 1]:
                counts[team][r] += 1

    print(" done.")

    results = {}
    for team in sorted(all_teams):
        results[team] = {r: round(counts[team][r] / n_sims, 4) for r in rounds}

    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_results(results, top_n=32):
    rounds = ["R16", "QF", "SF", "Final", "Champion"]
    sorted_teams = sorted(results, key=lambda t: -results[t].get("Champion", 0))[:top_n]

    print(f"\n{'─' * 76}")
    print(f"  {'Team':<26}  {'R16':>6}  {'QF':>6}  {'SF':>6}  {'Final':>6}  {'Champion':>8}")
    print(f"  {'─'*26}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*8}")

    for team in sorted_teams:
        d  = results[team]
        print(f"  {team:<26}  "
              f"{d['R16']*100:5.1f}%  "
              f"{d['QF']*100:5.1f}%  "
              f"{d['SF']*100:5.1f}%  "
              f"{d['Final']*100:5.1f}%  "
              f"{d['Champion']*100:7.1f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate the 2022 FIFA World Cup")
    ap.add_argument("--sims",       type=int, default=10_000)
    ap.add_argument("--seed",       type=int, default=None)
    ap.add_argument("--params",     type=str, default=None)
    ap.add_argument("--out",        type=str, default=None)
    ap.add_argument("--list-teams", action="store_true")
    args = ap.parse_args()

    params_path = pathlib.Path(args.params) if args.params else DATA_DIR / "dc_params.json"

    if args.list_teams:
        params = load_params(params_path)
        for t in sorted(params["attack"]):
            print(f"  {t}")
        return

    results = run(n_sims=args.sims, seed=args.seed, params_path=params_path)
    print_results(results)

    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved to {p}")


if __name__ == "__main__":
    main()

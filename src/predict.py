"""
Phase 4 — Single-match prediction engine
=========================================
Simulates an individual match 1,000 times using the fitted Dixon-Coles model
and reports a full scoreline prediction.

Usage (CLI)
-----------
    python src/predict.py \\
        --home "France" \\
        --away "Brazil" \\
        --neutral          \\          # omit for home-ground advantage
        --tournament "FIFA World Cup"  # optional, for display only
        --sims 1000                   # default

    # Non-neutral (home ground):
    python src/predict.py --home "Mexico" --away "South Korea"

    # List all known teams:
    python src/predict.py --list-teams

Usage (Python API)
------------------
    from src.predict import predict_match, load_params

    params = load_params()
    result = predict_match("Spain", "Germany", params, is_neutral=True)
    result.print_report()
"""

import argparse
import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import poisson

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEFAULT_PARAMS = DATA_DIR / "dc_params.json"
MAX_GOALS = 10   # max goals per team in PMF grid; tails beyond this are negligible


# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------

def load_params(path: pathlib.Path = DEFAULT_PARAMS) -> dict:
    """Load fitted DC parameters from JSON."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def _dc_joint_pmf(lh: float, la: float, rho: float, max_goals: int = MAX_GOALS
                  ) -> np.ndarray:
    """
    Build the Dixon-Coles corrected joint PMF matrix P(home=i, away=j).

    Shape: (max_goals+1, max_goals+1)
    Rows = home goals, columns = away goals.
    """
    h = np.arange(max_goals + 1)
    a = np.arange(max_goals + 1)

    # Independent Poisson marginals
    ph = poisson.pmf(h, lh)   # (max_goals+1,)
    pa = poisson.pmf(a, la)   # (max_goals+1,)

    # Joint (independent) distribution
    joint = np.outer(ph, pa)  # (max_goals+1, max_goals+1)

    # Dixon-Coles τ correction on the four low-score cells
    tau = np.ones_like(joint)
    tau[0, 0] = 1.0 - lh * la * rho
    tau[1, 0] = 1.0 + la * rho
    tau[0, 1] = 1.0 + lh * rho
    tau[1, 1] = 1.0 - rho
    tau = np.clip(tau, 1e-10, None)

    joint = joint * tau

    # Normalise so probabilities sum to 1 (small mass cut off beyond max_goals)
    joint /= joint.sum()
    return joint


def _simulate(joint_pmf: np.ndarray, n_sims: int, rng: np.random.Generator
              ) -> tuple[np.ndarray, np.ndarray]:
    """Draw n_sims (home_goals, away_goals) pairs from the joint PMF."""
    max_g = joint_pmf.shape[0]
    flat  = joint_pmf.flatten()
    idx   = rng.choice(len(flat), size=n_sims, p=flat)
    h_goals = idx // max_g
    a_goals = idx %  max_g
    return h_goals, a_goals


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MatchPrediction:
    home_team:   str
    away_team:   str
    tournament:  str
    is_neutral:  bool
    lambda_home: float
    lambda_away: float
    rho:         float
    n_sims:      int
    h_sims:      np.ndarray
    a_sims:      np.ndarray
    joint_pmf:   np.ndarray

    # ---- Derived statistics (computed in __post_init__) ----------------
    prob_home_win: float = field(init=False)
    prob_draw:     float = field(init=False)
    prob_away_win: float = field(init=False)
    expected_home: float = field(init=False)
    expected_away: float = field(init=False)
    mode_score:    tuple  = field(init=False)
    mode_prob:     float  = field(init=False)
    top_scores:    list   = field(init=False)   # [(h, a, prob), ...]

    def __post_init__(self):
        h, a = self.h_sims, self.a_sims
        n    = self.n_sims

        self.prob_home_win = float((h > a).sum() / n)
        self.prob_draw     = float((h == a).sum() / n)
        self.prob_away_win = float((h < a).sum() / n)
        self.expected_home = float(h.mean())
        self.expected_away = float(a.mean())

        # Most probable scorelines from the analytical PMF
        pmf   = self.joint_pmf
        max_g = pmf.shape[0]
        flat  = pmf.flatten()
        order = np.argsort(flat)[::-1]

        scores = []
        for idx in order[:20]:
            hi = idx // max_g
            ai = idx %  max_g
            scores.append((int(hi), int(ai), float(flat[idx])))
        self.top_scores = scores
        self.mode_score = (scores[0][0], scores[0][1])
        self.mode_prob  = scores[0][2]

    # ---- Confidence intervals (percentiles of simulation draws) --------
    def ci_home(self, pct: float = 90) -> tuple[int, int]:
        lo = (100 - pct) / 2
        return (int(np.percentile(self.h_sims, lo)),
                int(np.percentile(self.h_sims, 100 - lo)))

    def ci_away(self, pct: float = 90) -> tuple[int, int]:
        lo = (100 - pct) / 2
        return (int(np.percentile(self.a_sims, lo)),
                int(np.percentile(self.a_sims, 100 - lo)))

    # ---- Pretty-print --------------------------------------------------
    def print_report(self) -> None:
        venue = "Neutral venue" if self.is_neutral else f"{self.home_team} home"
        sep   = "=" * 60

        print(sep)
        print(f"  Match prediction  ({self.n_sims:,} simulations)")
        print(sep)
        print(f"  {self.home_team}  vs  {self.away_team}")
        print(f"  Tournament : {self.tournament}")
        print(f"  Venue      : {venue}")
        print(f"  λ home     : {self.lambda_home:.3f}  |  λ away : {self.lambda_away:.3f}")
        print(f"  ρ (DC)     : {self.rho:.4f}")
        print()

        print("  ── Result probabilities ──────────────────────────")
        hw = self.prob_home_win
        d  = self.prob_draw
        aw = self.prob_away_win
        bar_w = 30
        print(f"  {self.home_team[:18]:<18} win : {hw:5.1%}  {'█' * round(hw * bar_w)}")
        print(f"  Draw               {'':4}     : {d:5.1%}  {'█' * round(d  * bar_w)}")
        print(f"  {self.away_team[:18]:<18} win : {aw:5.1%}  {'█' * round(aw * bar_w)}")
        print()

        print("  ── Expected score ────────────────────────────────")
        print(f"  {self.home_team}: {self.expected_home:.2f}  "
              f"|  {self.away_team}: {self.expected_away:.2f}")
        lo_h, hi_h = self.ci_home()
        lo_a, hi_a = self.ci_away()
        print(f"  90% CI  →  {self.home_team}: [{lo_h}–{hi_h}]  "
              f"|  {self.away_team}: [{lo_a}–{hi_a}]")
        print()

        print("  ── Most likely scorelines ────────────────────────")
        print(f"  {'Score':<10} {'Probability':>12}  {'Bar'}")
        print("  " + "-" * 46)
        for h, a, p in self.top_scores[:10]:
            bar = "█" * round(p * 200)
            marker = " ◄ most likely" if (h, a) == self.mode_score else ""
            print(f"  {h}–{a:<7}  {p:>11.2%}  {bar}{marker}")
        print(sep)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_match(
    home_team:  str,
    away_team:  str,
    params:     dict,
    is_neutral: bool  = True,
    tournament: str   = "Friendly",
    n_sims:     int   = 1000,
    seed:       int | None = None,
    max_goals:  int   = MAX_GOALS,
) -> MatchPrediction:
    """
    Simulate a single match and return a MatchPrediction result object.

    Parameters
    ----------
    home_team  : name exactly as it appears in dc_params (case-sensitive)
    away_team  : same
    params     : dict loaded from dc_params_2016.json
    is_neutral : True  → no home-advantage applied (World Cup / neutral venue)
                 False → home-advantage μ added to home rate
    tournament : display only; doesn't affect rates in the base DC model
    n_sims     : Monte Carlo draws (≥1000 recommended)
    seed       : random seed for reproducibility (None = random)
    """
    known = set(params["attack"].keys())

    # Fallback for unknown teams: use average (0.0) parameters
    def _get(d, key):
        if key not in d:
            print(f"  ⚠  '{key}' not in fitted params — using league-average (0.0)")
        return d.get(key, 0.0)

    alpha_h = _get(params["attack"],  home_team)
    beta_h  = _get(params["defense"], home_team)
    alpha_a = _get(params["attack"],  away_team)
    beta_a  = _get(params["defense"], away_team)

    mu  = params["home_advantage"] if not is_neutral else 0.0
    rho = params["rho"]

    lh = np.exp(mu + alpha_h - beta_a)
    la = np.exp(alpha_a - beta_h)

    joint = _dc_joint_pmf(lh, la, rho, max_goals)

    rng = np.random.default_rng(seed)
    h_sims, a_sims = _simulate(joint, n_sims, rng)

    return MatchPrediction(
        home_team   = home_team,
        away_team   = away_team,
        tournament  = tournament,
        is_neutral  = is_neutral,
        lambda_home = float(lh),
        lambda_away = float(la),
        rho         = rho,
        n_sims      = n_sims,
        h_sims      = h_sims,
        a_sims      = a_sims,
        joint_pmf   = joint,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _list_teams(params: dict) -> None:
    teams = sorted(params["attack"].keys())
    print(f"{len(teams)} known teams:\n")
    for i, t in enumerate(teams, 1):
        print(f"  {i:>3}. {t}")


def main():
    parser = argparse.ArgumentParser(
        description="Dixon-Coles match score predictor (1,000 simulations)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/predict.py --home "France" --away "Brazil" --neutral
  python src/predict.py --home "Mexico" --away "South Korea" --tournament "FIFA World Cup"
  python src/predict.py --list-teams
        """,
    )
    parser.add_argument("--home",       type=str, help="Home team name")
    parser.add_argument("--away",       type=str, help="Away team name")
    parser.add_argument("--neutral",    action="store_true",
                        help="Neutral venue (no home advantage)")
    parser.add_argument("--tournament", type=str, default="FIFA World Cup",
                        help="Tournament name (display only)")
    parser.add_argument("--sims",       type=int, default=1000,
                        help="Number of Monte Carlo simulations (default: 1000)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--params",     type=str, default=str(DEFAULT_PARAMS),
                        help=f"Path to dc_params JSON (default: {DEFAULT_PARAMS.name})")
    parser.add_argument("--list-teams", action="store_true",
                        help="List all teams in the fitted model and exit")

    args = parser.parse_args()
    params = load_params(pathlib.Path(args.params))

    if args.list_teams:
        _list_teams(params)
        return

    if not args.home or not args.away:
        parser.error("--home and --away are required (or use --list-teams)")

    result = predict_match(
        home_team  = args.home,
        away_team  = args.away,
        params     = params,
        is_neutral = args.neutral,
        tournament = args.tournament,
        n_sims     = args.sims,
        seed       = args.seed,
    )
    result.print_report()


if __name__ == "__main__":
    main()

"""
Phase 2 — Fit the Dixon-Coles model
=====================================
Estimates per-team attack / defense strengths, a global home-advantage
parameter, and the Dixon-Coles low-score correction (ρ) via weighted MLE.

Model
-----
    log λ_home = μ·(1 − neutral) + α_home − β_away
    log λ_away =                   α_away − β_home

    home_goals ~ Poisson(λ_home)
    away_goals ~ Poisson(λ_away)

    P(x, y | λ_h, λ_a) = τ(x, y, λ_h, λ_a, ρ) · Pois(x|λ_h) · Pois(y|λ_a)

Dixon-Coles correction τ:
    (0,0) → 1 − λ_h λ_a ρ
    (1,0) → 1 + λ_a ρ
    (0,1) → 1 + λ_h ρ
    (1,1) → 1 − ρ
    else  → 1

Regularization
--------------
Tikhonov (L2) penalty with per-team coefficient shrinks sparse teams toward 0:
    reg_coeff[t] = base_reg + sparse_reg / max(eff_sample[t], 1)

An identifiability penalty (mean of attack params)² keeps the attack scale
anchored near zero without hard constraints.

Outputs
-------
    data/dc_params.json   — fitted parameters (attack, defense, μ, ρ)
    data/dc_ratings.csv   — per-team summary table sorted by overall strength
"""

import json
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Dixon-Coles τ and its partial derivatives  (vectorized)
# ---------------------------------------------------------------------------

def _dc_tau_and_grad(x, y, lh, la, rho):
    """
    Compute τ and (∂τ/∂lh, ∂τ/∂la, ∂τ/∂ρ) for every match simultaneously.

    Parameters
    ----------
    x, y   : integer arrays of home / away goals
    lh, la : float arrays of Poisson rate parameters
    rho    : scalar ρ

    Returns
    -------
    tau          : array  (clamped to ≥ 1e-10)
    dtau_dlh     : ∂τ/∂lh
    dtau_dla     : ∂τ/∂la
    dtau_drho    : ∂τ/∂ρ
    """
    tau = np.ones(len(x))
    dtau_dlh  = np.zeros(len(x))
    dtau_dla  = np.zeros(len(x))
    dtau_drho = np.zeros(len(x))

    m00 = (x == 0) & (y == 0)
    m10 = (x == 1) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lh[m00] * la[m00] * rho
    tau[m10] = 1.0 + la[m10] * rho
    tau[m01] = 1.0 + lh[m01] * rho
    tau[m11] = 1.0 - rho

    dtau_dlh[m00]  = -la[m00] * rho
    dtau_dla[m00]  = -lh[m00] * rho
    dtau_drho[m00] = -lh[m00] * la[m00]

    dtau_dla[m10]  = rho
    dtau_drho[m10] = la[m10]

    dtau_dlh[m01]  = rho
    dtau_drho[m01] = lh[m01]

    dtau_drho[m11] = -1.0

    tau = np.clip(tau, 1e-10, None)
    return tau, dtau_dlh, dtau_dla, dtau_drho


# ---------------------------------------------------------------------------
# Negative log-likelihood and analytical gradient
# ---------------------------------------------------------------------------

def _neg_loglik_and_grad(
    params,
    home_idx, away_idx,
    home_goals, away_goals,
    weights, is_neutral,
    N,
    reg_coeffs,       # per-team regularization coefficients (length N)
    id_penalty,       # coefficient for mean(attack)² identifiability constraint
):
    """
    Returns (neg_log_lik, gradient) suitable for scipy.optimize.minimize.
    All arrays are pre-converted to numpy for speed.
    """
    attack   = params[:N]
    defense  = params[N:2*N]
    home_adv = params[2*N]
    rho      = params[2*N + 1]

    # ---- Rates -------------------------------------------------------
    ha = attack[home_idx]
    ad = defense[away_idx]
    aa = attack[away_idx]
    hd = defense[home_idx]

    log_lh = np.where(is_neutral, ha - ad, home_adv + ha - ad)
    log_la = aa - hd

    lh = np.clip(np.exp(log_lh), 1e-6, 30.0)
    la = np.clip(np.exp(log_la), 1e-6, 30.0)

    # ---- Dixon-Coles correction --------------------------------------
    tau, dtau_dlh, dtau_dla, dtau_drho = _dc_tau_and_grad(
        home_goals, away_goals, lh, la, rho
    )

    # ---- Poisson log-PMF  (Stirling not needed — gammaln exact) ------
    log_ph = home_goals * log_lh - lh - gammaln(home_goals + 1)
    log_pa = away_goals * log_la - la - gammaln(away_goals + 1)

    # ---- Weighted log-likelihood  ------------------------------------
    log_lik = np.sum(weights * (np.log(tau) + log_ph + log_pa))

    # ---- Regularization  ---------------------------------------------
    reg_attack  = reg_coeffs * attack**2
    reg_defense = reg_coeffs * defense**2
    reg_term    = np.sum(reg_attack + reg_defense)

    mean_attack = attack.mean()
    id_term     = id_penalty * mean_attack**2

    objective = -(log_lik - reg_term - id_term)

    # ---- Gradient  ---------------------------------------------------
    # Residuals for lh and la directions
    # ∂log_lik/∂lh_i = w_i * (dtau_dlh_i / tau_i + home_goals_i/lh_i - 1)
    # Since lh_i = exp(log_lh_i), chain rule gives ×lh_i for log_lh derivative
    grad_loglh = weights * (dtau_dlh / tau + home_goals / lh - 1.0) * lh
    grad_logla = weights * (dtau_dla  / tau + away_goals / la - 1.0) * la

    # Accumulate into parameter gradients
    grad_attack  = np.zeros(N)
    grad_defense = np.zeros(N)

    # attack[home]  → log_lh  +1
    np.add.at(grad_attack,  home_idx, grad_loglh)
    # defense[away] → log_lh  -1
    np.add.at(grad_defense, away_idx, -grad_loglh)
    # attack[away]  → log_la  +1
    np.add.at(grad_attack,  away_idx, grad_logla)
    # defense[home] → log_la  -1
    np.add.at(grad_defense, home_idx, -grad_logla)

    # home_advantage → log_lh for non-neutral matches only
    grad_home_adv = np.sum(grad_loglh[~is_neutral])

    # rho
    grad_rho = np.sum(weights * dtau_drho / tau)

    # Regularization gradients
    grad_attack  -= 2.0 * reg_coeffs * attack
    grad_defense -= 2.0 * reg_coeffs * defense

    # Identifiability gradient: 2 * id_penalty * mean(attack) / N
    grad_attack -= (2.0 * id_penalty * mean_attack / N)

    grad = np.concatenate([
        -grad_attack,
        -grad_defense,
        [-grad_home_adv],
        [-grad_rho],
    ])

    return objective, grad


# ---------------------------------------------------------------------------
# Per-team regularization coefficients
# ---------------------------------------------------------------------------

def _compute_reg_coeffs(teams, df, base_reg=1e-3, sparse_reg=0.05, eff_scale=10.0):
    """
    Tikhonov coefficient for each team.

        reg_coeff[t] = base_reg + sparse_reg / (1 + eff_sample[t] / eff_scale)

    Teams with little effective weight get pulled strongly toward 0;
    data-rich teams get only the small baseline penalty.
    """
    combined = pd.concat([
        df[["home_team", "weight"]].rename(columns={"home_team": "team"}),
        df[["away_team", "weight"]].rename(columns={"away_team": "team"}),
    ])
    eff = combined.groupby("team")["weight"].sum()

    coeffs = np.array([
        base_reg + sparse_reg / (1.0 + eff.get(t, 0.0) / eff_scale)
        for t in teams
    ])
    return coeffs


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def fit(
    df: pd.DataFrame,
    min_weight: float = 1e-6,
    base_reg: float = 1e-3,
    sparse_reg: float = 0.05,
    eff_scale: float = 10.0,
    id_penalty: float = 10.0,
    init_home_adv: float = 0.25,
    init_rho: float = -0.10,
    maxiter: int = 2000,
    verbose: bool = True,
) -> dict:
    """
    Fit the Dixon-Coles model via weighted MLE.

    Parameters
    ----------
    df           : results_weighted.csv (Phase 1 output) as a DataFrame
    min_weight   : discard rows with weight < this (avoids ~0-weight noise)
    base_reg     : L2 penalty applied to every team's parameters
    sparse_reg   : extra penalty for low-effective-sample teams
    eff_scale    : effective-sample scale for sparse penalty
    id_penalty   : coefficient for mean(attack)² identifiability anchor
    init_home_adv: starting value for home-advantage parameter
    init_rho     : starting value for Dixon-Coles ρ
    maxiter      : max L-BFGS-B iterations
    verbose      : print progress

    Returns
    -------
    dict with keys: attack, defense, home_advantage, rho, teams, opt_result
    """
    # ---- Prepare data ------------------------------------------------
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    df = df.dropna(subset=["home_score", "away_score", "weight"])
    df = df[df["weight"] >= min_weight].copy()

    df["home_score"] = df["home_score"].astype(int).clip(0, 25)
    df["away_score"] = df["away_score"].astype(int).clip(0, 25)

    if verbose:
        print(f"Training rows after weight filter (≥{min_weight}): {len(df):,}")

    # ---- Team index --------------------------------------------------
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    N = len(teams)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    home_idx   = df["home_team"].map(team_to_idx).to_numpy(dtype=np.int32)
    away_idx   = df["away_team"].map(team_to_idx).to_numpy(dtype=np.int32)
    home_goals = df["home_score"].to_numpy(dtype=np.int32)
    away_goals = df["away_score"].to_numpy(dtype=np.int32)
    weights    = df["weight"].to_numpy(dtype=np.float64)
    is_neutral = df["neutral"].to_numpy(dtype=bool)

    if verbose:
        print(f"Teams: {N}   Parameters: {2*N + 2}")

    # ---- Regularization coefficients ---------------------------------
    reg_coeffs = _compute_reg_coeffs(teams, df, base_reg, sparse_reg, eff_scale)

    # ---- Initial params: attack=0, defense=0, home_adv, rho ---------
    x0 = np.zeros(2*N + 2)
    x0[2*N]     = init_home_adv
    x0[2*N + 1] = init_rho

    # ---- Bounds: rho ∈ (-1, 0] to keep τ corrections valid ----------
    bounds = [(None, None)] * (2*N) + [(None, None)] + [(-0.99, 0.0)]

    # ---- Pack args ---------------------------------------------------
    args = (
        home_idx, away_idx,
        home_goals, away_goals,
        weights, is_neutral,
        N,
        reg_coeffs,
        id_penalty,
    )

    if verbose:
        print("Optimizing via L-BFGS-B …")

    result = minimize(
        _neg_loglik_and_grad,
        x0,
        args=args,
        jac=True,                  # function returns (value, gradient)
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8,
                 "iprint": 10 if verbose else -1},
    )

    if verbose:
        print(f"Converged: {result.success}  |  message: {result.message}")
        print(f"Iterations: {result.nit}  |  final -logL: {result.fun:.4f}")

    # ---- Extract parameters -----------------------------------------
    params = result.x
    attack  = dict(zip(teams, params[:N]))
    defense = dict(zip(teams, params[N:2*N]))
    home_adv = float(params[2*N])
    rho      = float(params[2*N + 1])

    return {
        "teams": teams,
        "attack": attack,
        "defense": defense,
        "home_advantage": home_adv,
        "rho": rho,
        "opt_result": result,
    }


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

def save_params(fit_result: dict, out_dir: pathlib.Path = DATA_DIR,
                suffix: str = "") -> None:
    """Save fitted parameters to JSON and a ranked ratings CSV.

    Parameters
    ----------
    suffix : appended to output filenames, e.g. '_2016' → dc_params_2016.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    payload = {
        "home_advantage": fit_result["home_advantage"],
        "rho": fit_result["rho"],
        "attack": fit_result["attack"],
        "defense": fit_result["defense"],
    }
    json_path = out_dir / f"dc_params{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved: {json_path.name}")

    # Ratings CSV
    # overall = exp(α) − exp(−β)
    #   exp(α)  ≈ goals scored vs an average opponent (average β ≈ 0)
    #   exp(−β) ≈ goals conceded from an average opponent (average α ≈ 0)
    # Positive overall = net goal surplus against a typical side.
    # This correctly rewards high α (good attack) AND high β (good defense).
    teams = fit_result["teams"]
    rows = []
    for t in teams:
        atk = fit_result["attack"][t]
        dfn = fit_result["defense"][t]
        overall = float(np.exp(atk) - np.exp(-dfn))
        rows.append({"team": t, "attack": atk, "defense": dfn, "overall": overall})

    ratings = (
        pd.DataFrame(rows)
        .sort_values("overall", ascending=False)
        .reset_index(drop=True)
    )
    ratings.index += 1
    csv_path = out_dir / f"dc_ratings{suffix}.csv"
    ratings.to_csv(csv_path, index_label="rank")
    print(f"✓ Saved: {csv_path.name}")


def load_params(in_dir: pathlib.Path = DATA_DIR) -> dict:
    """Load fitted parameters from dc_params.json."""
    with open(in_dir / "dc_params.json") as f:
        p = json.load(f)
    return p


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_diagnostics(fit_result: dict, top_n: int = 20) -> None:
    """Print top-N teams by overall strength and key scalar params.

    overall = exp(α) − exp(−β)
      = expected goals scored minus expected goals conceded against an average
        opponent (one with α_avg ≈ 0, β_avg ≈ 0 after regularisation centres
        the parameters).  Positive = net goal surplus; properly rewards both
        good attack (high α) and good defence (high β, i.e. hard to score against).
    """
    print(f"\n  home_advantage : {fit_result['home_advantage']:.4f}")
    print(f"  rho (DC corr.) : {fit_result['rho']:.4f}")

    teams = fit_result["teams"]
    overall = {
        t: float(np.exp(fit_result["attack"][t]) - np.exp(-fit_result["defense"][t]))
        for t in teams
    }
    ranked = sorted(overall.items(), key=lambda x: -x[1])

    print(f"\n  Top {top_n} teams by overall strength  [exp(α) − exp(−β)]:")
    print(f"  {'Rank':<5} {'Team':<30} {'Attack α':>9} {'Defense β':>10} {'Overall':>9}")
    print("  " + "-" * 67)
    for rank, (t, ov) in enumerate(ranked[:top_n], 1):
        print(f"  {rank:<5} {t:<30} {fit_result['attack'][t]:>9.4f} "
              f"{fit_result['defense'][t]:>10.4f} {ov:>9.4f}")

    print(f"\n  Bottom 5 teams:")
    for rank, (t, ov) in enumerate(ranked[-5:], len(ranked) - 4):
        print(f"  {rank:<5} {t:<30} {fit_result['attack'][t]:>9.4f} "
              f"{fit_result['defense'][t]:>10.4f} {ov:>9.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    input_file: str | None = None,
    min_weight: float = 1e-6,
    base_reg: float = 1e-3,
    sparse_reg: float = 0.05,
    eff_scale: float = 10.0,
) -> dict:
    """
    Parameters
    ----------
    input_file : path to the weighted CSV (default: data/results_weighted.csv).
                 The output suffix is inferred from the filename stem, e.g.
                 'results_weighted_2016.csv' → outputs 'dc_params_2016.json'.
    """
    print("=" * 60)
    print("Phase 2 — Fit Dixon-Coles Model")
    print("=" * 60)

    if input_file:
        src = pathlib.Path(input_file)
    else:
        src = DATA_DIR / "results_weighted.csv"

    df = pd.read_csv(src)
    print(f"Loaded {len(df):,} rows from {src.name}")

    result = fit(
        df,
        min_weight=min_weight,
        base_reg=base_reg,
        sparse_reg=sparse_reg,
        eff_scale=eff_scale,
        verbose=True,
    )

    print("\n=== Fitted parameters ===")
    print_diagnostics(result)

    save_params(result, suffix="")

    print("\n=== Phase 2 complete ===")
    return result


if __name__ == "__main__":
    import sys

    # Usage: python model.py [input_file] [min_weight] [base_reg] [sparse_reg]
    kw = {}
    if len(sys.argv) > 1: kw["input_file"]  = sys.argv[1]
    if len(sys.argv) > 2: kw["min_weight"]  = float(sys.argv[2])
    if len(sys.argv) > 3: kw["base_reg"]    = float(sys.argv[3])
    if len(sys.argv) > 4: kw["sparse_reg"]  = float(sys.argv[4])

    run(**kw)

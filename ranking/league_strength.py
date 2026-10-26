"""
Stage 1: estimate one strength offset per league from international series only.

Model (Bradley-Terry on the Elo scale):
    P(A beats B) = 1 / (1 + 10^((s_B - s_A) / 400))
where s = league offset (+ an optional, shrunken per-team term).

Each series' log-likelihood is weighted by format weight x age weight. Each league offset has a
Gaussian prior centred on its tier's prior offset, and each team term a Gaussian prior centred on 0.
We find the MAP estimate with scipy, with the tier order enforced as constraints during the fit:
every league in tier k must end up >= every league in tier k+1.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LN10_OVER_400 = np.log(10) / 400


def league_priors(cfg):
    """Prior offset and tier for each league, from the config."""
    return {code: (cfg["tiers"][v["tier"]]["prior_offset"], v["tier"]) for code, v in cfg["leagues"].items()}


def intl_series(series, cfg):
    """International series with the weight each one gets in the likelihood."""
    intl = series[series["is_intl"]].copy()
    fmt = intl["bo"].map(cfg["league_strength"]["format_weights"])
    intl["fit_weight"] = fmt * intl["age_w"]
    return intl


def neg_log_posterior(params, data, cfg_ls, prior_mean):
    """
    Minus the log posterior, and its gradient.
    data holds index arrays (winner league, loser league, winner team, loser team) and weights.
    params = [league offsets..., team terms...].
    """
    n_leagues = len(prior_mean)
    leagues, teams = params[:n_leagues], params[n_leagues:]
    diff = leagues[data["wl"]] - leagues[data["ll"]]
    if teams.size:
        diff = diff + teams[data["wt"]] - teams[data["lt"]]
    x = LN10_OVER_400 * diff
    # -log P(winner wins) = log(1 + e^-x), computed stably
    nll = np.sum(data["w"] * np.logaddexp(0, -x))
    # derivative of -log sigmoid(x) w.r.t. diff is -(1 - sigmoid(x)) * ln10/400
    d_diff = -data["w"] * (1 - 1 / (1 + np.exp(-x))) * LN10_OVER_400

    grad = np.zeros_like(params)
    np.add.at(grad, data["wl"], d_diff)
    np.add.at(grad, data["ll"], -d_diff)
    if teams.size:
        np.add.at(grad, n_leagues + data["wt"], d_diff)
        np.add.at(grad, n_leagues + data["lt"], -d_diff)

    # Gaussian priors
    sd = cfg_ls["prior_sd"]
    nll += np.sum((leagues - prior_mean) ** 2) / (2 * sd ** 2)
    grad[:n_leagues] += (leagues - prior_mean) / sd ** 2
    if teams.size:
        tsd = cfg_ls["team_sd"]
        nll += np.sum(teams ** 2) / (2 * tsd ** 2)
        grad[n_leagues:] += teams / tsd ** 2
    return nll, grad


def tier_constraints(codes, tiers):
    """One inequality per (league in tier k, league in tier k+1) pair: offset_k - offset_k+1 >= 0."""
    constraints = []
    ordered_tiers = sorted(set(tiers))
    for upper, lower in zip(ordered_tiers, ordered_tiers[1:]):
        for i, _ in enumerate(codes):
            for j, _ in enumerate(codes):
                if tiers[i] == upper and tiers[j] == lower:
                    vec = np.zeros(len(codes))
                    vec[i], vec[j] = 1.0, -1.0
                    constraints.append(vec)
    return constraints


def fit_league_strength(series, cfg):
    """
    Fit the league offsets. Returns (offsets dict, team terms dict, league table DataFrame).
    With league_strength.enabled = false every offset is 0 (used by the ablation).
    """
    priors = league_priors(cfg)
    codes = list(priors)
    prior_mean = np.array([priors[c][0] for c in codes], dtype=float)
    tiers = [priors[c][1] for c in codes]
    cfg_ls = cfg["league_strength"]
    intl = intl_series(series, cfg)

    if not cfg_ls["enabled"]:
        offsets, team_terms = {c: 0.0 for c in codes}, {}
        return offsets, team_terms, league_table(intl, priors, offsets)

    # Index arrays for the likelihood
    team_list = sorted(set(intl["winner"]) | set(intl["loser"])) if cfg_ls["team_term"] else []
    li = {c: i for i, c in enumerate(codes)}
    ti = {t: i for i, t in enumerate(team_list)}
    data = {
        "wl": intl["winner_league"].map(li).to_numpy(int),
        "ll": intl["loser_league"].map(li).to_numpy(int),
        "wt": intl["winner"].map(ti).fillna(0).to_numpy(int),
        "lt": intl["loser"].map(ti).fillna(0).to_numpy(int),
        "w": intl["fit_weight"].to_numpy(float),
    }

    # Start at the prior, which already satisfies the tier order.
    start = np.concatenate([prior_mean, np.zeros(len(team_list))])
    n_team = len(team_list)
    constraints = [{"type": "ineq",
                    "fun": (lambda p, v=vec: v @ p[:len(codes)]),
                    "jac": (lambda p, v=vec: np.concatenate([v, np.zeros(n_team)]))}
                   for vec in tier_constraints(codes, tiers)]
    result = minimize(neg_log_posterior, start, args=(data, cfg_ls, prior_mean), jac=True,
                      method="SLSQP", constraints=constraints, options={"maxiter": 500, "ftol": 1e-10})
    assert result.success, f"league strength fit failed: {result.message}"

    offsets = {c: float(result.x[i]) for i, c in enumerate(codes)}
    team_terms = {t: float(result.x[len(codes) + i]) for i, t in enumerate(team_list)}
    return offsets, team_terms, league_table(intl, priors, offsets)


def league_table(intl, priors, offsets):
    """League, tier, prior offset, fitted offset, cross-league international series and win rate."""
    cross = intl[intl["cross_league"]]
    rows = []
    for code, (prior, tier) in priors.items():
        wins = int((cross["winner_league"] == code).sum())
        played = wins + int((cross["loser_league"] == code).sum())
        rows.append({"league": code, "tier": tier, "prior_offset": prior,
                     "fitted_offset": round(offsets[code], 1), "intl_series": played,
                     "intl_wins": wins, "win_rate": round(wins / played, 3) if played else np.nan})
    return pd.DataFrame(rows).sort_values(["tier", "fitted_offset"], ascending=[True, False]).reset_index(drop=True)

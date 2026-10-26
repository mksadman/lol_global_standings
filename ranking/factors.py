"""
Stage 2b: the four résumé factors (each between 0 and 1) and the starting rating.

  Stakes Earned       placement points, as a ratio to the 5th-best team, through curve()
  Quality Wins        best 10 wins, valued by the beaten team's Stakes ratio, through curve()
  Weighted Network    best 10 wins, valued by the beaten team's network ratio
  International Wins  best 5 international wins, valued by event weight x age weight

Only series WINS count toward any "best N" list. Every value is also multiplied by the series'
event weight and age weight, so early-season and low-stakes wins count for less.
"""

import numpy as np
import pandas as pd

FACTOR_NAMES = ["stakes", "quality_wins", "network", "international"]


def curve(x):
    """Valve's curve: 1 / (1 + |log10(x)|), with curve(0) = 0. Squeezes small ratios up toward 1."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(x > 0, 1 / (1 + np.abs(np.log10(np.where(x > 0, x, 1)))), 0.0)


def ratio_to_reference(values, rank):
    """Each value divided by the value of the team in place `rank` (5th by default), capped at 1."""
    ordered = values.sort_values(ascending=False)
    reference = ordered.iloc[rank - 1] if len(ordered) >= rank else ordered.iloc[-1]
    if reference <= 0:
        return (values > 0).astype(float)
    return (values / reference).clip(upper=1.0)


def best_n_average(wins, n, teams):
    """
    Average of each team's best n win values, with empty slots counting as 0.
    Also marks which wins made it into the list (`counted`) so the detail pages can show them.
    """
    wins = wins.sort_values(["team", "value", "start"], ascending=[True, False, True]).copy()
    wins["slot"] = wins.groupby("team").cumcount()
    wins["counted"] = wins["slot"] < n
    totals = wins[wins["counted"]].groupby("team")["value"].sum()
    return (totals.reindex(teams, fill_value=0.0) / n), wins.drop(columns="slot")


def stakes_factor(placements, teams, cfg):
    """Sum each team's placement points, take the ratio to the 5th-best team, then curve() it."""
    fc = cfg["factors"]
    totals = placements.groupby("team")["points"].sum().reindex(teams, fill_value=0.0)
    ratio = ratio_to_reference(totals, fc["reference_rank"])
    factor = pd.Series(curve(ratio) if fc["stakes"]["curve"] else ratio, index=teams)
    return totals, ratio, factor


def win_rows(series, value_of_loser, name):
    """One row per series win: the winner, the beaten team and a value = loser value x event x age weight."""
    wins = series[["series_id", "start", "winner", "loser", "event_weight", "age_w", "is_intl"]].copy()
    wins = wins.rename(columns={"winner": "team", "loser": "opponent"})
    wins["opponent_value"] = wins["opponent"].map(value_of_loser).fillna(0.0)
    wins["value"] = wins["opponent_value"] * wins["event_weight"] * wins["age_w"]
    wins["factor"] = name
    return wins


def quality_wins_factor(series, stakes_ratio, teams, cfg):
    """Best-N average of (beaten team's Stakes ratio x event weight x age weight), then curve()."""
    fc = cfg["factors"]["quality_wins"]
    wins = win_rows(series, stakes_ratio, "quality_wins")
    avg, wins = best_n_average(wins, fc["best_n"], teams)
    factor = pd.Series(curve(avg) if fc["curve"] else avg.to_numpy(), index=teams)
    return factor, wins


def own_network(series, home, teams, cfg):
    """
    Sum, over the DISTINCT teams a team has beaten, of the beaten team's league multiplier.
    With network.age_weighted on, each beaten team is also scaled by the most recent
    (highest) age weight among the wins over it, so old wins fade here too.
    """
    mult = {code: cfg["tiers"][v["tier"]]["domestic_multiplier"] for code, v in cfg["leagues"].items()}
    pairs = series.groupby(["winner", "loser"])["age_w"].max().reset_index()
    pairs["value"] = pairs["loser"].map(home).map(mult)
    if cfg["factors"]["network"]["age_weighted"]:
        pairs["value"] *= pairs["age_w"]
    return pairs.groupby("winner")["value"].sum().reindex(teams, fill_value=0.0)


def network_factor(series, home, teams, cfg):
    """Best-N average of (beaten team's network ratio x event weight x age weight)."""
    fc = cfg["factors"]["network"]
    network = own_network(series, home, teams, cfg)
    ratio = ratio_to_reference(network, cfg["factors"]["reference_rank"])
    wins = win_rows(series, ratio, "network")
    avg, wins = best_n_average(wins, fc["best_n"], teams)
    factor = pd.Series(curve(avg) if fc["curve"] else avg.to_numpy(), index=teams)
    return network, ratio, factor, wins


def international_factor(series, teams, cfg):
    """Best-N average of (event weight x age weight) over international series wins."""
    fc = cfg["factors"]["international"]
    intl = series[series["is_intl"]]
    wins = win_rows(intl, pd.Series(1.0, index=teams), "international")
    avg, wins = best_n_average(wins, fc["best_n"], teams)
    factor = pd.Series(curve(avg) if fc["curve"] else avg.to_numpy(), index=teams)
    return factor, wins


def rescale(values, low, high):
    """Linear map so the best value becomes `high` and the worst `low` (as Valve does)."""
    span = values.max() - values.min()
    if span == 0:
        return pd.Series((low + high) / 2, index=values.index)
    return low + (values - values.min()) / span * (high - low)


def compute_factors(series, placements, home, offsets, cfg):
    """
    All four factors, the seed value and the starting rating for every team.
    Returns (factors DataFrame indexed by team, every win row with its value and `counted` flag).
    """
    teams = sorted(home)
    stakes_total, stakes_ratio, f_stakes = stakes_factor(placements, teams, cfg)
    f_quality, quality_rows = quality_wins_factor(series, stakes_ratio, teams, cfg)
    network, network_ratio, f_network, network_rows = network_factor(series, home, teams, cfg)
    f_intl, intl_rows = international_factor(series, teams, cfg)

    table = pd.DataFrame({
        "league": pd.Series(home).reindex(teams),
        "stakes_total": stakes_total, "stakes_ratio": stakes_ratio, "stakes": f_stakes,
        "quality_wins": f_quality,
        "own_network": network, "network_ratio": network_ratio, "network": f_network,
        "international": f_intl,
    })

    # Seed value = mean of the factors that are switched on (the ablation turns them off one at a time).
    enabled = [f for f in FACTOR_NAMES if cfg["factors"][f]["enabled"]]
    table["seed_value"] = table[enabled].mean(axis=1)
    table["seed_rating"] = rescale(table["seed_value"], cfg["seed_range"]["low"], cfg["seed_range"]["high"])
    table["league_offset"] = table["league"].map(offsets)
    table["start_rating"] = table["seed_rating"] + table["league_offset"]

    contributions = pd.concat([quality_rows, network_rows, intl_rows], ignore_index=True)
    return table, contributions

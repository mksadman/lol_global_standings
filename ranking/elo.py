"""
Stage 3: replay every series in date order with Elo, starting from the starting ratings.

    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    K   = base_K x format multiplier x international multiplier x age weight
    R_A += K x (result - E_A)        (result = 1 for a series win, 0 for a loss)

The rating change for each team in each series is its "H2H adjustment".
"""

import pandas as pd


def expected(r_a, r_b):
    """Probability that a team rated r_a beats a team rated r_b."""
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def k_factor(row, cfg):
    """K for one series: base K scaled by format, cross-league and age weight."""
    ec = cfg["elo"]
    k = ec["base_k"] * row.age_w
    if ec["format_multiplier_enabled"]:
        k *= ec["format_multipliers"][row.bo]
    if ec["international_multiplier_enabled"] and row.cross_league:
        k *= ec["international_multiplier"]
    return k


def replay(series, start_ratings, cfg):
    """
    Run Elo over the series (already sorted by start time).
    Returns (final ratings dict, log with one row per team per series).
    """
    ratings = dict(start_ratings)
    log = []
    for row in series.itertuples(index=False):
        w, l = row.winner, row.loser
        e_w = expected(ratings[w], ratings[l])
        k = k_factor(row, cfg)
        change = k * (1 - e_w)   # the winner gains exactly what the loser drops
        for team, opp, result, e, delta in ((w, l, "W", e_w, change), (l, w, "L", 1 - e_w, -change)):
            log.append({"series_id": row.series_id, "team": team, "opponent": opp, "result": result,
                        "rating_before": ratings[team], "expected": e, "k": k, "change": delta})
        ratings[w] += change
        ratings[l] -= change
    return ratings, pd.DataFrame(log)


def predict(ratings, series):
    """P(the series' recorded winner wins) for each series, using fixed (frozen) ratings."""
    return [expected(ratings[w], ratings[l]) for w, l in zip(series["winner"], series["loser"])]

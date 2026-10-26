"""
The whole model as one function (build_model), the plain-Elo baseline, and the validation:
holdout tests, ablation of each piece, and experiments that compare config variants.

Every variant is just a config override, so all of them run through exactly the same code.
"""

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .elo import predict, replay
from .factors import compute_factors
from .league_strength import fit_league_strength
from .load import add_age_weights, filter_before, season_cfg
from .placements import compute_placements


@dataclass
class Model:
    series: pd.DataFrame          # series used, with age weights
    offsets: dict                 # league -> fitted offset
    team_terms: dict              # team -> Stage 1 team term (fit only, not used in ratings)
    league_table: pd.DataFrame
    placements: pd.DataFrame
    factors: pd.DataFrame         # per team: factors, seed and starting rating
    contributions: pd.DataFrame   # every win's value per factor, with `counted` flags
    ratings: dict                 # final ratings after the replay
    h2h: pd.DataFrame             # rating change per team per series
    standings: pd.DataFrame


def build_model(series, home, cfg, override_path=None, unfinished=()):
    """
    Run Stages 1-3 on these series (the reference date is the last series in them).
    `unfinished` = title events still being played at a holdout cutoff (they earn no placement points).
    """
    series = add_age_weights(series, cfg)
    offsets, team_terms, league_table = fit_league_strength(series, cfg)
    placements = compute_placements(series, cfg, override_path, unfinished)
    factors, contributions = compute_factors(series, placements, home, offsets, cfg)
    ratings, h2h = replay(series, factors["start_rating"].to_dict(), cfg)
    standings = make_standings(series, factors, ratings, h2h)
    return Model(series, offsets, team_terms, league_table, placements, factors,
                 contributions, ratings, h2h, standings)


def make_standings(series, factors, ratings, h2h):
    """The global table: one row per team, ranked by final points."""
    wins = series["winner"].value_counts()
    losses = series["loser"].value_counts()
    table = factors[["league", "start_rating", "stakes", "quality_wins", "network", "international"]].copy()
    table["points"] = pd.Series(ratings)
    table["h2h_total"] = h2h.groupby("team")["change"].sum().reindex(table.index, fill_value=0.0)
    table["wins"] = wins.reindex(table.index, fill_value=0)
    table["losses"] = losses.reindex(table.index, fill_value=0)
    table = table.sort_values("points", ascending=False).rename_axis("team").reset_index()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table[["rank", "team", "league", "points", "start_rating", "h2h_total",
                  "stakes", "quality_wins", "network", "international", "wins", "losses"]]


def baseline_ratings(series, home, cfg):
    """Plain Elo: everyone starts at the same rating, same K rules (incl. age weight), no résumé."""
    series = add_age_weights(series, cfg)
    start = {team: cfg["elo"]["baseline_start"] for team in home}
    ratings, _ = replay(series, start, cfg)
    return ratings


def with_overrides(cfg, overrides):
    """A copy of the config with dotted keys replaced, e.g. {"age_weight.floor": 0.25}."""
    cfg = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        node = cfg
        *path, last = dotted.split(".")
        # YAML reads tier numbers as ints, so "tiers.1.x" has to look up the key 1, not "1".
        key_of = lambda n, k: int(k) if k not in n and k.isdigit() else k
        for key in path:
            node = node[key_of(node, key)]
        last = key_of(node, last)
        assert last in node, f"unknown config key {dotted}"
        node[last] = value
    return cfg


def seed_overrides(low, high):
    """Config override for a seed range."""
    return {"seed_range.low": low, "seed_range.high": high}


def scores(p):
    """Per-series log-loss (p = probability given to the actual winner) and accuracy."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    accuracy = np.where(p > 0.5, 1.0, np.where(p == 0.5, 0.5, 0.0))
    return -np.log(p), accuracy


def bootstrap_se(values, cfg):
    """Standard error of the mean of per-series values, by resampling series."""
    rng = np.random.default_rng(cfg["validation"]["bootstrap_seed"])
    idx = rng.integers(0, len(values), size=(cfg["validation"]["bootstrap_samples"], len(values)))
    return float(values[idx].mean(axis=1).std())


def select_test(all_series, holdout):
    """The held-out series: by tournament names or by international event key, optionally playoffs only."""
    if "event" in holdout:
        test = all_series[all_series["event"] == holdout["event"]]
    else:
        test = all_series[all_series["tournament"].isin(holdout["tournaments"])]
    if holdout.get("playoffs_only"):
        test = test[test["is_playoffs"] == 1]
    assert len(test), f"no series in the data for holdout {holdout['name']}"
    return test


def holdout_cutoff(holdout, test):
    """The configured cutoff, or else the start time of the first held-out series."""
    return pd.Timestamp(holdout["cutoff"]) if "cutoff" in holdout else test["start"].min()


def unfinished_events(all_series, cutoff):
    """Title events with series on both sides of the cutoff, i.e. still being played at the cutoff."""
    before = set(all_series.loc[all_series["start"] < cutoff, "title_event"])
    after = set(all_series.loc[all_series["start"] >= cutoff, "title_event"])
    return sorted(before & after)


def holdout_predictions(all_series, home, cfg, holdout, model=True):
    """
    Train on series before the cutoff, then predict every held-out series with the ratings
    frozen at the cutoff. Returns the probability given to each actual winner.
    """
    test = select_test(all_series, holdout)
    cutoff = holdout_cutoff(holdout, test)
    assert (test["start"] >= cutoff).all(), f"{holdout['name']}: held-out series start before the cutoff"
    train = filter_before(all_series, cutoff)
    if model:
        ratings = build_model(train, home, cfg, unfinished=unfinished_events(all_series, cutoff)).ratings
    else:
        ratings = baseline_ratings(train, home, cfg)
    return np.array(predict(ratings, test))


def experiment_holdouts(cfg, groups=None):
    """Holdouts in the given groups (default: the groups used to choose settings, never Worlds)."""
    groups = groups or cfg["validation"]["experiment_groups"]
    return [h for h in season_cfg(cfg)["holdouts"] if h["group"] in groups]


def holdout_losses(all_series, home, cfg, model=True, groups=None):
    """Per-series (group, log-loss, accuracy) for each holdout, keyed by holdout name."""
    return {h["name"]: (h["group"], *scores(holdout_predictions(all_series, home, cfg, h, model=model)))
            for h in experiment_holdouts(cfg, groups)}


def pool(losses):
    """Join per-series (log-loss, accuracy) by holdout group, plus 'all' for every holdout together."""
    pooled = {}
    for group, ll, acc in losses.values():
        pooled.setdefault(group, []).append((ll, acc))
    pooled["all"] = [(ll, acc) for _, ll, acc in losses.values()]
    return {g: (np.concatenate([ll for ll, _ in parts]), np.concatenate([acc for _, acc in parts]))
            for g, parts in pooled.items()}


def holdout_report(all_series, home, cfg, groups=None):
    """Log-loss and accuracy of the full model and the baseline on each holdout and each pooled group."""
    model = holdout_losses(all_series, home, cfg, groups=groups)
    base = holdout_losses(all_series, home, cfg, model=False, groups=groups)

    def row(label, ll_m, acc_m, ll_b, acc_b):
        return {"holdout": label, "series": len(ll_m),
                "model_logloss": ll_m.mean(), "baseline_logloss": ll_b.mean(),
                "diff (model-base)": (ll_m - ll_b).mean(), "diff_se": bootstrap_se(ll_m - ll_b, cfg),
                "model_acc": acc_m.mean(), "baseline_acc": acc_b.mean()}

    rows = [row(name, *model[name][1:], *base[name][1:]) for name in model]
    pooled_m, pooled_b = pool(model), pool(base)
    rows += [row(f"POOLED {g}", *pooled_m[g], *pooled_b[g]) for g in pooled_m]
    return pd.DataFrame(rows).round(3)


def pooled_scores(all_series, home, cfg, model=True):
    """Pooled log-loss for each holdout group and for all holdouts together, plus overall accuracy."""
    pooled = pool(holdout_losses(all_series, home, cfg, model=model))
    out = {f"{g}_logloss": ll.mean() for g, (ll, _) in pooled.items()}
    out["all_acc"] = pooled["all"][1].mean()
    return out


def compare_variants(all_series, home, cfg, variants):
    """
    Score each (label, overrides) variant on the holdouts. For each holdout group and for all of
    them together: pooled log-loss and its bootstrap SE. Then the gap between the model and the
    plain-Elo baseline over all holdouts (baseline run with the same config, so the same age
    weights and K rules), with the bootstrap SE of that paired gap.
    """
    rows = []
    for label, overrides in variants:
        c = with_overrides(cfg, overrides)
        model = pool(holdout_losses(all_series, home, c))
        base = pool(holdout_losses(all_series, home, c, model=False))
        row = {"variant": label}
        for g, (ll, _) in model.items():
            row[f"{g}_logloss"] = ll.mean()
            row[f"{g}_se"] = bootstrap_se(ll, cfg)
        gap = model["all"][0] - base["all"][0]
        row.update({"all_acc": model["all"][1].mean(), "baseline_all": base["all"][0].mean(),
                    "diff_all (model-base)": gap.mean(), "diff_se": bootstrap_se(gap, cfg)})
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def frozen_holdout_report(all_series, holdout, ratings, baseline, cfg):
    """
    Score FROZEN ratings (e.g. a snapshot's) on one holdout: nothing is rebuilt or refitted,
    so the test cannot influence any setting. `cfg` only supplies the bootstrap settings.
    """
    test = select_test(all_series, holdout)
    cutoff = holdout_cutoff(holdout, test)
    assert (test["start"] >= cutoff).all(), f"{holdout['name']}: held-out series start before the cutoff"
    unknown = (set(test["winner"]) | set(test["loser"])) - set(ratings)
    assert not unknown, f"teams in the test with no frozen rating: {sorted(unknown)}"
    ll_m, acc_m = scores(predict(ratings, test))
    ll_b, acc_b = scores(predict(baseline, test))
    return pd.DataFrame([{"holdout": holdout["name"], "series": len(ll_m),
                          "model_logloss": ll_m.mean(), "model_se": bootstrap_se(ll_m, cfg),
                          "baseline_logloss": ll_b.mean(), "diff (model-base)": (ll_m - ll_b).mean(),
                          "diff_se": bootstrap_se(ll_m - ll_b, cfg),
                          "model_acc": acc_m.mean(), "baseline_acc": acc_b.mean()}]).round(3)


ABLATIONS = {
    "full model": {},
    "no league offset": {"league_strength.enabled": False},
    "no Stage 1 team term": {"league_strength.team_term": False},
    "no Stakes factor": {"factors.stakes.enabled": False},
    "no Quality Wins factor": {"factors.quality_wins.enabled": False},
    "no Network factor": {"factors.network.enabled": False},
    "no International factor": {"factors.international.enabled": False},
    "network not age-weighted": {"factors.network.age_weighted": False},
    "no international K mult": {"elo.international_multiplier_enabled": False},
    "no format K mult": {"elo.format_multiplier_enabled": False},
    "no age weight": {"age_weight.enabled": False},
}


def ablation_report(all_series, home, cfg):
    """
    Turn each piece off in turn. `helps` = log-loss without the piece minus log-loss with it:
    positive means the piece improves predictions.
    """
    full = pooled_scores(all_series, home, cfg)
    rows = []
    for name, overrides in ABLATIONS.items():
        result = full if not overrides else pooled_scores(all_series, home, with_overrides(cfg, overrides))
        rows.append({"variant": name, **result, "helps": result["all_logloss"] - full["all_logloss"]})
    return pd.DataFrame(rows).round(3)


def seed_range_sweep(all_series, home, cfg):
    """Experiment 1: each seed range in validation.seed_range_sweep."""
    variants = [(f"seed {lo}-{hi}", seed_overrides(lo, hi)) for lo, hi in cfg["validation"]["seed_range_sweep"]]
    return compare_variants(all_series, home, cfg, variants)


def double_counting_test(all_series, home, cfg):
    """
    Experiment 2: does scaling domestic results by league multiplier AND adding the league offset
    count league strength twice? Rerun with every domestic multiplier flattened (offset still on).
    """
    v = cfg["validation"]
    flat = {f"tiers.{t}.domestic_multiplier": v["double_counting_multiplier"] for t in cfg["tiers"]}
    variants = []
    for lo, hi in v["double_counting_seed_ranges"]:
        variants.append((f"seed {lo}-{hi}, current multipliers", seed_overrides(lo, hi)))
        variants.append((f"seed {lo}-{hi}, all multipliers {v['double_counting_multiplier']}",
                         {**seed_overrides(lo, hi), **flat}))
    return compare_variants(all_series, home, cfg, variants)


def international_k_sweep(all_series, home, cfg):
    """Experiment 3: the cross-league K multiplier, at the default seed range."""
    variants = [(f"intl K x{m}", {"elo.international_multiplier": m})
                for m in cfg["validation"]["international_multiplier_sweep"]]
    return compare_variants(all_series, home, cfg, variants)


def floor_sweep(all_series, home, cfg):
    """Experiment 4: the age-weight floor, at the default seed range."""
    variants = [(f"floor {f}", {"age_weight.floor": f}) for f in cfg["validation"]["floor_sweep"]]
    return compare_variants(all_series, home, cfg, variants)


def compare_standings(all_series, home, cfg):
    """
    Experiment 5: full-season standings under each seed range, top N side by side.
    Also returns a summary: Spearman rank correlation over all teams and the biggest rank moves.
    """
    v = cfg["validation"]
    ranges = v["compare_standings_seed_ranges"]
    tables = {}
    for lo, hi in ranges:
        s = build_model(all_series, home, with_overrides(cfg, seed_overrides(lo, hi))).standings
        tables[f"{lo}-{hi}"] = s.set_index("team")
    labels = list(tables)
    n = v["compare_standings_top_n"]
    side = pd.DataFrame({"rank": range(1, n + 1)})
    for label in labels:
        top = tables[label].sort_values("rank").head(n)
        side[f"team ({label})"] = top.index
        side[f"pts ({label})"] = top["points"].round(0).astype(int).to_numpy()

    a, b = tables[labels[0]], tables[labels[1]].reindex(tables[labels[0]].index)
    moves = pd.DataFrame({"team": a.index, f"rank ({labels[0]})": a["rank"], f"rank ({labels[1]})": b["rank"],
                          "change": a["rank"] - b["rank"]})
    moves = moves.reindex(moves["change"].abs().sort_values(ascending=False).index).head(10).reset_index(drop=True)
    rho = spearmanr(a["rank"], b["rank"]).statistic
    common = len(set(side[f"team ({labels[0]})"]) & set(side[f"team ({labels[1]})"]))
    summary = (f"Spearman rank correlation over all {len(a)} teams: {rho:.3f}. "
               f"{common} of the top {n} teams are the same under both ranges.")
    return side, moves, summary

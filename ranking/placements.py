"""
Stage 2a: work out every team's placement in every title event, and the placement points it earns.

Placement rules, for the series of one title event:
  1. Champion = winner of the event's last series. Runner-up = its loser.
  2. Every other team that played an IsPlayoffs = 1 series is ranked by how late it was knocked out
     (calendar date of its last playoff series loss). Teams knocked out on the same day share a rank.
  3. Teams that never played an IsPlayoffs = 1 series rank below all of those, by series win rate
     (equal win rates share a rank).
If an event has no playoff series at all, everyone is ranked by win rate.

An event is finished once its final (a playoff series whose stage matches `final_stage`) has been
played. Until then a domestic event gives no placement points, and an international event gives each
team the share of the best finish it has already secured (`in_progress_placements` in the config).
"""

import re
from pathlib import Path

import pandas as pd

from .load import age_weight_for_dates, season_cfg


def competition_rank(keys, start_rank):
    """Rank a list of sort keys (already sorted best first) so equal keys share a rank: 1, 2, 2, 4."""
    ranks, prev = [], object()
    for i, key in enumerate(keys):
        ranks.append(start_rank + i if key != prev else ranks[-1])
        prev = key
    return ranks


def series_record(event):
    """Series wins and losses per team inside one event."""
    wins = event["winner"].value_counts()
    losses = event["loser"].value_counts()
    teams = sorted(set(wins.index) | set(losses.index))
    return pd.DataFrame({"wins": wins.reindex(teams, fill_value=0),
                         "losses": losses.reindex(teams, fill_value=0)})


def event_placements(event):
    """Placement (1 = champion) for every team in one title event's series."""
    record = series_record(event)
    record["win_rate"] = record["wins"] / (record["wins"] + record["losses"])
    playoffs = event[event["is_playoffs"] == 1]
    placements = {}

    if not playoffs.empty:
        final = playoffs.iloc[-1]   # series are sorted by start time, so this is the final
        placements[final["winner"]] = 1
        placements[final["loser"]] = 2
        # Everyone else in the bracket, latest knock-out first. A team with no playoff loss yet
        # (only possible if the bracket is unfinished) counts as knocked out last.
        bracket_teams = (set(playoffs["winner"]) | set(playoffs["loser"])) - set(placements)
        last_loss = playoffs.groupby("loser")["date"].max()
        out_dates = sorted(((last_loss.get(t, event["date"].max()), t) for t in bracket_teams), reverse=True)
        ranks = competition_rank([d for d, _ in out_dates], start_rank=3)
        placements.update({t: r for (_, t), r in zip(out_dates, ranks)})

    rest = record[~record.index.isin(placements)].sort_values("win_rate", ascending=False)
    ranks = competition_rank(list(rest["win_rate"]), start_rank=len(placements) + 1)
    placements.update(dict(zip(rest.index, ranks)))

    out = record.assign(placement=pd.Series(placements)).reset_index(names="team")
    out["played_playoffs"] = out["team"].isin(set(playoffs["winner"]) | set(playoffs["loser"]))
    return out.sort_values(["placement", "team"]).reset_index(drop=True)


def placement_share(placement, cfg):
    """Fraction of the event's base points a placement earns (1st 1.0, 2nd 0.6, 3rd-4th 0.4 ...)."""
    for worst_place, share in cfg["placement_shares"]["bands"]:
        if placement <= worst_place:
            return share
    return cfg["placement_shares"]["everyone_else"]


def is_finished(event, final_stage):
    """True once a playoff series of this event has a stage matching the final-stage regex."""
    final = event["stage"].map(lambda s: bool(re.search(final_stage, s))) & (event["is_playoffs"] == 1)
    return bool(final.any())


def secured_placements(event, table):
    """
    For an unfinished international event: each team's best secured finish, from the furthest stage
    it has played (the table's rows are ordered best first; the first row matching any of its stages).
    """
    record = series_record(event)
    rows = []
    for team in record.index:
        stages = set(event.loc[(event["winner"] == team) | (event["loser"] == team), "stage"])
        best = next(r for r in table if any(re.search(r["stage"], s) for s in stages))
        rows.append({"team": team, "secured": best["secured"], "share": best["share"]})
    out = record.reset_index(names="team").merge(pd.DataFrame(rows), on="team")
    out["placement"] = pd.NA
    out["played_playoffs"] = out["team"].isin(set(event.loc[event["is_playoffs"] == 1, "winner"])
                                              | set(event.loc[event["is_playoffs"] == 1, "loser"]))
    return out


def final_stages(cfg):
    """Final-stage regex for each domestic title event (its own, or the config default)."""
    return {te["name"]: te.get("final_stage", cfg["final_stage"]) for te in season_cfg(cfg)["title_events"]}


def event_info(cfg):
    """For each domestic title event name: (kind label, league code, base points)."""
    info = {}
    for te in season_cfg(cfg)["title_events"]:
        base = cfg["domestic"]["season_final_base_points" if te["season_final"] else "other_title_base_points"]
        info[te["name"]] = ("season final" if te["season_final"] else "domestic title", te["league"], base)
    return info


def apply_overrides(placements, path):
    """Replace placements with any hand fixes in the override CSV (columns: title_event, team, placement)."""
    if not path or not Path(path).exists():
        return placements
    overrides = pd.read_csv(path, encoding="utf-8")
    if overrides.empty:
        return placements
    key = ["title_event", "team"]
    merged = placements.merge(overrides[key + ["placement"]], on=key, how="left", suffixes=("", "_override"))
    fixed = merged["placement_override"].notna()
    merged.loc[fixed, "placement"] = merged.loc[fixed, "placement_override"].astype(int)
    merged["overridden"] = fixed
    return merged.drop(columns="placement_override")


def compute_placements(series, cfg, override_path=None, unfinished=()):
    """
    Placements and placement points for every team in every title event.

    points = base points x league multiplier (domestic only) x placement share
             x age weight of the event's last series date

    Domestic events whose final has not been played (and any title event in `unfinished`, which
    holdout runs pass for events still being played at their cutoff) earn no placement points.
    International events whose final has not been played earn the share of each team's secured finish.
    """
    ref_dates = series["date"].min(), series["date"].max()   # from ALL series, before dropping any
    series = series[~series["title_event"].isin(unfinished)]
    tiers = cfg["tiers"]
    league_mult = {code: tiers[v["tier"]]["domestic_multiplier"] for code, v in cfg["leagues"].items()}
    intl_cfg = cfg["international_events"]
    info = event_info(cfg)
    domestic_finals = final_stages(cfg)
    first_date, ref_date = ref_dates
    frames = []
    # groupby drops series with no title event (untitled tournaments), so they earn no points.
    for title, event in series.groupby("title_event", sort=False):
        is_intl = event["is_intl"].iloc[0]
        finished = is_finished(event, cfg["final_stage"] if is_intl else domestic_finals[title])
        if not finished and not is_intl:
            continue
        if finished:
            table = event_placements(event)
            table["status"], table["secured"], table["share"] = "final", "", float("nan")
        else:
            key = event["event"].iloc[0]
            tables = cfg["in_progress_placements"]
            table = secured_placements(event, tables.get(key, tables["default"]))
            table["status"] = "in progress"
        table["title_event"] = title
        table["last_date"] = event["date"].max()
        if is_intl:
            key = event["event"].iloc[0]
            table["kind"], table["event_league"] = intl_cfg[key]["label"], "INTL"
            table["base_points"], table["multiplier"] = intl_cfg[key]["base_points"], 1.0
        else:
            kind, league, base = info[title]
            table["kind"], table["event_league"] = kind, league
            table["base_points"], table["multiplier"] = base, league_mult[league]
        frames.append(table)

    placements = pd.concat(frames, ignore_index=True)
    placements["overridden"] = False
    placements = apply_overrides(placements, override_path)
    # Finished events (and hand-fixed placements) get their share from the placement bands;
    # in-progress international events already carry the share of each team's secured finish.
    from_bands = placements["share"].isna() | placements["overridden"]
    placements.loc[from_bands, "share"] = placements.loc[from_bands, "placement"].map(
        lambda p: placement_share(p, cfg))
    placements["age_w"] = age_weight_for_dates(placements["last_date"], first_date, ref_date, cfg).to_numpy()
    placements["points"] = (placements["base_points"] * placements["multiplier"]
                            * placements["share"] * placements["age_w"])
    cols = ["title_event", "kind", "event_league", "status", "last_date", "team", "placement", "secured", "overridden",
            "played_playoffs", "wins", "losses", "base_points", "multiplier", "share", "age_w", "points"]
    return placements[cols].sort_values(["last_date", "title_event", "placement"]).reset_index(drop=True)

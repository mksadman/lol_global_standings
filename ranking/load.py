"""
Read the game data and turn it into one row per series, tagged with everything later stages need.

Main entry point: prepare_series(cfg) -> DataFrame of series, sorted by start time.
Age weights depend on which series are being used, so they are added separately by add_age_weights().
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TEAM_COLS = ["Team1", "Team2", "WinTeam", "LossTeam"]


def load_config(path="config.yaml", season=None):
    """Read config.yaml into a plain dict, optionally switching to another season."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if season is not None:
        cfg["season"] = int(season)
    assert cfg["season"] in cfg["seasons"], f"season {cfg['season']} has no block under `seasons` in the config"
    return cfg


def season_cfg(cfg):
    """The config block for the season being run: input file, title events, holdouts, final test."""
    return cfg["seasons"][cfg["season"]]


def load_games(path):
    """Read the per-game CSV (UTF-8, because of names like LÖS) and parse the timestamps."""
    games = pd.read_csv(path, encoding="utf-8")
    games["DateTime UTC"] = pd.to_datetime(games["DateTime UTC"])
    return games


def apply_aliases(games, aliases):
    """Rename teams using the optional alias map from the config."""
    if aliases:
        games[TEAM_COLS] = games[TEAM_COLS].replace(aliases)
    return games


def stage_of(match_id):
    """The stage: the part of a MatchId between the first "_" and the last "_" ("Week 1", "Finals")."""
    return match_id[match_id.index("_") + 1:match_id.rindex("_")]


def collapse_to_series(games):
    """
    Turn games into series (one row per MatchId).

    The result comes from the LAST game's running score, not from counting game rows,
    because a few series are missing a game row. Team1/Team2 can swap sides between games,
    and the running score belongs to that row's Team1, so we read both off the same last row.
    """
    games = games.sort_values(["DateTime UTC", "GameId"])
    grouped = games.groupby("MatchId", sort=False)
    first = grouped.head(1).set_index("MatchId")
    last = grouped.tail(1).set_index("MatchId")

    series = pd.DataFrame(index=last.index)
    series["start"] = first["DateTime UTC"]
    series["date"] = series["start"].dt.normalize()
    series["tournament"] = last["Tournament"]
    series["stage"] = [stage_of(m) for m in last.index]
    series["league_name"] = last["League"]
    series["level"] = last["TournamentLevel"]
    series["is_playoffs"] = last["IsPlayoffs"].astype(int)
    series["bo"] = last["BO"]
    series["winner"] = last["WinTeam"]
    series["loser"] = last["LossTeam"]
    # The last game's winner also won the series, so its running score is the bigger one.
    winner_is_team1 = last["WinTeam"] == last["Team1"]
    series["winner_score"] = np.where(winner_is_team1, last["Team1Score"], last["Team2Score"]).astype(int)
    series["loser_score"] = np.where(winner_is_team1, last["Team2Score"], last["Team1Score"]).astype(int)
    series["n_games"] = grouped.size()

    assert (series["winner_score"] > series["loser_score"]).all(), "series winner must have the higher score"
    series = series.reset_index().rename(columns={"MatchId": "series_id"})
    return series.sort_values(["start", "series_id"]).reset_index(drop=True)


def assign_home_leagues(series, cfg):
    """
    Map every team to its domestic league code (LCK, LPL, ...).

    Fails loudly if a team plays in two domestic leagues or only appears at international events.
    """
    name_to_code = {v["name"]: code for code, v in cfg["leagues"].items()}
    domestic = series[series["league_name"].isin(name_to_code)]
    rows = pd.concat([
        domestic[["winner", "league_name"]].set_axis(["team", "league_name"], axis=1),
        domestic[["loser", "league_name"]].set_axis(["team", "league_name"], axis=1),
    ])
    leagues_per_team = rows.groupby("team")["league_name"].unique()
    multi = leagues_per_team[leagues_per_team.map(len) > 1]
    assert multi.empty, f"teams in more than one domestic league: {multi.to_dict()}"
    home = {team: name_to_code[names[0]] for team, names in leagues_per_team.items()}

    all_teams = set(series["winner"]) | set(series["loser"])
    intl_only = all_teams - set(home)
    assert not intl_only, f"teams that only appear at international events: {sorted(intl_only)}"
    return home


def tag_events(series, cfg, home):
    """
    Add the columns that describe each series' event:
      event        'domestic' or an international key (msi, ewc, ...)
      is_intl      True for international events (TournamentLevel = International)
      event_weight from the config: international weight, or domestic playoff / regular weight
      title_event  which placement group the series belongs to (empty for untitled tournaments)
      winner_league, loser_league, cross_league
    """
    intl_name_to_key = {v["name"]: key for key, v in cfg["international_events"].items()}
    season = season_cfg(cfg)
    tournament_to_title = {t: te["name"] for te in season["title_events"] for t in te["tournaments"]}
    untitled = set(season.get("untitled_tournaments", []))

    series = series.copy()
    series["is_intl"] = series["level"] == "International"
    unknown_intl = set(series.loc[series["is_intl"], "league_name"]) - set(intl_name_to_key)
    assert not unknown_intl, f"international events missing from config: {unknown_intl}"
    series["event"] = np.where(series["is_intl"], series["league_name"].map(intl_name_to_key), "domestic")

    intl_weight = series["event"].map({k: v["weight"] for k, v in cfg["international_events"].items()})
    domestic_weight = np.where(series["is_playoffs"] == 1,
                               cfg["domestic"]["playoff_weight"], cfg["domestic"]["regular_weight"])
    series["event_weight"] = np.where(series["is_intl"], intl_weight, domestic_weight)

    # Each international event is its own title event, named "<label> <season>", so an event split into
    # several Leaguepedia tournaments (e.g. a Worlds Play-In and Main Event) still counts once.
    # Domestic title events come from the season's config; untitled tournaments get none.
    labels = {k: v["label"] for k, v in cfg["international_events"].items()}
    intl_title = series["event"].map(labels) + f" {cfg['season']}"
    series["title_event"] = np.where(series["is_intl"], intl_title,
                                     series["tournament"].map(tournament_to_title))
    missing = set(series.loc[series["title_event"].isna(), "tournament"]) - untitled
    assert not missing, f"domestic tournaments in no title event and not listed as untitled: {sorted(missing)}"
    intl_untitled = set(series.loc[series["is_intl"], "tournament"]) & untitled
    assert not intl_untitled, f"untitled tournaments must be domestic, but these are international: {intl_untitled}"

    series["winner_league"] = series["winner"].map(home)
    series["loser_league"] = series["loser"].map(home)
    series["cross_league"] = series["winner_league"] != series["loser_league"]
    check_event_classes(series)
    return series


def check_event_classes(series):
    """
    Guard against mislabelled rows. Leaguepedia files qualifiers such as the LPL Regional Finals
    under the league "World Championship"; if one slips through un-relabelled, it would count as Worlds.
    An international tournament must contain at least one cross-league series.
    """
    intl = series[series["is_intl"]]
    one_league = intl.groupby("tournament")["cross_league"].sum() == 0
    assert not one_league.any(),         f"international tournaments with only same-league series (a mislabelled qualifier?): {list(one_league[one_league].index)}"


def prepare_series(cfg):
    """Load, alias, collapse and tag. This is everything that does not depend on the cutoff date."""
    games = apply_aliases(load_games(season_cfg(cfg)["games_csv"]), cfg.get("team_aliases"))
    series = collapse_to_series(games)
    home = assign_home_leagues(series, cfg)
    return tag_events(series, cfg, home), home


def age_weight_for_dates(dates, first_date, ref_date, cfg):
    """
    Age weight for each date: 1.0 within `full_weight_days` of the reference date,
    then falling in a straight line to `floor` at the first date in the data.
    """
    dates = pd.to_datetime(pd.Series(dates))
    if not cfg["age_weight"]["enabled"]:
        return pd.Series(1.0, index=dates.index)
    full = cfg["age_weight"]["full_weight_days"]
    floor = cfg["age_weight"]["floor"]
    days_ago = (ref_date - dates).dt.days
    span = (ref_date - first_date).days
    if span <= full:
        return pd.Series(1.0, index=dates.index)
    decayed = floor + (1 - floor) * (1 - (days_ago - full) / (span - full))
    return pd.Series(np.where(days_ago <= full, 1.0, decayed), index=dates.index)


def add_age_weights(series, cfg):
    """Add `age_w` to the series. The reference date is the last series in THIS data, not today."""
    series = series.copy()
    first_date, ref_date = series["date"].min(), series["date"].max()
    series["age_w"] = age_weight_for_dates(series["date"], first_date, ref_date, cfg).to_numpy()
    series.attrs["first_date"], series.attrs["ref_date"] = first_date, ref_date
    return series


def filter_before(series, cutoff):
    """Keep only series that started before the cutoff date (used by the holdout tests)."""
    return series[series["start"] < pd.Timestamp(cutoff)].reset_index(drop=True)


def age_table(series):
    """Smallest and largest age weight per tournament, in date order, for checking by eye."""
    table = series.groupby("tournament").agg(first=("date", "min"), last=("date", "max"),
                                             min_w=("age_w", "min"), max_w=("age_w", "max"))
    return table.sort_values("first").round({"min_w": 3, "max_w": 3})

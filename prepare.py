"""
Turn a raw Leaguepedia export into the cleaned games file the model reads.

    python prepare.py data/raw/games_2026.csv --out data/games_2026_clean.csv [--before 2026-10-26]

Rules (all lists and regexes live under `prepare` in config.yaml):
  a. keep TournamentLevel = Primary rows from the listed leagues; drop Promotion/Qualifier tournaments
  b. rename teams with `team_aliases`
  c. relabel tournaments listed in the season's `relabel_tournaments` (the LPL Regional Finals)
  d. TournamentLevel = International for the international leagues, Regional otherwise
  e. IsPlayoffs from the stage name, by the ordered `playoff_rules`
  f. BO = bo1/bo3/bo5 from the series winner's final running score
  g. checks: errors stop the run, warnings go into the report
"""

import argparse
import re

import pandas as pd

from ranking.load import load_config, season_cfg, stage_of

TEAM_COLS = ["Team1", "Team2", "WinTeam", "LossTeam"]
BO_BY_WINS = {1: "bo1", 2: "bo3", 3: "bo5"}


def playoff_flag(tournament, stage, cfg):
    """IsPlayoffs for one series by the first matching rule; None means no rule matched (fallback)."""
    for rule in cfg["prepare"]["playoff_rules"]:
        if all(re.search(rule[field], value) for field, value in
               (("tournament", tournament), ("stage", stage)) if field in rule):
            return rule["value"]
    return None


def select_rows(raw, cfg):
    """Rule a: Primary rows from the listed leagues, minus promotion and qualifier tournaments."""
    pc = cfg["prepare"]
    keep = (raw["TournamentLevel"] == pc["tournament_level"]) & raw["League"].isin(pc["leagues"])
    keep &= ~raw["Tournament"].str.contains(pc["drop_tournaments"], regex=True)
    return raw[keep].copy()


def series_formats(games):
    """Rule f: bo1/bo3/bo5 per game, from its series' final running score (works with missing rows)."""
    last = games.sort_values(["DateTime UTC", "GameId"]).groupby("MatchId").tail(1)
    wins = last[["Team1Score", "Team2Score"]].max(axis=1).astype(int)
    return games["MatchId"].map(dict(zip(last["MatchId"], wins.map(BO_BY_WINS))))


def clean(raw, cfg, before=None):
    """
    Apply rules a-f. Returns (cleaned games, list of (tournament, stage) that hit the playoff fallback).
    With `before`, only series that started before that date are kept.
    """
    pc = cfg["prepare"]
    games = select_rows(raw, cfg)
    games[TEAM_COLS] = games[TEAM_COLS].replace(cfg.get("team_aliases") or {})
    for tournament, fields in (season_cfg(cfg).get("relabel_tournaments") or {}).items():
        rows = games["Tournament"] == tournament
        for column, value in fields.items():
            games.loc[rows, column] = value
    games["TournamentLevel"] = games["League"].isin(pc["international_leagues"]).map(
        {True: "International", False: "Regional"})

    flags, fallback = [], set()
    for tournament, match_id in zip(games["Tournament"], games["MatchId"]):
        flag = playoff_flag(tournament, stage_of(match_id), cfg)
        if flag is None:
            flag = pc["playoff_fallback"]
            fallback.add((tournament, stage_of(match_id)))
        flags.append(flag)
    games["IsPlayoffs"] = flags
    games["BO"] = series_formats(games)

    # Keep each series' games together, with series ordered by when their first game started.
    games["_series_start"] = games.groupby("MatchId")["DateTime UTC"].transform("min")
    if before:
        games = games[games["_series_start"] < str(before)]
    games = games.sort_values(["_series_start", "MatchId", "DateTime UTC", "GameId"])
    return games.drop(columns="_series_start").reset_index(drop=True), sorted(fallback)


def check(games, cfg, previous=None):
    """
    Rule g. Stops on duplicate GameIds or a winner who is not one of the two teams.
    Returns warnings: international-only teams, series missing game rows, new teams and tournaments.
    """
    dupes = games.loc[games["GameId"].duplicated(), "GameId"].tolist()
    if dupes:
        raise SystemExit(f"ERROR: duplicate GameIds: {dupes[:10]}")
    bad = games[(games["WinTeam"] != games["Team1"]) & (games["WinTeam"] != games["Team2"])]
    if len(bad):
        raise SystemExit(f"ERROR: winner is not one of the two teams in games: {bad['GameId'].tolist()[:10]}")

    warnings = []
    teams = lambda g: set(g["Team1"]) | set(g["Team2"])
    intl = games["League"].isin(cfg["prepare"]["international_leagues"])
    intl_only = sorted(teams(games[intl]) - teams(games[~intl]))
    if intl_only:
        warnings.append(f"Teams that only appear at international events: {', '.join(intl_only)}")

    last = games.sort_values(["DateTime UTC", "GameId"]).groupby("MatchId").agg(
        rows=("GameId", "size"), score1=("Team1Score", "last"), score2=("Team2Score", "last"),
        team1=("Team1", "last"), team2=("Team2", "last"), tournament=("Tournament", "last"))
    short = last[last["rows"] < last["score1"] + last["score2"]]
    for match_id, r in short.iterrows():
        warnings.append(f"Series missing game rows: {r['team1']} vs {r['team2']} ({match_id}): "
                        f"{r['rows']} rows for a {r['score1']}-{r['score2']} final score")

    if previous is not None:
        for label, now, before in (("teams", teams(games), teams(previous)),
                                   ("tournaments", set(games["Tournament"]), set(previous["Tournament"]))):
            new = sorted(now - before)
            if new:
                warnings.append(f"New {label} since the previous update: {', '.join(new)}")
    return warnings


def report_text(raw_path, raw, games, fallback, warnings):
    """The run report as Markdown."""
    lines = [f"# Prepare report\n",
             f"- Raw file: `{raw_path}` ({len(raw)} rows)",
             f"- Cleaned: {len(games)} games, {games['MatchId'].nunique()} series, "
             f"{games['DateTime UTC'].min()} to {games['DateTime UTC'].max()}\n",
             "## Stages that hit the IsPlayoffs fallback (set to the fallback value)\n",
             "Check these when a new stage name appears.\n"]
    lines += [f"- {t} | {s}" for t, s in fallback] or ["- none"]
    lines += ["", "## Warnings\n"] + ([f"- {w}" for w in warnings] or ["- none"])
    return "\n".join(lines) + "\n"


def prepare(raw_path, cfg, before=None, previous=None):
    """Read a raw export, clean and check it. Returns (cleaned games, report text)."""
    raw = pd.read_csv(raw_path, encoding="utf-8")
    games, fallback = clean(raw, cfg, before)
    warnings = check(games, cfg, previous)
    return games, report_text(raw_path, raw, games, fallback, warnings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a raw Leaguepedia export")
    parser.add_argument("raw", help="raw export, e.g. data/raw/games_2026.csv")
    parser.add_argument("--out", required=True, help="where to write the cleaned games")
    parser.add_argument("--before", help="keep only series that started before this date (YYYY-MM-DD)")
    parser.add_argument("--previous", help="the previous update's cleaned file, to list new teams and tournaments")
    parser.add_argument("--season", type=int, help="season block to use (default: `season` in config.yaml)")
    args = parser.parse_args()
    cfg = load_config(season=args.season)
    previous = pd.read_csv(args.previous, encoding="utf-8") if args.previous else None
    games, report = prepare(args.raw, cfg, args.before, previous)
    games.to_csv(args.out, index=False, encoding="utf-8")
    print(report)
    print(f"Saved {len(games)} games -> {args.out}")

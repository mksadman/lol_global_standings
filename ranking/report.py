"""
Write the outputs: global standings, regional tables, per-team detail pages, league strength, placements.
"""

import re
from pathlib import Path

import pandas as pd

FACTOR_COLS = ["stakes", "quality_wins", "network", "international"]


def md_table(df):
    """Render a DataFrame as a Markdown table (no extra dependency like tabulate)."""
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def display_standings(standings, movement=None):
    """
    Standings rounded for reading, with a W-L series record column. With `movement` (indexed by
    team: movement, rank_change, points_change), those columns go right after the team's league.
    """
    out = standings.copy()
    for col in ["points", "start_rating", "h2h_total"]:
        out[col] = out[col].round(0).astype(int)
    out[FACTOR_COLS] = out[FACTOR_COLS].round(3)
    out["record"] = out["wins"].astype(str) + "-" + out["losses"].astype(str)
    out = out.drop(columns=["wins", "losses"])
    if movement is not None:
        moves = movement[["movement", "rank_change", "points_change"]].reindex(out["team"]).reset_index(drop=True)
        moves["points_change"] = moves["points_change"].round(1)
        moves["rank_change"] = moves["rank_change"].astype("Int64")
        moves = moves.astype(object).where(moves.notna(), "")   # a new team has no previous rank
        at = out.columns.get_loc("league") + 1
        for i, col in enumerate(moves.columns):
            out.insert(at + i, col, moves[col])
    return out


def regional_tables(standings):
    """The global table filtered to each league and renumbered. Points are NOT recalculated."""
    tables = {}
    for league, table in standings.groupby("league", sort=False):
        table = table.rename(columns={"rank": "global_rank"}).copy()
        table.insert(0, "regional_rank", range(1, len(table) + 1))
        tables[league] = table.reset_index(drop=True)
    return tables


def slug(name):
    """A file-name-safe version of a team name (keeps letters like Ö)."""
    return re.sub(r"[^\w]+", "-", name.lower()).strip("-")


def write_global(standings, out_dir, movement=None):
    table = display_standings(standings, movement)
    table.to_csv(out_dir / "global_standings.csv", index=False, encoding="utf-8")
    text = "# Global standings, 2026 season\n\n" + md_table(table) + "\n"
    (out_dir / "global_standings.md").write_text(text, encoding="utf-8")


def write_regional(standings, out_dir, movement=None):
    regional_dir = out_dir / "regional"
    regional_dir.mkdir(parents=True, exist_ok=True)
    for league, table in regional_tables(display_standings(standings, movement)).items():
        text = f"# {league} standings (global points)\n\n" + md_table(table) + "\n"
        (regional_dir / f"{league}.md").write_text(text, encoding="utf-8")


def team_series_table(model, team):
    """Every series the team played, with factor-list membership and its H2H change."""
    s = model.series.set_index("series_id")
    h2h = model.h2h[model.h2h["team"] == team].set_index("series_id")
    counted = model.contributions[(model.contributions["team"] == team) & model.contributions["counted"]]
    lists = counted.groupby("series_id")["factor"].apply(
        lambda f: ", ".join({"quality_wins": "QW", "network": "NW", "international": "IW"}[x] for x in f))
    rows = []
    for sid, h in h2h.iterrows():
        row = s.loc[sid]
        won = h["result"] == "W"
        score = f"{row['winner_score']}-{row['loser_score']}" if won else f"{row['loser_score']}-{row['winner_score']}"
        rows.append({"date": row["date"].date(), "event": row["tournament"], "opponent": h["opponent"],
                     "format": row["bo"], "result": f"{h['result']} {score}",
                     "event_weight": row["event_weight"], "age_w": round(row["age_w"], 3),
                     "factor_lists": lists.get(sid, ""), "h2h_change": round(h["change"], 1)})
    return pd.DataFrame(rows)


def factor_section(model, team, factor, title, explain):
    """The wins that made up one best-N factor, best first."""
    c = model.contributions
    rows = c[(c["team"] == team) & (c["factor"] == factor) & c["counted"]]
    s = model.series.set_index("series_id")
    table = pd.DataFrame({
        "date": [s.loc[i, "date"].date() for i in rows["series_id"]],
        "event": [s.loc[i, "tournament"] for i in rows["series_id"]],
        "beaten": rows["opponent"].to_numpy(),
        "opponent_value": rows["opponent_value"].round(3).to_numpy(),
        "event_weight": rows["event_weight"].to_numpy(),
        "age_w": rows["age_w"].round(3).to_numpy(),
        "value": rows["value"].round(3).to_numpy(),
    })
    value = model.factors.loc[team, factor]
    body = md_table(table) if len(table) else "_No qualifying wins._"
    return f"## {title}: {value:.3f}\n\n{explain}\n\n{body}\n"


def write_details(model, cfg, out_dir):
    """One Markdown page per team, in the style of Valve's details pages."""
    details_dir = out_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    fc = cfg["factors"]
    for row in model.standings.itertuples(index=False):
        team, f = row.team, model.factors.loc[row.team]
        places = model.placements[model.placements["team"] == team][
            ["title_event", "kind", "placement", "base_points", "multiplier", "share", "age_w", "points"]].round(3)
        beaten = sorted(model.series.loc[model.series["winner"] == team, "loser"].unique())
        parts = [
            f"# {row.rank}. {team} ({row.league})\n",
            f"- Starting rating: **{row.start_rating:.0f}** "
            f"(seed {f['seed_rating']:.0f} + league offset {f['league_offset']:+.0f})",
            f"- H2H adjustment: **{row.h2h_total:+.0f}**",
            f"- Final points: **{row.points:.0f}**",
            f"- Series record: {row.wins}-{row.losses}\n",
            f"## Stakes Earned: {f['stakes']:.3f}\n\n"
            f"Total placement points {f['stakes_total']:.1f}; ratio to rank-{fc['reference_rank']} team "
            f"{f['stakes_ratio']:.3f}{' then curve()' if fc['stakes']['curve'] else ''}.\n\n"
            + (md_table(places) if len(places) else "_No placements._") + "\n",
            factor_section(model, team, "quality_wins", "Quality Wins",
                           f"Best {fc['quality_wins']['best_n']} wins by beaten team's Stakes ratio x event weight x "
                           f"age weight, averaged{' then curve()' if fc['quality_wins']['curve'] else ''}."),
            factor_section(model, team, "network", "Weighted Network",
                           f"Own network {f['own_network']:.3f} (ratio {f['network_ratio']:.3f}) from "
                           f"{len(beaten)} distinct beaten teams: {', '.join(beaten) or 'none'}.\n\n"
                           f"Best {fc['network']['best_n']} wins by beaten team's network ratio x event weight x age weight."),
            factor_section(model, team, "international", "International Wins",
                           f"Best {fc['international']['best_n']} international wins by event weight x age weight."),
            "## Series\n\nFactor lists: QW = Quality Wins, NW = Weighted Network, IW = International Wins.\n\n"
            + md_table(team_series_table(model, team)) + "\n",
        ]
        (details_dir / f"{row.rank:02d}--{slug(team)}.md").write_text("\n".join(parts), encoding="utf-8")


def write_all(model, cfg, out_dir, movement=None):
    """Write every output file into out_dir. `movement` adds change-since-last-update columns to the standings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_global(model.standings, out_dir, movement)
    write_regional(model.standings, out_dir, movement)
    write_details(model, cfg, out_dir)
    model.league_table.to_csv(out_dir / "league_strength.csv", index=False, encoding="utf-8")
    placements = model.placements.copy()
    placements["last_date"] = placements["last_date"].dt.date
    placements.round(3).to_csv(out_dir / "placements.csv", index=False, encoding="utf-8")

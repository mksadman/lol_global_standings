"""
The power ranking update, every two weeks: python main.py --date YYYY-MM-DD [--raw FILE] [--force]

  1. check the date is on the schedule (update.anchor_date + a multiple of update.interval_days)
  2. clean the newest raw export in data/raw/ (prepare.py), keeping only series before the date
  3. rebuild the ranking from all season data with the current settings
  4. compare it with the previous biweekly snapshot (or the frozen baseline for the first one)
  5. write everything to output/biweekly/<date>/ and freeze it

Every team's points change splits exactly into four parts:
  series since the last update   rating changes from every series played since the previous snapshot
  league offset        change in its league's Stage 1 offset
  résumé seed          change in its seed (placements, factors, and the age weights moving on)
  earlier games        change in what its older series gave in the replay (new starting ratings
                       and age weights change every earlier Elo update a little)
"""

import copy
import shutil
from pathlib import Path

import pandas as pd
import yaml

from prepare import prepare

from .load import prepare_series, season_cfg
from .report import md_table, write_all
from .snapshot import write_snapshot_files
from .validate import baseline_ratings, build_model, holdout_report


def update_root(cfg):
    return Path(cfg["data"]["output_dir"]) / cfg["update"]["dir"]


def on_schedule(cfg, date):
    """True if the date is the anchor date plus a whole number of update intervals (before or after)."""
    days = (pd.Timestamp(date) - pd.Timestamp(cfg["update"]["anchor_date"])).days
    return days % cfg["update"]["interval_days"] == 0


def newest_raw(cfg):
    """The most recently modified CSV in data/raw/ (skipping a half-finished fetch's .partial.csv)."""
    files = sorted((p for p in Path(cfg["data"]["raw_dir"]).glob("*.csv") if not p.name.endswith(".partial.csv")),
                   key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No raw export found in {cfg['data']['raw_dir']}/")
    return files[-1]


def previous_snapshot(cfg, date):
    """The latest biweekly snapshot before this date, else the baseline. Returns (name, folder, games csv)."""
    earlier = sorted(p.name for p in update_root(cfg).glob("*")
                     if p.is_dir() and not p.name.startswith(".") and p.name < date)
    if earlier:
        folder = update_root(cfg) / earlier[-1]
        return f"{cfg['update']['dir']}/{earlier[-1]}", folder, folder / "games_clean.csv"
    name = cfg["update"]["baseline_snapshot"]
    return name, Path(cfg["data"]["output_dir"]) / name, Path(cfg["update"]["baseline_games_csv"])


def previous_state(folder):
    """
    A snapshot's per-team numbers (rank, points, start, seed, offset, h2h total), its league offsets
    and the date from which games count as new. Older snapshots lack the seed and offset columns;
    those are rebuilt from league_strength.csv (start = seed + offset).
    """
    ratings = pd.read_csv(folder / "ratings.csv", encoding="utf-8").set_index("team")
    offsets = pd.read_csv(folder / "league_strength.csv", encoding="utf-8").set_index("league")["fitted_offset"]
    if "league_offset" not in ratings:
        ratings["league_offset"] = ratings["league"].map(offsets)
        ratings["seed_rating"] = ratings["start_rating"] - ratings["league_offset"]
        ratings["h2h_total"] = ratings["points"] - ratings["start_rating"]
    with open(folder / "snapshot_info.yaml", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    cutoff = info.get("data_cutoff") or str((pd.Timestamp(info["last_series_date"]) + pd.Timedelta(days=1)).date())
    return ratings, offsets, cutoff


def series_text(model, team, new_ids):
    """Every series a team played since the last update, e.g. 'W 3-1 vs Gen.G (+12.0); L 0-1 vs T1 (-5.2)'."""
    s = model.series.set_index("series_id")
    h = model.h2h[(model.h2h["team"] == team) & model.h2h["series_id"].isin(new_ids)]
    parts = []
    for row in h.itertuples(index=False):
        r = s.loc[row.series_id]
        score = f"{r['winner_score']}-{r['loser_score']}" if row.result == "W" else f"{r['loser_score']}-{r['winner_score']}"
        parts.append(f"{row.result} {score} vs {row.opponent} ({row.change:+.1f})")
    return "; ".join(parts)


def movement_table(model, prev, new_ids):
    """Per team: rank move, points change and the four parts it splits into."""
    now = model.standings.set_index("team")
    f = model.factors
    out = pd.DataFrame(index=now.index)
    out["rank"] = now["rank"]
    out["prev_rank"] = prev["rank"].reindex(out.index)
    out["rank_change"] = out["prev_rank"] - out["rank"]          # positive = moved up
    out["movement"] = [("new" if pd.isna(c) else "same" if c == 0 else f"up {int(c)}" if c > 0 else f"down {int(-c)}")
                       for c in out["rank_change"]]
    out["points"] = now["points"]
    out["points_change"] = now["points"] - prev["points"].reindex(out.index)
    since_last = model.h2h[model.h2h["series_id"].isin(new_ids)].groupby("team")["change"].sum()
    out["since_last_update"] = since_last.reindex(out.index, fill_value=0.0)
    out["league_offset_change"] = f["league_offset"] - prev["league_offset"].reindex(out.index)
    out["seed_change"] = f["seed_rating"] - prev["seed_rating"].reindex(out.index)
    out["earlier_games_change"] = (out["points_change"] - out["since_last_update"]
                                   - out["league_offset_change"] - out["seed_change"])
    out["series"] = [series_text(model, t, new_ids) for t in out.index]
    return out.sort_values("rank")


PARTS = {"since_last_update": "series since the last update", "league_offset_change": "league offset",
         "seed_change": "résumé seed", "earlier_games_change": "re-run of earlier games"}


def main_reason(row):
    """The biggest of the four parts, in words."""
    if pd.isna(row["points_change"]):
        return "new since the last update"
    part = max(PARTS, key=lambda p: abs(row[p]))
    return f"{PARTS[part]} ({row[part]:+.1f})"


def changes_md(date, prev_name, cutoff, model, moves, prev_offsets, new_ids):
    """output/biweekly/<date>/changes.md: short, plain, every team."""
    new = model.series[model.series["series_id"].isin(new_ids)]
    events = ", ".join(f"{t} ({n})" for t, n in new.groupby("tournament").size().items()) or "none"
    lines = [f"# Power ranking changes: update of {date}\n",
             f"Compared with `{prev_name}` (games before {cutoff}). Series since the last update: "
             f"{len(new)}: {events}.\n",
             "Points change = series since the last update + league offset + résumé seed + re-run of earlier games. "
             "The last part is the small drift in older series when starting ratings and age weights move.\n"]

    moved = moves.dropna(subset=["points_change"])
    up = moved.sort_values("rank_change", ascending=False).head(5)
    down = moved.sort_values("rank_change").head(5)
    lines.append("## Biggest movers\n")
    for label, table in (("Up", up), ("Down", down)):
        for team, r in table.iterrows():
            if r["rank_change"] != 0:
                lines.append(f"- {label}: **{team}** {int(r['prev_rank'])} -> {int(r['rank'])} "
                             f"({r['points_change']:+.1f} points; main reason: {main_reason(r)})")
    lines.append("")

    offsets = pd.DataFrame({"before": prev_offsets, "now": pd.Series(model.offsets).round(1)})
    offsets["change"] = (offsets["now"] - offsets["before"]).round(1)
    lines += ["## League offsets\n", md_table(offsets.reset_index(names="league")), ""]

    table = pd.DataFrame({
        "rank": moves["rank"], "move": moves["movement"], "team": moves.index,
        "points": moves["points"].round(0).astype(int), "points change": moves["points_change"].round(1),
        "series since last update": moves["since_last_update"].round(1),
        "league offset": moves["league_offset_change"].round(1), "résumé seed": moves["seed_change"].round(1),
        "earlier games": moves["earlier_games_change"].round(1),
        "main reason": [main_reason(r) for _, r in moves.iterrows()],
        "every series since the last update": moves["series"]})
    lines += ["## All teams\n", md_table(table), ""]
    return "\n".join(lines)


def run_update(cfg, date, raw=None, force=False):
    date = str(pd.Timestamp(date).date())
    if not on_schedule(cfg, date) and not force:
        u = cfg["update"]
        raise SystemExit(f"{date} is not an update date (every {u['interval_days']} days from {u['anchor_date']}). "
                         f"Pass --force to run it anyway.")
    final_dir = update_root(cfg) / date
    if final_dir.exists() and not force:
        raise SystemExit(f"Snapshot {final_dir} already exists and is frozen. Pass --force to overwrite it.")
    # Build in a temporary folder and move it into place at the end, so a failed run leaves nothing half-written.
    work = update_root(cfg) / f".tmp-{date}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    raw_path = Path(raw) if raw else newest_raw(cfg)
    prev_name, prev_dir, prev_games_csv = previous_snapshot(cfg, date)
    prev_games = pd.read_csv(prev_games_csv, encoding="utf-8") if prev_games_csv.exists() else None
    games, report = prepare(raw_path, cfg, before=date, previous=prev_games)
    games.to_csv(work / "games_clean.csv", index=False, encoding="utf-8")
    (work / "prepare_report.md").write_text(report, encoding="utf-8")
    print(report)

    run_cfg = copy.deepcopy(cfg)
    season_cfg(run_cfg)["games_csv"] = str(work / "games_clean.csv")
    series, home = prepare_series(run_cfg)
    model = build_model(series, home, run_cfg, override_path=season_cfg(cfg).get("placements_override_csv"))

    prev, prev_offsets, cutoff = previous_state(prev_dir)
    new_ids = set(series.loc[series["start"] >= pd.Timestamp(cutoff), "series_id"])
    moves = movement_table(model, prev, new_ids)
    write_all(model, run_cfg, work, moves)
    (work / "changes.md").write_text(changes_md(date, prev_name, cutoff, model, moves, prev_offsets, new_ids),
                                     encoding="utf-8")

    # Prediction holdouts: a sanity check only, never used to tune anything.
    sanity = holdout_report(series, home, run_cfg)
    (work / "validation.md").write_text(
        f"# Sanity check, update of {date}\n\nPrediction holdouts (not used for tuning; Worlds is not included).\n\n"
        + md_table(sanity) + "\n", encoding="utf-8")

    season_cfg(run_cfg)["games_csv"] = str(final_dir / "games_clean.csv")
    extra = {"date": date, "data_cutoff": date, "raw_file": str(raw_path),
             "raw_rows": int(len(pd.read_csv(raw_path, encoding="utf-8"))), "compared_with": prev_name}
    write_snapshot_files(work, model, baseline_ratings(series, home, run_cfg), run_cfg,
                         final_dir / "games_clean.csv", len(games), extra)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    work.rename(final_dir)
    print(f"\n=== Top 20, update of {date} (vs {prev_name}) ===")
    cols = ["rank", "movement", "points", "points_change"]
    print(moves[cols].head(20).round(1).to_string())
    print(f"\nSnapshot frozen in {final_dir}/")
    return final_dir

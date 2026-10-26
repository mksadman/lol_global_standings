"""
The command line. Every run writes a frozen snapshot folder under output/.

    python main.py --date 2026-10-26                biweekly update (every 14 days from update.anchor_date):
                                                    newest export in data/raw/, games before that date,
                                                    frozen in output/biweekly/2026-10-26/
    python main.py --date DATE --raw FILE           the same with a specific raw export
    python main.py --date DATE --force              allow a date off the schedule, or overwrite a snapshot
    python main.py --snapshot NAME                  full-season ranking of the season's games_csv plus every
                                                    validation experiment, frozen in output/NAME/
    python main.py --final-test                     phase 2: score the season's frozen snapshot on Worlds
    python main.py --season 2027 ...                use another season's block in config.yaml
"""

import argparse
import copy

import pandas as pd

from ranking.load import age_table, load_config, prepare_series, season_cfg
from ranking.report import display_standings, md_table, write_all
from ranking.snapshot import load_snapshot, prepare_snapshot_dir, snapshot_dir, write_snapshot_files
from ranking.update import run_update
from ranking.validate import (ablation_report, baseline_ratings, build_model, compare_standings,
                              double_counting_test, floor_sweep, frozen_holdout_report, holdout_report,
                              international_k_sweep, seed_range_sweep)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 100)


def parse_args():
    parser = argparse.ArgumentParser(description="LoL global team ranking")
    parser.add_argument("--snapshot", help="write a frozen snapshot to output/<NAME>/")
    parser.add_argument("--force", action="store_true",
                        help="allow overwriting a frozen snapshot or final test, or an off-schedule --date")
    parser.add_argument("--season", type=int, help="season to run (default: `season` in config.yaml)")
    parser.add_argument("--date", help="power ranking update using games before this date (YYYY-MM-DD)")
    parser.add_argument("--raw", help="raw Leaguepedia export for --date (default: newest file in data/raw/)")
    parser.add_argument("--final-test", action="store_true",
                        help="score the season's frozen snapshot on its final-test holdout (Worlds)")
    return parser.parse_args()


def section(title, table, log):
    """Print a table and keep a Markdown copy for validation.md."""
    print(f"\n=== {title} ===\n{table.to_string(index=False)}")
    log.append(f"## {title}\n\n{md_table(table)}\n")


def validation_log(series, home, cfg):
    """Every validation table, in the order they appear in validation.md."""
    log = []
    s = cfg["seed_range"]
    settings = (f"seed range {s['low']}-{s['high']}, international K x{cfg['elo']['international_multiplier']}, "
                f"floor {cfg['age_weight']['floor']}")
    section(f"Final validation numbers ({settings})", holdout_report(series, home, cfg), log)
    log.append("All experiments train only on data before each cutoff and predict with frozen ratings. "
               "Groups: `international` = MSI + EWC; `domestic` = the six season-final playoffs, each "
               "trained on everything before that playoff started; `all` = both together. Worlds is not "
               "used. `*_se` is the bootstrap SE of that pooled log-loss; `diff_se` is the bootstrap SE "
               "of the paired gap to the plain-Elo baseline over all holdouts. Experiments change one "
               "setting at a time from the defaults above.\n")
    section("1. Seed range grid", seed_range_sweep(series, home, cfg), log)
    section("2. Double counting: domestic multipliers flattened (league offset still on)",
            double_counting_test(series, home, cfg), log)
    section("3. International K multiplier", international_k_sweep(series, home, cfg), log)
    section("4. Age-weight floor", floor_sweep(series, home, cfg), log)
    side, moves, summary = compare_standings(series, home, cfg)
    section("5. Full-season top 20 side by side", side, log)
    section("5b. Biggest rank changes between the two seed ranges", moves, log)
    print(summary)
    log.append(summary + "\n")
    section("Ablation (helps > 0 means the piece improves all-holdout log-loss)",
            ablation_report(series, home, cfg), log)
    return log


def run_ranking(cfg, snapshot, force=False):
    """Build the full-season ranking, run every validation experiment and freeze it as output/<snapshot>/."""
    # Check the snapshot folder first, so a refused overwrite fails before the long run.
    out_dir = prepare_snapshot_dir(cfg, snapshot, force)
    season = season_cfg(cfg)
    series, home = prepare_series(cfg)
    n_rows = len(pd.read_csv(season["games_csv"], encoding="utf-8"))
    print(f"Season {cfg['season']}: {season['games_csv']} ({n_rows} game rows), {len(series)} series, "
          f"{len(home)} teams, {series['date'].min().date()} to {series['date'].max().date()}")

    model = build_model(series, home, cfg, override_path=season.get("placements_override_csv"))
    write_all(model, cfg, out_dir)

    ages = age_table(model.series).reset_index()
    ages[["first", "last"]] = ages[["first", "last"]].apply(lambda c: c.dt.date)
    print(f"\n=== Age weight per tournament ===\n{ages.to_string(index=False)}")
    print(f"\n=== Top 20 ===\n{display_standings(model.standings).head(20).to_string(index=False)}")
    print(f"\n=== League strength (Stage 1) ===\n{model.league_table.to_string(index=False)}")

    log = validation_log(series, home, cfg)
    (out_dir / "validation.md").write_text(f"# Validation — snapshot {snapshot}\n\n" + "\n".join(log),
                                           encoding="utf-8")
    write_snapshot_files(out_dir, model, baseline_ratings(series, home, cfg), cfg, season["games_csv"], n_rows)
    print(f"\nSnapshot '{snapshot}' frozen in {out_dir}/")


def run_final_test(cfg, force=False):
    """
    Phase 2: predict the final-test holdout (Worlds) with the frozen snapshot's ratings.
    The snapshot's own config is used for reading and tagging the data; nothing is refitted.
    """
    final = season_cfg(cfg)["final_test"]
    if not final["enabled"]:
        raise SystemExit("The final test is switched off (final_test.enabled: false in config.yaml).")
    snap_cfg, info, ratings, baseline = load_snapshot(cfg, final["snapshot"])
    holdout = final["holdout"]
    assert pd.Timestamp(info["last_series_date"]) < pd.Timestamp(holdout["cutoff"]), \
        "the snapshot contains games on or after the final-test cutoff"

    # Read the CURRENT data file (now with Worlds) using the snapshot's settings.
    read_cfg = copy.deepcopy(snap_cfg)
    season_cfg(read_cfg)["games_csv"] = final.get("games_csv", season_cfg(cfg)["games_csv"])
    series, _ = prepare_series(read_cfg)
    before = (series["start"] < pd.Timestamp(holdout["cutoff"])).sum()
    if before != info["series"]:
        print(f"WARNING: the data now has {before} series before the cutoff, the snapshot had {info['series']}.")

    out = snapshot_dir(cfg, final["snapshot"]) / "final_test.md"
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists. Pass --force to overwrite it.")
    table = frozen_holdout_report(series, holdout, ratings, baseline, snap_cfg)
    print(f"\n=== FINAL TEST: {holdout['name']} with frozen snapshot {final['snapshot']} ===\n"
          f"{table.to_string(index=False)}")
    out.write_text(f"# Final test: {holdout['name']}\n\nRatings frozen in snapshot `{final['snapshot']}` "
                   f"(data to {info['last_series_date']}). Nothing was refitted.\n\n{md_table(table)}\n",
                   encoding="utf-8")


def main():
    args = parse_args()
    cfg = load_config(season=args.season)
    if args.date:
        run_update(cfg, args.date, raw=args.raw, force=args.force)
    elif args.final_test:
        run_final_test(cfg, force=args.force)
    elif args.snapshot:
        run_ranking(cfg, args.snapshot, force=args.force)
    else:
        raise SystemExit("Nothing to do. Usage: python main.py --date YYYY-MM-DD  (see python main.py --help)")


if __name__ == "__main__":
    main()

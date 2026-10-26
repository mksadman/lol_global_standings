"""
Frozen snapshots of a ranking.

A snapshot is a folder output/<name>/ holding every output file plus what is needed to reuse it later:
  config_used.yaml     the exact config the ranking was built with
  snapshot_info.yaml   input file name, its row count, series count and date range, creation time
  ratings.csv          every team's final points, starting rating and its parts (machine-readable)
  baseline_ratings.csv the plain-Elo baseline's ratings on the same data, for the final test

Once written a snapshot is frozen: writing it again is refused unless force=True.
Phase 2 (the Worlds test) reads a snapshot's ratings and config instead of rebuilding anything.
Phase 3 (the next season) can read the final snapshot's ratings with load_snapshot().
"""

import datetime as dt
import shutil
from pathlib import Path

import pandas as pd
import yaml


def snapshot_dir(cfg, name):
    return Path(cfg["data"]["output_dir"]) / name


def prepare_snapshot_dir(cfg, name, force=False):
    """Create an empty snapshot folder, refusing to touch an existing one unless forced."""
    path = snapshot_dir(cfg, name)
    if path.exists():
        if not force:
            raise SystemExit(f"Snapshot '{name}' already exists at {path} and is frozen. "
                             f"Pass --force to overwrite it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def write_snapshot_files(path, model, baseline, cfg, input_csv, n_rows, extra_info=None):
    """Save the config, input details and ratings next to the normal outputs."""
    with open(path / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    info = {
        "season": cfg["season"],
        "input_file": str(input_csv),
        "input_rows": int(n_rows),
        "series": int(len(model.series)),
        "first_series_date": str(model.series["date"].min().date()),
        "last_series_date": str(model.series["date"].max().date()),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **(extra_info or {}),
    }
    with open(path / "snapshot_info.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(info, f, sort_keys=False, allow_unicode=True)
    ratings = model.standings[["rank", "team", "league", "start_rating", "points", "h2h_total"]].merge(
        model.factors[["seed_rating", "league_offset"]], left_on="team", right_index=True)
    ratings.to_csv(path / "ratings.csv", index=False, encoding="utf-8")
    pd.Series(baseline, name="points").rename_axis("team").reset_index().to_csv(
        path / "baseline_ratings.csv", index=False, encoding="utf-8")


def load_snapshot(cfg, name):
    """Read a frozen snapshot back: (its config, its info, final ratings dict, baseline ratings dict)."""
    path = snapshot_dir(cfg, name)
    assert path.exists(), f"snapshot '{name}' not found at {path}"
    with open(path / "config_used.yaml", encoding="utf-8") as f:
        snap_cfg = yaml.safe_load(f)
    with open(path / "snapshot_info.yaml", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    ratings = pd.read_csv(path / "ratings.csv", encoding="utf-8").set_index("team")["points"].to_dict()
    baseline = pd.read_csv(path / "baseline_ratings.csv", encoding="utf-8").set_index("team")["points"].to_dict()
    return snap_cfg, info, ratings, baseline

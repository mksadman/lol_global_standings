"""Checks on prepare.py, in-progress placements and the biweekly update. Run: python -m pytest tests"""

import copy
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # config paths are relative to the project folder

from prepare import check, clean, prepare  # noqa: E402
from ranking.load import load_config, prepare_series, season_cfg  # noqa: E402
from ranking.placements import secured_placements  # noqa: E402
from ranking.update import movement_table, on_schedule, previous_snapshot, previous_state, run_update  # noqa: E402
from ranking.validate import build_model  # noqa: E402

RAW = Path("data/raw/games_2026.csv")
REFERENCE = Path("tests/fixtures/primary_games_2026_with_worlds.csv")   # the hand-cleaned file
MAJOR = Path("data/games_2026_major.csv")                     # earlier cleaned file, same games
DATE = "2026-10-26"


def read_back(games, tmp_path):
    """Write and re-read, so values are compared exactly as they appear in a CSV."""
    path = tmp_path / "clean.csv"
    games.to_csv(path, index=False, encoding="utf-8")
    return pd.read_csv(path, encoding="utf-8")


def same_rows(a, b):
    a = a.sort_values("GameId").reset_index(drop=True)
    b = b.sort_values("GameId").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_like=True)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def cleaned(cfg):
    games, fallback = clean(pd.read_csv(RAW, encoding="utf-8"), cfg)
    return games, fallback


@pytest.fixture(scope="module")
def update_model(cfg, cleaned, tmp_path_factory):
    games, _ = cleaned
    path = tmp_path_factory.mktemp("update") / "games_clean.csv"
    games[games["DateTime UTC"] < DATE].to_csv(path, index=False, encoding="utf-8")
    run_cfg = copy.deepcopy(cfg)
    season_cfg(run_cfg)["games_csv"] = str(path)
    series, home = prepare_series(run_cfg)
    return series, build_model(series, home, run_cfg)


@pytest.mark.skipif(not REFERENCE.exists(), reason=f"{REFERENCE} is missing")
def test_prepare_reproduces_hand_cleaned_file(cleaned, tmp_path):
    games, _ = cleaned
    reference = pd.read_csv(REFERENCE, encoding="utf-8")
    assert len(reference) == 2595
    same_rows(read_back(games, tmp_path), reference)


def test_prepare_matches_earlier_cleaned_file(cleaned, tmp_path):
    # The earlier cleaned file set IsQualifier = 0 for the Regional Finals and the Worlds Play-In;
    # prepare.py keeps Leaguepedia's value. Every other column must match exactly.
    games, _ = cleaned
    mine = read_back(games, tmp_path).drop(columns="IsQualifier")
    same_rows(mine, pd.read_csv(MAJOR, encoding="utf-8").drop(columns="IsQualifier"))
    assert len(mine) == 2595


def test_regional_finals_are_lpl_playoffs_not_worlds(cleaned):
    games, _ = cleaned
    rf = games[games["Tournament"] == "LPL 2026 Regional Finals"]
    assert len(rf) == 12
    assert (rf["League"] == "Tencent LoL Pro League").all()
    assert (rf["Region"] == "China").all()
    assert (rf["TournamentLevel"] == "Regional").all()
    assert (rf["IsPlayoffs"] == 1).all()


def test_fallback_stages_are_reported(cleaned):
    _, fallback = cleaned
    assert ("Worlds 2026 Main Event", "Round 1") in fallback
    assert all(t in ("LCP 2026 Split 3", "Worlds 2026 Main Event") for t, _ in fallback)


def test_prepare_stops_on_duplicate_game_or_bad_winner(cfg, cleaned):
    games, _ = cleaned
    with pytest.raises(SystemExit, match="duplicate GameIds"):
        check(pd.concat([games, games.head(1)]), cfg)
    bad = games.copy()
    bad.loc[bad.index[0], "WinTeam"] = "Nobody"
    with pytest.raises(SystemExit, match="winner is not one of the two teams"):
        check(bad, cfg)


def test_prepare_warns_on_missing_rows_and_new_tournaments(cfg):
    _, report = prepare(RAW, cfg, before=DATE, previous=pd.read_csv("data/games_2026_pre-worlds.csv"))
    assert "Dplus Kia vs Gen.G (LCK/2026 Season/Rounds 3-4_Week 10_8)" in report
    assert "New tournaments since the previous update: Worlds 2026 Main Event, Worlds 2026 Play-In" in report


def test_worlds_in_progress_gives_secured_shares(update_model):
    _, model = update_model
    worlds = model.placements[model.placements["title_event"] == "Worlds 2026"]
    assert len(worlds) and (worlds["status"] == "in progress").all()
    assert (worlds["share"] == 0.1).all()          # Play-In and Swiss only so far: participants


def test_secured_share_follows_furthest_stage(cfg, update_model):
    series, _ = update_model
    event = series[series["title_event"] == "Worlds 2026"].copy()
    first = event.index[-1]
    event.loc[first, ["stage", "is_playoffs"]] = ["Quarterfinals", 1]
    table = secured_placements(event, cfg["in_progress_placements"]["default"]).set_index("team")
    for team in (event.loc[first, "winner"], event.loc[first, "loser"]):
        assert table.loc[team, "share"] == 0.2
    others = set(table.index) - {event.loc[first, "winner"], event.loc[first, "loser"]}
    assert (table.loc[sorted(others), "share"] == 0.1).all()


def test_unfinished_domestic_event_gives_no_points(cfg):
    series, home = prepare_series(cfg)
    before_final = series[series["start"] < "2026-09-13"]   # LPL Grand Finals final is on 13 Sep
    placements = build_model(before_final, home, cfg).placements
    assert "LPL Split 3" not in set(placements["title_event"])
    assert "LCK Season" not in set(placements["title_event"])
    assert "LEC Summer" not in set(placements["title_event"])   # final on 20 Sep
    assert "LCP Split 3" in set(placements["title_event"])      # final on 30 Aug: finished


def test_points_change_splits_into_its_parts(cfg, update_model):
    series, model = update_model
    prev, _, cutoff = previous_state(Path(cfg["data"]["output_dir"]) / cfg["update"]["baseline_snapshot"])
    new_ids = set(series.loc[series["start"] >= pd.Timestamp(cutoff), "series_id"])
    moves = movement_table(model, prev, new_ids)
    start_change = model.standings.set_index("team")["start_rating"] - prev["start_rating"]
    # start = seed + offset; the baseline's offsets are stored rounded to 0.1
    assert ((moves["seed_change"] + moves["league_offset_change"] - start_change).abs() < 0.06).all()
    parts = moves[["since_last_update", "league_offset_change", "seed_change", "earlier_games_change"]].sum(axis=1)
    assert ((parts - moves["points_change"]).abs() < 1e-9).all()
    assert set(series.loc[series["series_id"].isin(new_ids), "event"]) == {"worlds"}


def test_schedule_is_anchor_plus_multiples_of_14_days(cfg):
    for date in ("2026-10-26", "2026-11-09", "2026-12-07"):
        assert on_schedule(cfg, date)
    for date in ("2026-11-02", "2026-10-27"):
        assert not on_schedule(cfg, date)


def test_off_schedule_date_is_refused_without_force(cfg):
    with pytest.raises(SystemExit, match="not an update date"):
        run_update(cfg, "2026-11-02")


def test_previous_snapshot_is_last_biweekly_or_baseline(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["data"]["output_dir"] = str(tmp_path)
    assert previous_snapshot(c, "2026-10-26")[0] == cfg["update"]["baseline_snapshot"]
    for d in ("2026-10-26", "2026-11-09", ".tmp-2026-11-23"):
        (tmp_path / cfg["update"]["dir"] / d).mkdir(parents=True)
    name, folder, games = previous_snapshot(c, "2026-11-23")
    assert name == f"{cfg['update']['dir']}/2026-11-09"
    assert games == folder / "games_clean.csv"


def test_biweekly_snapshot_is_frozen(cfg):
    if not (Path(cfg["data"]["output_dir"]) / cfg["update"]["dir"] / DATE).exists():
        pytest.skip("first biweekly snapshot not built yet")
    with pytest.raises(SystemExit, match="frozen"):
        run_update(cfg, DATE)

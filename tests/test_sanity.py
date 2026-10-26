"""Sanity checks on the data handling and the model's outputs. Run: python -m pytest tests"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)   # config paths are relative to the project folder

from ranking.factors import FACTOR_NAMES  # noqa: E402
from ranking.load import collapse_to_series, load_config, load_games, prepare_series, season_cfg, tag_events  # noqa: E402
from ranking.report import regional_tables  # noqa: E402
from ranking.snapshot import prepare_snapshot_dir  # noqa: E402
from ranking.validate import build_model  # noqa: E402


@pytest.fixture(scope="module")
def built():
    cfg = load_config()
    series, home = prepare_series(cfg)
    return cfg, series, home, build_model(series, home, cfg)


def test_one_series_per_match_id(built):
    cfg, series, _, _ = built
    games = load_games(season_cfg(cfg)["games_csv"])
    assert len(series) == games["MatchId"].nunique()


def test_known_missing_game_series_uses_final_score(built):
    _, series, _, _ = built
    row = series[series["series_id"] == "LCK/2026 Season/Rounds 3-4_Week 10_8"].iloc[0]
    assert (row["winner_score"], row["loser_score"], row["n_games"]) == (2, 0, 1)


def test_team_count(built):
    _, series, home, _ = built
    assert len(home) == 60
    assert set(series["winner"]) | set(series["loser"]) == set(home)


def test_no_international_only_team(built):
    _, series, home, _ = built
    domestic = series[~series["is_intl"]]
    assert set(home) == set(domestic["winner"]) | set(domestic["loser"])


def test_no_lpl_vs_lpl_series_before_worlds_is_classed_as_worlds(built):
    # Leaguepedia files the LPL Regional Finals under "World Championship". LPL teams can meet each
    # other at Worlds itself, so the check covers everything before the Worlds cutoff.
    cfg, series, _, _ = built
    cutoff = pd.Timestamp(season_cfg(cfg)["final_test"]["holdout"]["cutoff"])
    lpl_vs_lpl = series[(series["winner_league"] == "LPL") & (series["loser_league"] == "LPL")
                        & (series["start"] < cutoff)]
    assert not (lpl_vs_lpl["event"] == "worlds").any()


def test_mislabelled_qualifier_is_caught(built):
    # If a one-league qualifier keeps Leaguepedia's "World Championship" label, loading must fail.
    cfg, _, home, _ = built
    series = collapse_to_series(load_games(season_cfg(cfg)["games_csv"]))
    fake = series["tournament"] == "LPL 2026 Grand Finals"
    series.loc[fake, ["level", "league_name"]] = ["International", "World Championship"]
    with pytest.raises(AssertionError, match="same-league"):
        tag_events(series, cfg, home)


def test_untitled_tournaments_give_no_placements(built):
    cfg, series, _, model = built
    untitled = season_cfg(cfg)["untitled_tournaments"]
    assert series.loc[series["tournament"].isin(untitled), "title_event"].isna().all()
    lpl_final = [te for te in season_cfg(cfg)["title_events"] if te["name"] == "LPL Split 3"][0]
    assert not set(untitled) & set(lpl_final["tournaments"])
    assert not model.placements["title_event"].isin(untitled).any()


def test_lpl_season_champion_is_al(built):
    _, _, _, model = built
    p = model.placements
    champion = p[(p["title_event"] == "LPL Split 3") & (p["placement"] == 1)]["team"].tolist()
    assert champion == ["Anyone's Legend"]


def test_tier_order_holds(built):
    _, _, _, model = built
    tiers = sorted(set(model.league_table["tier"]))
    by_tier = model.league_table.groupby("tier")["fitted_offset"]
    for upper, lower in zip(tiers, tiers[1:]):
        assert by_tier.min()[upper] >= by_tier.max()[lower] - 1e-6


def test_regional_tables_rebuild_global(built):
    _, _, _, model = built
    rebuilt = pd.concat(regional_tables(model.standings).values())
    rebuilt = rebuilt.sort_values("global_rank").drop(columns="regional_rank").rename(columns={"global_rank": "rank"})
    pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True), model.standings.reset_index(drop=True))


def test_factors_between_0_and_1(built):
    _, _, _, model = built
    values = model.factors[FACTOR_NAMES]
    assert ((values >= 0) & (values <= 1)).all().all()


def test_snapshot_is_frozen(tmp_path):
    cfg = {"data": {"output_dir": str(tmp_path)}}
    prepare_snapshot_dir(cfg, "snap")
    (tmp_path / "snap" / "marker.txt").write_text("keep")
    with pytest.raises(SystemExit, match="frozen"):
        prepare_snapshot_dir(cfg, "snap")
    assert (tmp_path / "snap" / "marker.txt").exists()
    prepare_snapshot_dir(cfg, "snap", force=True)
    assert not (tmp_path / "snap" / "marker.txt").exists()

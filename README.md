# LoL global power ranking

A power ranking of every team in the six major League of Legends leagues (LCK, LPL, LEC, LCS, LCP, CBLOL), updated every two weeks. It is adapted from Valve's Counter-Strike Regional Standings and works in three steps:

1. **League strength:** one offset per league, fitted from international results only.
2. **Résumé:** four factors (placements, quality wins, the network of teams beaten, international wins) set each team's starting rating, plus its league's offset.
3. **Elo replay:** every series of the season is replayed in date order, with recent games counting more.

## Folder layout

```
config.yaml          every setting
main.py              the command you run
prepare.py           raw Leaguepedia export -> cleaned games file
fetch_leaguepedia.py downloads the raw export into data/raw/
scrape_golgg.py      backup scraper for stats Leaguepedia lacks (not used by the ranking)
ranking/             the model: load, league_strength, placements, factors, elo, report, validate,
                     snapshot, update
tests/               tests; tests/fixtures/ holds files only the tests use
data/raw/            raw Leaguepedia exports
data/                cleaned games files and hand fixes to placements
output/              frozen snapshots: 2026_pre_worlds/ and biweekly/<date>/
```

## Running a biweekly update

Updates fall on 26 Oct 2026 and every 14 days after it (9 Nov, 23 Nov, ...).

1. Install once: `pip install -r requirements.txt`
2. Download the export: `python fetch_leaguepedia.py 2026`. This writes `data/raw/games_2026.csv`. Any Leaguepedia export with the same columns also works if you drop it into `data/raw/`.
3. Run: `python main.py --date 2026-11-09`

The run uses the newest file in `data/raw/` and every series that started before the date. It writes `output/biweekly/<date>/`:
- **changes.md:** who moved and why
- **standings and regional tables:** with movement columns
- **per-team detail pages**
- **prepare_report.md:** data warnings to read

A snapshot is frozen, so running the same date again is refused unless you add `--force`. Dates off the schedule also need `--force`.

Other commands:
- `python main.py --snapshot NAME`: a full-season snapshot with every validation experiment.
- `python main.py --final-test`: the Worlds test.
- `python -m pytest tests`: the tests.

## Where the settings live in config.yaml

| What | Key |
|---|---|
| Team renames (the name map) | `team_aliases` |
| Which raw rows are kept, playoff rules | `prepare` |
| Leagues, tiers, multipliers, prior offsets | `leagues`, `tiers` |
| International event weights and placement points | `international_events` |
| Domestic weights and placement points | `domestic`, `placement_shares` |
| Events in progress (when finished, secured shares) | `final_stage`, `in_progress_placements` |
| Recency (age weight) | `age_weight` |
| League strength fit | `league_strength` |
| Résumé factors and seed range | `factors`, `seed_range` |
| Elo replay | `elo` |
| Update schedule and baseline | `update` |
| Season files, title events, relabelled tournaments, holdouts, Worlds test | `seasons: 2026:` |
| Validation experiments | `validation` |

Data warnings to watch in `prepare_report.md`:
- new teams or tournaments
- stage names that hit the playoff fallback
- series with missing game rows

A new team name usually needs a line in `team_aliases`. A new stage name may need a rule in `prepare.playoff_rules`.

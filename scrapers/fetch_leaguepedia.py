"""
Pull every pro League of Legends game result from Leaguepedia (lol.fandom.com).
Leaguepedia has an official query API (called "Cargo"), so this is not scraping:
you ask for tables and get clean rows back.

Install:  pip install -r requirements.txt
Run:      python fetch_leaguepedia.py 2026
Output:   data/raw/games_<year>.csv  (one row per game, the raw export prepare.py reads)
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from mwclient.errors import APIError, MaximumRetriesExceeded
from requests.exceptions import ConnectionError as RequestsConnectionError
from mwrogue.auth_credentials import AuthCredentials
from mwrogue.esports_client import EsportsClient

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging in with a bot password gets much higher rate limits than anonymous access.
# Credentials live in ~/.config/mwcleric/wiki_account_lol.json (or the env vars
# WIKI_USERNAME_LOL / WIKI_PASSWORD_LOL); without them we fall back to anonymous.
CRED_FILE = Path(AuthCredentials.config_path) / "wiki_account_lol.json"
if CRED_FILE.exists() or os.getenv("WIKI_PASSWORD_LOL"):
    site = EsportsClient("lol", credentials=AuthCredentials(user_file="lol"))
    print(f"Logged in as {site.credentials.username}")
else:
    site = EsportsClient("lol")
    print("No credentials found, using anonymous access (lower rate limits)")


def query_with_retry(max_tries: int = 8, **kwargs):
    # Leaguepedia rate-limits and sometimes drops connections; back off and retry instead of crashing.
    wait = 30
    for attempt in range(1, max_tries + 1):
        try:
            return site.cargo_client.query(**kwargs)
        except (APIError, MaximumRetriesExceeded, RequestsConnectionError) as e:
            if isinstance(e, APIError) and e.code != "ratelimited":
                raise
            if attempt == max_tries:
                raise
            reason = "rate-limited" if isinstance(e, APIError) else "connection dropped"
            print(f"  {reason}, waiting {wait}s (attempt {attempt}/{max_tries})")
            time.sleep(wait)
            wait = min(wait * 2, 300)


def fetch_year(year: int) -> pd.DataFrame:
    # Each page is saved to a .partial file so a crash resumes instead of starting over.
    partial = OUT_DIR / f"games_{year}.partial.csv"
    if partial.exists():
        rows = pd.read_csv(partial, dtype=str, keep_default_na=False).to_dict("records")
        print(f"  {year}: resuming from {len(rows)} saved games")
    else:
        rows = []
    offset = len(rows)
    page_size = 500  # Leaguepedia's max per request

    while True:
        result = query_with_retry(
            tables="ScoreboardGames=SG, Tournaments=T",
            join_on="SG.OverviewPage=T.OverviewPage",
            fields=(
                "SG.GameId, SG.MatchId, SG.DateTime_UTC, SG.Tournament, "
                "SG.OverviewPage, SG.Team1, SG.Team2, SG.WinTeam, SG.LossTeam, "
                "SG.Team1Score, SG.Team2Score, SG.Gamelength_Number, SG.Patch, "
                "T.Region, T.League, T.TournamentLevel, T.IsQualifier, "
                "T.IsPlayoffs, T.IsOfficial, T.Prizepool, T.Country"
            ),
            where=f"SG.DateTime_UTC >= '{year}-01-01' AND SG.DateTime_UTC < '{year + 1}-01-01'",
            order_by="SG.DateTime_UTC, SG.GameId",  # GameId tiebreak keeps paging stable
            limit=page_size,
            offset=offset,
        )
        if not result:
            break
        rows.extend(result)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(f"  {year}: {len(rows)} games so far")
        offset += len(result)
        if len(result) < page_size:
            break
        time.sleep(5)  # be polite; Leaguepedia rate-limits fast clients

    df = pd.DataFrame(rows)
    partial.unlink(missing_ok=True)
    return df


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or [2026]
    for y in years:
        path = OUT_DIR / f"games_{y}.csv"
        if path.exists() and y != max(years):
            print(f"Skipping {y}, already have {path}")
            continue
        df = fetch_year(y)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} games -> {path}")

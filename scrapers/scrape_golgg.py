"""
Backup option: scrape gol.gg directly.
Use this only for numbers Leaguepedia doesn't have (e.g. gold diff at 15, dragons per game).

gol.gg page layout (check these in your browser first, the site changes sometimes):
  Tournament list : loaded by JavaScript from a POST to /tournament/ajax.trlist.php
  Match list      : https://gol.gg/tournament/tournament-matchlist/<Tournament Name>/
  One game        : https://gol.gg/game/stats/<game_id>/page-game/

How to confirm: open the page, press F12 -> Network tab -> reload, and look for
the request that returns the tournament list. Copy its URL and form data here.

Install:  pip install -r requirements.txt
Run:      python scrape_golgg.py "LCK 2025 Split 2"
"""

import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://gol.gg"
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project; contact: you@example.com)"}
CACHE = Path("cache_golgg")
CACHE.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update(HEADERS)


def get_html(url: str) -> str:
    """Download a page once and keep it on disk, so re-runs don't hit the site again."""
    name = re.sub(r"[^A-Za-z0-9]+", "_", url)[-150:] + ".html"
    path = CACHE / name
    if path.exists():
        return path.read_text(encoding="utf-8")
    r = session.get(url, timeout=30)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    time.sleep(2)  # wait between requests; don't hammer a small fan-run site
    return r.text


def list_tournaments(season: str = "S15") -> pd.DataFrame:
    """Tournament names for one season. The form fields may differ: check the Network tab."""
    r = session.post(f"{BASE}/tournament/ajax.trlist.php", data={"season": season}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())  # usually a list of dicts with a 'trname' key


def match_list(tournament_name: str) -> pd.DataFrame:
    url = f"{BASE}/tournament/tournament-matchlist/{quote(tournament_name)}/"
    soup = BeautifulSoup(get_html(url), "html.parser")

    rows = []
    for tr in soup.select("table tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        link = cells[0].find("a")
        game_id = None
        if link and "/game/stats/" in link.get("href", ""):
            game_id = link["href"].split("/game/stats/")[1].split("/")[0]
        rows.append(
            {
                "tournament": tournament_name,
                "match": cells[0].get_text(strip=True),
                "team1": cells[1].get_text(strip=True),
                "score": cells[2].get_text(strip=True),   # e.g. "2 - 1"
                "team2": cells[3].get_text(strip=True),
                "date": cells[-1].get_text(strip=True),
                "game_id": game_id,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "LCK 2025 Split 2"
    df = match_list(name)
    out = f"golgg_{re.sub(r'[^A-Za-z0-9]+', '_', name)}.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} matches -> {out}")
    print(df.head())

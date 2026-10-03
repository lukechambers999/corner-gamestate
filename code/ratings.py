"""
Venue-specific team attack/defence corner ratings.

1. League baseline: a lagged rolling average of home-side and away-side corner
   values across the whole league, sized to cover the same period as one team's
   N-game window (N home games <-> N * teams_in_league league matches).
2. Each match's corner value is divided by its league baseline, so every
   observation is a ratio centred on 1.00.
3. A team's rating is the lagged rolling mean of those ratios over its last N
   home (or away) games. Because each observation is already a ratio, the
   window can span a promotion or relegation within the same country.
4. A window never crosses a gap of more than MAX_GAP_DAYS (e.g. a spell in an
   untracked division), or the closed-doors COVID period ('era').

Metric columns: raw counts, nr (repeaters removed), adj (adjusted values) and
adj_nr (adjusted values, repeaters removed).
"""

import math

import numpy as np
import pandas as pd

MAX_GAP_DAYS = 300


def add_league_baselines(df: pd.DataFrame, home_col: str, away_col: str, window_size: int) -> pd.DataFrame:
    """Lagged league-wide averages of home and away corner values (current match excluded)."""
    df = df.sort_values(["League", "era", "Date", "Matchid"]).copy()
    df["league_window"] = (window_size * df["teams_in_league_season"]).round().clip(lower=1).astype(int)
    df["league_home_baseline"] = np.nan
    df["league_away_baseline"] = np.nan

    for _, grp in df.groupby(["League", "era"], sort=False):
        for W in grp["league_window"].unique():
            idx = grp.index[(grp["league_window"] == W).values]
            df.loc[idx, "league_home_baseline"] = grp[home_col].rolling(W, min_periods=W).mean().shift(1)[idx]
            df.loc[idx, "league_away_baseline"] = grp[away_col].rolling(W, min_periods=W).mean().shift(1)[idx]
    return df


def _rolling_shifted(s: pd.Series, window_size: int) -> pd.Series:
    """Mean of the previous `window_size` games; needs at least half the window to be valid."""
    return s.rolling(window_size, min_periods=math.ceil(window_size / 2)).mean().shift(1)


def _assign_stints(df: pd.DataFrame, keys: list) -> pd.DataFrame:
    """New stint whenever a team goes more than MAX_GAP_DAYS between games at this venue."""
    gap = df.groupby(keys)["Date"].diff().dt.days
    df["stint"] = (gap.isna() | (gap > MAX_GAP_DAYS)).groupby([df[k] for k in keys]).cumsum()
    return df


def add_team_ratings(df: pd.DataFrame, home_col: str, away_col: str, window_size: int) -> pd.DataFrame:
    """Adds home/away attack and defence ratings for each match, using only earlier matches."""
    df = df.copy()
    df["ratio_home"] = df[home_col] / df["league_home_baseline"]
    df["ratio_away"] = df[away_col] / df["league_away_baseline"]
    keys = ["Team", "Country", "era"]
    cols = ["Matchid", "Date", "Country", "era", "ratio_home", "ratio_away"]

    home = df[cols + ["Home"]].rename(columns={"Home": "Team"}).sort_values(keys + ["Date"])
    home = _assign_stints(home, keys)
    g = home.groupby(keys + ["stint"])
    home["home_corner_attack_rating"] = g["ratio_home"].transform(lambda s: _rolling_shifted(s, window_size))
    home["home_corner_defence_rating"] = g["ratio_away"].transform(lambda s: _rolling_shifted(s, window_size))

    away = df[cols + ["Away"]].rename(columns={"Away": "Team"}).sort_values(keys + ["Date"])
    away = _assign_stints(away, keys)
    g = away.groupby(keys + ["stint"])
    away["away_corner_attack_rating"] = g["ratio_away"].transform(lambda s: _rolling_shifted(s, window_size))
    away["away_corner_defence_rating"] = g["ratio_home"].transform(lambda s: _rolling_shifted(s, window_size))

    df = df.merge(home[["Matchid", "home_corner_attack_rating", "home_corner_defence_rating"]], on="Matchid")
    return df.merge(away[["Matchid", "away_corner_attack_rating", "away_corner_defence_rating"]], on="Matchid")

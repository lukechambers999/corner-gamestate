"""
Game-state x supremacy corner rates, and adjusted corner values.

1. Bucket team-matches by pre-match supremacy (30 quantiles), separately for
   home and away sides.
2. In each bucket and game state, corner rate = corners / minutes.
3. Fit rate ~ supremacy (OLS) per (home/away, game state): six lines.
4. Value each non-repeater corner as
       avg_rate(home/away) / expected_rate(home/away, state, supremacy)
   so a corner won in a context where corners are usually rare is worth > 1,
   and one won where corners are usually plentiful is worth < 1.
   Repeaters are worth 0.
"""

import numpy as np
import pandas as pd

STATES = {  # team-perspective state -> (minutes column, corners column)
    "Drawing": ("teamdrawingmins", "team_drawing_corners"),
    "Winning": ("teamleadingmins", "team_leading_corners"),
    "Losing": ("teamlosingmins", "team_losing_corners"),
}
N_BUCKETS = 30


def supremacy_buckets(team_matches: pd.DataFrame) -> pd.DataFrame:
    """Corners-per-minute by supremacy bucket and game state, for one side (home or away)."""
    tm = team_matches.copy()
    tm["sup_bucket"] = pd.qcut(tm["teamsup"], N_BUCKETS, labels=False, duplicates="drop")
    frames = []
    for state, (mins_col, corners_col) in STATES.items():
        g = tm.groupby("sup_bucket").agg(
            mins=(mins_col, "sum"), corners=(corners_col, "sum"), sup_mean=("teamsup", "mean")
        ).reset_index()
        g["cpm"] = g["corners"] / g["mins"]
        g["game_state"] = state
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def fit_lines(buckets: pd.DataFrame) -> dict:
    """{state: (slope, intercept)} from an OLS fit of cpm on mean supremacy."""
    lines = {}
    for state, g in buckets.groupby("game_state"):
        slope, intercept = np.polyfit(g["sup_mean"], g["cpm"], 1)
        lines[state] = (slope, intercept)
    return lines


def team_state(row) -> str:
    """Game state from the corner-taking team's perspective."""
    if row["gamestate"] == 0:
        return "Drawing"
    home_leading = row["gamestate"] == 1
    is_home = row["Team"] == row["Home"]
    return "Winning" if home_leading == is_home else "Losing"


def adjusted_values(events: pd.DataFrame, home_buckets, away_buckets) -> pd.DataFrame:
    """Adds expected_rate, avg_rate and adj_corner_value to each corner event."""
    lines = {"home": fit_lines(home_buckets), "away": fit_lines(away_buckets)}
    avg = {
        "home": home_buckets["corners"].sum() / home_buckets["mins"].sum(),
        "away": away_buckets["corners"].sum() / away_buckets["mins"].sum(),
    }

    ev = events[events["Corner"] == 1].copy()
    side = np.where(ev["Team"] == ev["Home"], "home", "away")
    state = ev.apply(team_state, axis=1)
    slope = [lines[s][st][0] for s, st in zip(side, state)]
    intercept = [lines[s][st][1] for s, st in zip(side, state)]

    ev["side"] = side
    ev["team_state"] = state
    ev["expected_rate"] = np.array(intercept) + ev["teamsup"].to_numpy() * np.array(slope)
    ev["avg_rate"] = [avg[s] for s in side]
    ev["adj_corner_value"] = np.where(ev["Repeater"] == 1, 0.0, ev["avg_rate"] / ev["expected_rate"])
    return ev

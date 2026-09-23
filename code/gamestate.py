"""
Game-state processing for match event data.

Input: one row per match event (goals, corners, cards) scraped from
totalcorner.com, with columns Home, Away, Team, Minute, Event, matchid,
teammatchid. Output adds, for every row:

  - gamestate            0 = level, 1 = home leading, 2 = away leading
  - home/away/drawing minutes spent in each state over the whole match
  - team_{leading,losing,drawing}_corners for the row's team

Matches whose minutes or corners don't reconcile are dropped.
"""

import numpy as np
import pandas as pd

MIN_MATCH_MINUTES = 97  # assumed match length incl. stoppage when no later event exists


def add_gamestate(df: pd.DataFrame) -> pd.DataFrame:
    """Running score and game state at every event, plus where the state changed."""
    df = df.copy()
    is_goal = df["Event"] == "Goal"
    df["home_goals"] = (is_goal & (df["Team"] == df["Home"])).groupby(df["matchid"]).cumsum()
    df["away_goals"] = (is_goal & (df["Team"] == df["Away"])).groupby(df["matchid"]).cumsum()

    df["gamestate"] = np.select(
        [df["home_goals"] > df["away_goals"], df["home_goals"] < df["away_goals"]],
        [1, 2],
        default=0,
    )
    # Signed change in state code at this event (0 = no change)
    df["gamestate_change"] = df.groupby("matchid")["gamestate"].diff().fillna(0)
    return df


def add_gamestate_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Minutes spent home-leading / away-leading / level in each match."""
    df = df.copy()
    first_goal = (df.groupby(["matchid", "Event"]).cumcount() == 0) & (df["Event"] == "Goal")

    # Opening level period ends at the first goal
    df["drawingmins"] = np.where(first_goal, df["Minute"], 0)

    # Minute of each state change (first goal handled via drawingmins)
    df["gs_change_min"] = np.where(
        df["gamestate_change"].isin([1, 2, -1, -2]), df["Minute"], df["drawingmins"]
    )

    last_event = df.groupby("matchid")["Minute"].transform("max")
    df["matchmins"] = np.maximum(last_event, MIN_MATCH_MINUTES)

    # Final state lasts from the last change to full time
    df["final_gs_mins"] = df["matchmins"] - df.groupby("matchid")["gs_change_min"].transform("max")
    last_row = df.groupby("matchid").cumcount(ascending=False) == 0
    df["homeleadmins"] = np.where((df["gamestate"] == 1) & last_row, df["final_gs_mins"], 0)
    df["awayleadmins"] = np.where((df["gamestate"] == 2) & last_row, df["final_gs_mins"], 0)
    df["drawingmins"] = np.where((df["gamestate"] == 0) & last_row, df["final_gs_mins"], df["drawingmins"])

    # Intermediate states: gap between successive change minutes, credited to the
    # state that just ended (a change of -1 means a home lead just ended, etc.)
    df["gs_mins"] = df.groupby("matchid")["gs_change_min"].transform(
        lambda x: x.where(x != 0).ffill().fillna(0).diff().fillna(0)
    )
    df["gs_mins"] = np.where(df["gs_mins"] < 0, 1, df["gs_mins"])  # two changes either side of HT
    df["homeleadmins"] = np.where(df["gamestate_change"] == -1, df["gs_mins"], df["homeleadmins"])
    df["awayleadmins"] = np.where(df["gamestate_change"] == -2, df["gs_mins"], df["awayleadmins"])
    df["drawingmins"] = np.where(df["gamestate_change"].isin([1, 2]), df["gs_mins"], df["drawingmins"])

    for col in ["homeleadmins", "awayleadmins", "drawingmins"]:
        df[col] = df.groupby("matchid")[col].transform("sum")

    # Same minutes from the row's team perspective
    is_home = df["Team"] == df["Home"]
    df["teamleadingmins"] = np.where(is_home, df["homeleadmins"], df["awayleadmins"])
    df["teamlosingmins"] = np.where(is_home, df["awayleadmins"], df["homeleadmins"])
    df["teamdrawingmins"] = df["drawingmins"]

    # Drop matches whose state minutes don't add up to the match length
    reconciles = df["matchmins"] == df["homeleadmins"] + df["awayleadmins"] + df["drawingmins"]
    return df[reconciles].copy()


def add_gamestate_corners(df: pd.DataFrame) -> pd.DataFrame:
    """Corners won by each side in each state, per match and per team."""
    df = df.copy()
    corner = df["Event"] == "Corner"
    home, away = df["Team"] == df["Home"], df["Team"] == df["Away"]
    gs = df["gamestate"]

    flags = {
        "home_leading_corners": corner & home & (gs == 1),
        "home_losing_corners": corner & home & (gs == 2),
        "home_drawing_corners": corner & home & (gs == 0),
        "away_leading_corners": corner & away & (gs == 2),
        "away_losing_corners": corner & away & (gs == 1),
        "away_drawing_corners": corner & away & (gs == 0),
    }
    for col, flag in flags.items():
        df[col] = flag.astype(int).groupby(df["matchid"]).transform("sum")

    for state in ["leading", "losing", "drawing"]:
        df[f"team_{state}_corners"] = np.where(
            home, df[f"home_{state}_corners"], df[f"away_{state}_corners"]
        )

    # Drop matches whose per-state corners don't sum to the scraped total
    per_state_total = df[list(flags)].sum(axis=1)
    return df[per_state_total == df["total_corners"]].copy()


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: state -> minutes -> corners."""
    return add_gamestate_corners(add_gamestate_minutes(add_gamestate(df)))

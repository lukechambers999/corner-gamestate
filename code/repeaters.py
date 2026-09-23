"""
Repeater detection.

A repeater is a corner that immediately follows another corner by the same
team, in the same half, within one minute. Repeaters aren't independent of the
corner before them, so they're excluded from corner ratings.
"""

import numpy as np
import pandas as pd


def flag_repeaters(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Corner (0/1) and Repeater (0/1). Expects rows in match-event order."""
    df = df.copy()
    df["Corner"] = (df["Event"] == "Corner").astype(int)

    prev = df.shift(1)
    minutes_since_prev = (df["Minute"] - df.groupby("matchid")["Minute"].shift(1)).fillna(df["Minute"])

    df["Repeater"] = np.where(
        (df["Corner"] == 1)
        & (prev["Corner"] == 1)
        & (df["half"] == prev["half"])
        & (df["teammatchid"] == prev["teammatchid"])
        & (minutes_since_prev < 2),
        1,
        0,
    )
    return df


def repeater_rates(df: pd.DataFrame, by) -> pd.DataFrame:
    """Share of corners that are repeaters, grouped by `by`."""
    out = df.groupby(by).agg(
        matches=("teammatchid", "nunique"),
        corners=("Corner", "sum"),
        repeaters=("Repeater", "sum"),
    ).reset_index()
    out["repeater_rate"] = out["repeaters"] / out["corners"]
    return out

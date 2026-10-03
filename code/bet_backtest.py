"""
Handicap betting backtest against Pinnacle.

For each priced match, compare the model's predicted corner supremacy with
Pinnacle's implied corner supremacy. If they differ by EDGE_THRESHOLD or more,
place a 1-unit Asian handicap bet on the side the model prefers, at Pinnacle's
line and price, and settle it on the actual raw corner margin.
"""

import pandas as pd

EDGE_THRESHOLD = 0.5


def settle_asian_single(margin: float, line: float, odds: float) -> float:
    """1-unit bet on a whole or half line. Whole lines can push."""
    if line == int(line) and margin == line:
        return 0.0
    return odds - 1.0 if margin > line else -1.0


def settle_asian(margin: float, line: float, odds: float) -> float:
    """Quarter lines (.25/.75) split into two half-stakes on the neighbouring lines."""
    if line % 0.5 == 0.25:
        return 0.5 * settle_asian_single(margin, line - 0.25, odds) + 0.5 * settle_asian_single(margin, line + 0.25, odds)
    return settle_asian_single(margin, line, odds)


def simulate_bets(df: pd.DataFrame, threshold: float = EDGE_THRESHOLD) -> pd.DataFrame:
    """
    df: one row per match with predicted_supremacy, Pinnacle's corner_supremacy,
    Corner Handicap (away-side line), home/away handicap odds and raw corners.
    Returns one row per bet with its profit/loss in units.
    """
    margin = df["home_corners_raw"] - df["away_corners_raw"]
    edge = df["predicted_supremacy"] - df["corner_supremacy"]
    line = df["Corner Handicap"]

    home = edge >= threshold
    away = edge <= -threshold
    pnl_home = [settle_asian(m, -l, o) for m, l, o in
                zip(margin[home], line[home], df.loc[home, "Home Handicap Corners Odds"])]
    pnl_away = [settle_asian(-m, l, o) for m, l, o in
                zip(margin[away], line[away], df.loc[away, "Away Handicap Corners Odds"])]

    return pd.concat([
        df.loc[home, ["Matchid", "League"]].assign(side="home", pnl=pnl_home),
        df.loc[away, ["Matchid", "League"]].assign(side="away", pnl=pnl_away),
    ], ignore_index=True)


def summarise(bets: pd.DataFrame) -> pd.Series:
    return pd.Series({"n_bets": len(bets), "total_pnl": bets["pnl"].sum(), "roi": bets["pnl"].mean()})

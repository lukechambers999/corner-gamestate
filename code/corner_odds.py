"""
Pinnacle corner odds -> fair expected corners and corner supremacy.

Pinnacle prices two corner markets for each match: Asian corner totals
(e.g. 10.5 at 1.970 / 1.862) and a corner handicap (e.g. home -0.5 at
1.925 / 1.877). The same method as odds_to_goals.py is applied to corners:

  - expected total corners  (Poisson mean that prices the totals line fairly)
  - corner supremacy        (expected home-minus-away corners that prices the
                             handicap fairly under a Skellam model)

These fair values are the market benchmark the model is tested against.
Adapted from pin_scraper/pinn_odds_proc_web.py.
"""

from scipy.optimize import brentq

from odds_to_goals import asian_home_probability, asian_over_probability, remove_margin


def fit_corner_total(line, over_odds, under_odds):
    """Expected total corners implied by the Asian corners line."""
    p_over, _ = remove_margin(over_odds, under_odds)
    return brentq(lambda lam: asian_over_probability(lam, line) - p_over, 1.0, 30.0)


def fit_corner_supremacy(line, home_odds, away_odds, corner_total):
    """Expected home-minus-away corners implied by the corner handicap."""
    p_home, _ = remove_margin(home_odds, away_odds)
    base = corner_total / 2
    return brentq(lambda d: asian_home_probability(d, line, base) - p_home, -15.0, 15.0)


def process_corner_odds(df):
    """Adds corner_total and corner_supremacy to a frame of scraped Pinnacle odds."""
    df = df.dropna(subset=["Asian Corners Line", "Asian Corners O Odds", "Asian Corners U Odds",
                           "Corner Handicap", "Home Handicap Corners Odds", "Away Handicap Corners Odds"]).copy()
    df["corner_total"] = df.apply(
        lambda r: fit_corner_total(r["Asian Corners Line"], r["Asian Corners O Odds"], r["Asian Corners U Odds"]),
        axis=1)
    df["corner_supremacy"] = df.apply(
        lambda r: fit_corner_supremacy(r["Corner Handicap"], r["Home Handicap Corners Odds"],
                                       r["Away Handicap Corners Odds"], r["corner_total"]),
        axis=1)
    return df


if __name__ == "__main__":
    # Bournemouth v Aston Villa, 7 Feb 2026
    total = fit_corner_total(10.5, 1.970, 1.862)
    sup = fit_corner_supremacy(-0.5, 1.925, 1.877, total)
    print(f"expected corners {total:.2f}, corner supremacy {sup:+.2f}")

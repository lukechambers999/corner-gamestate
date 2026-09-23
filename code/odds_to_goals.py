"""
Pre-match odds -> expected goals and supremacy.

Each match on totalcorner.com comes with a pre-match goal line and Asian
handicap, e.g. Goal line "Full (2.5, 3.0)" at 1.89 / 2.01 and handicap
"Full (-0.5, -1.0)" at 1.86 / 2.04. This file turns those into:

  - expected total goals  (the Poisson mean that prices the goal line fairly)
  - supremacy             (the expected goal difference, home minus away, that
                           prices the handicap fairly under a Skellam model)

Both are solved numerically: remove the bookmaker's margin to get a fair
probability, then find the parameter that reproduces it.
Adapted from tc_scraper/1_core_proc_web.py.
"""

from scipy.optimize import brentq
from scipy.stats import poisson, skellam


def parse_line(text):
    """'Full (2.5, 3.0)' -> 2.75 (a split line is the average of its two halves)."""
    inner = text.strip()[len("Full ("):-1]
    parts = [float(p) for p in inner.split(",")]
    return sum(parts) / len(parts)


def remove_margin(odds_a, odds_b):
    """Fair probabilities from a two-way price, scaling out the bookmaker's margin."""
    p_a, p_b = 1 / odds_a, 1 / odds_b
    return p_a / (p_a + p_b), p_b / (p_a + p_b)


def asian_over_probability(lam, line):
    """Value of an Asian 'over' bet when the total is Poisson(lam).

    Whole lines push on an exact hit (half stake back), half lines can't push,
    and quarter lines split the stake across the two neighbouring lines.
    """
    def whole(l):
        return (1 - poisson.cdf(l, lam)) + 0.5 * poisson.pmf(l, lam)

    def half(l):
        return 1 - poisson.cdf(l - 0.5, lam)

    r = line % 1.0
    if r == 0.0:
        return whole(line)
    if r == 0.5:
        return half(line)
    if r == 0.25:
        return 0.5 * (whole(line - 0.25) + half(line + 0.25))
    return 0.5 * (half(line - 0.25) + whole(line + 0.25))  # r == 0.75


def asian_home_probability(delta, line, base):
    """Value of a home Asian handicap bet when goal difference is Skellam.

    delta: expected goal difference (home - away), the unknown
    line:  handicap as quoted for the home side (negative = home favoured)
    base:  expected goals per team, so home = base + delta/2, away = base - delta/2
    """
    mu_home = max(base + delta / 2, 0.01)
    mu_away = max(base - delta / 2, 0.01)
    t = -line  # home -0.5 wins when home - away > 0.5

    def whole(l):
        return (1 - skellam.cdf(l, mu_home, mu_away)) + 0.5 * skellam.pmf(int(l), mu_home, mu_away)

    def half(l):
        return 1 - skellam.cdf(l - 0.5, mu_home, mu_away)

    r = t % 1.0
    if r == 0.0:
        return whole(t)
    if r == 0.5:
        return half(t)
    if r == 0.25:
        return 0.5 * (whole(t - 0.25) + half(t + 0.25))
    return 0.5 * (half(t - 0.25) + whole(t + 0.25))  # r == 0.75


def fit_expected_goals(line, over_odds, under_odds):
    """Expected total goals implied by the goal line."""
    p_over, _ = remove_margin(over_odds, under_odds)
    return brentq(lambda lam: asian_over_probability(lam, line) - p_over, 0.1, 15.0)


def fit_supremacy(line, home_odds, away_odds, expected_goals):
    """Expected goal difference implied by the Asian handicap."""
    p_home, _ = remove_margin(home_odds, away_odds)
    base = expected_goals / 2
    return brentq(lambda d: asian_home_probability(d, line, base) - p_home, -10.0, 10.0)


if __name__ == "__main__":
    # Newcastle v Crystal Palace, 16 Apr 2025
    goals = fit_expected_goals(parse_line("Full (2.5, 3.0)"), 1.89, 2.01)
    sup = fit_supremacy(parse_line("Full (-0.5, -1.0)"), 1.86, 2.04, goals)
    print(f"expected goals {goals:.2f}, supremacy {sup:+.2f}")

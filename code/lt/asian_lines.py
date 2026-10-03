"""
Pre-match Asian lines -> expected home and away goals.

Every match on totalcorner.com has a pre-match goal line and Asian handicap.
The goal line is solved for the Poisson total-goals rate that prices it fairly,
the handicap for the goal difference (supremacy) that prices it fairly under a
Skellam model, and the two are combined into the market's expected goals for
each side ("Asian goals"). These are the odds-based metric in the ratings.
From LT_model.ipynb (TotalCorner import and dataset merging).
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import poisson, skellam


# Converts AH Line and Goal Line strings into a single numeric value (NaN when there is no line, e.g. "N.A.")
def parse_line(value):
    if isinstance(value, str):
        value = value[6:-1]
    value = str(value)
    try:
        if ',' in value:
            a, b = (p.strip() for p in value.split(','))
            return (float(a) + float(b)) / 2
        return float(value)
    except ValueError:
        return np.nan


# Removes margin from a pair of odds to get demargined probabilities
def remove_margin(odds_a, odds_b):
    p_a_raw = 1 / odds_a
    p_b_raw = 1 / odds_b
    total = p_a_raw + p_b_raw
    return p_a_raw / total, p_b_raw / total


# Probability that total match goals go over an asian totals line
def asian_over_probability(lam, line):
    remainder = line % 1.0

    if remainder == 0.0:
        return (1 - poisson.cdf(line, lam)) + 0.5 * poisson.pmf(line, lam)
    elif remainder == 0.5:
        return 1 - poisson.cdf(line - 0.5, lam)
    elif remainder == 0.25:
        lower = line - 0.25
        upper = line + 0.25
        p_lower = (1 - poisson.cdf(lower, lam)) + 0.5 * poisson.pmf(lower, lam)
        p_upper = 1 - poisson.cdf(upper - 0.5, lam)
        return 0.5 * (p_lower + p_upper)
    else:
        lower = line - 0.25
        upper = line + 0.25
        p_lower = 1 - poisson.cdf(lower - 0.5, lam)
        p_upper = (1 - poisson.cdf(upper, lam)) + 0.5 * poisson.pmf(upper, lam)
        return 0.5 * (p_lower + p_upper)


# Solves for the total goals rate implied by an asian totals line and its over/under odds
def fit_lambda(line, over_odds, under_odds):
    if pd.isna(line) or pd.isna(over_odds) or pd.isna(under_odds):
        return np.nan
    p_over, _ = remove_margin(over_odds, under_odds)
    objective = lambda lam: asian_over_probability(lam, line) - p_over
    try:
        return brentq(objective, 0.1, 15.0)
    except ValueError:
        return np.nan


# Probability that the home team covers an asian handicap line
def skellam_home_prob(delta, line, base):
    mu_home = max(base + delta / 2, 0.01)
    mu_away = max(base - delta / 2, 0.01)

    remainder = line % 1.0
    if remainder < 0:
        remainder += 1.0

    if remainder == 0.0:
        p_win = 1 - skellam.cdf(line, mu_home, mu_away)
        p_push = skellam.pmf(int(line), mu_home, mu_away)
        return p_win + 0.5 * p_push
    elif remainder == 0.5:
        return 1 - skellam.cdf(line - 0.5, mu_home, mu_away)
    elif remainder == 0.25:
        lower = line - 0.25
        upper = line + 0.25
        p_lower = 1 - skellam.cdf(lower, mu_home, mu_away) + 0.5 * skellam.pmf(int(lower), mu_home, mu_away)
        p_upper = 1 - skellam.cdf(upper - 0.5, mu_home, mu_away)
        return 0.5 * (p_lower + p_upper)
    else:
        lower = line - 0.25
        upper = line + 0.25
        p_lower = 1 - skellam.cdf(lower - 0.5, mu_home, mu_away)
        p_upper = 1 - skellam.cdf(upper, mu_home, mu_away) + 0.5 * skellam.pmf(int(upper), mu_home, mu_away)
        return 0.5 * (p_lower + p_upper)


# Solves for the goal difference implied by an asian handicap line and its home/away odds
def fit_delta(line, home_odds, away_odds, base):
    if pd.isna(line) or pd.isna(home_odds) or pd.isna(away_odds) or pd.isna(base):
        return np.nan
    p_home, _ = remove_margin(home_odds, away_odds)
    objective = lambda delta: skellam_home_prob(delta, line, base) - p_home
    bound = max(2 * base - 0.01, 0.01)
    try:
        return brentq(objective, -bound, bound)
    except ValueError:
        return np.nan


# Adds asian total goals, supremacy and each side's expected goals to a totalcorner fixtures frame
def add_asian_goals(fixtures_tc):
    df = fixtures_tc.copy()
    for col in ['AH.Home.Odds', 'AH.Away.Odds', 'Goal.O.Odds', 'Goal.U.Odds']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['hcaplevel'] = df['AH.Line'].apply(parse_line)
    df['sup_level'] = 0 - df['hcaplevel']
    df['goallevel'] = df['Goal.Line'].apply(parse_line)

    df['asian_total_goals'] = df.apply(
        lambda row: fit_lambda(row['goallevel'], row['Goal.O.Odds'], row['Goal.U.Odds']), axis=1)
    df['asian_sup'] = df.apply(
        lambda row: fit_delta(row['sup_level'], row['AH.Home.Odds'], row['AH.Away.Odds'],
                              row['asian_total_goals'] / 2), axis=1)

    df['AsianHomeGoals'] = (df['asian_sup'] + df['asian_total_goals']) / 2
    df['AsianAwayGoals'] = df['asian_total_goals'] - df['AsianHomeGoals']
    return df


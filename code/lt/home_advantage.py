"""
Home advantage factor.

Matches played behind closed doors (COVID) are removed first, as they show
almost no home advantage. The factor is the home-minus-away goal difference as a
share of total goals, so it scales with how many goals a match is expected to
have. From LT_model.ipynb (Home advantage factor).
"""

import pandas as pd

COVID_PERIOD_START = '2020-03-01'
COVID_PERIOD_END = '2022-07-01'


# Removes matches played behind closed doors
def remove_covid_period(df, date_col='Date'):
    dates = pd.to_datetime(df[date_col])
    return df[(dates < COVID_PERIOD_START) | (dates > COVID_PERIOD_END)].copy()


# Finds average home and away goals and defines the home advantage factor
def home_advantage(fixtures_past):
    home_goals = fixtures_past['goals_h'].mean()
    away_goals = fixtures_past['goals_a'].mean()

    total_goals = home_goals + away_goals
    homeadv = home_goals - away_goals

    return float(homeadv / total_goals)

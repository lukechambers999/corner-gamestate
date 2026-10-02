"""
Team attack and defence ratings, and the expected goals they give a fixture.

Each team gets an attack strength (AS) and a defence strength (DS): a weighted
blend of three rolling averages over its last n matches:
  - goals scored / conceded
  - xG for / against (Understat)
  - Asian goals for / against (the market's pre-match expected goals)
A fixture's expected goals are AS x opposition DS / league average DS, with the
home advantage factor then shifted from the away side to the home side.
From LT_model.ipynb (Team performance dataframe, Applying optimal parameters).
"""

import pandas as pd

COLS_TO_AVG_TEAM = ['GoalsScored', 'GoalsConceded', 'xG_for', 'xG_conc', 'Asian_for', 'Asian_conc']


# Creates team focused dataframe for team strength: one row per team per match, most recent first
def team_strength_frame(fixtures_past):
    df = fixtures_past.copy()
    df['total_goals'] = df['goals_h'] + df['goals_a']
    df['total_xG'] = df['xG_h'] + df['xG_a']
    df['asian_total_goals'] = df['AsianHomeGoals'] + df['AsianAwayGoals']

    shared = ['matchid', 'Date', 'total_goals', 'total_xG', 'asian_total_goals']
    shared += ['League'] if 'League' in df else []

    home = df[shared + ['Home', 'goals_h', 'goals_a', 'xG_h', 'xG_a', 'AsianHomeGoals', 'AsianAwayGoals']].rename(columns={
        'Home': 'Team', 'goals_h': 'GoalsScored', 'goals_a': 'GoalsConceded',
        'xG_h': 'xG_for', 'xG_a': 'xG_conc', 'AsianHomeGoals': 'Asian_for', 'AsianAwayGoals': 'Asian_conc'})
    home['home_or_away'] = 'Home'

    away = df[shared + ['Away', 'goals_a', 'goals_h', 'xG_a', 'xG_h', 'AsianAwayGoals', 'AsianHomeGoals']].rename(columns={
        'Away': 'Team', 'goals_a': 'GoalsScored', 'goals_h': 'GoalsConceded',
        'xG_a': 'xG_for', 'xG_h': 'xG_conc', 'AsianAwayGoals': 'Asian_for', 'AsianHomeGoals': 'Asian_conc'})
    away['home_or_away'] = 'Away'

    team_strength = pd.concat([home, away], ignore_index=True)
    return team_strength.sort_values(['Date', 'matchid'], ascending=False).reset_index(drop=True)


# Weighted attack / defence ratings from a frame holding each metric's rolling average
def weighted_ratings(df, goals_wgt, xG_wgt, asian_wgt):
    attack = df['GoalsScored_roll'] * goals_wgt + df['xG_for_roll'] * xG_wgt + df['Asian_for_roll'] * asian_wgt
    defence = df['GoalsConceded_roll'] * goals_wgt + df['xG_conc_roll'] * xG_wgt + df['Asian_conc_roll'] * asian_wgt
    return attack, defence


# Current ratings: each team's average over its most recent n_matches, then weighted
def current_ratings(team_strength, current_teams, n_matches, goals_wgt, xG_wgt, asian_wgt):
    recent = team_strength[team_strength['Team'].isin(current_teams)].groupby('Team').head(n_matches)

    # Teams with fewer than n_matches matches get no rating (min_periods = n_matches in the notebook)
    counts = recent.groupby('Team').size()
    recent = recent[recent['Team'].isin(counts[counts == n_matches].index)]

    # Means skip missing values: promoted teams' second-division rows have Asian goals only
    ratings = recent.groupby('Team')[COLS_TO_AVG_TEAM].mean().add_suffix('_roll').reset_index()

    # A team with no goals or xG in its window falls back on its Asian goals average
    for col, asian in [('GoalsScored', 'Asian_for'), ('xG_for', 'Asian_for'),
                       ('GoalsConceded', 'Asian_conc'), ('xG_conc', 'Asian_conc')]:
        ratings[f'{col}_roll'] = ratings[f'{col}_roll'].fillna(ratings[f'{asian}_roll'])
    ratings['weight_AS'], ratings['weight_DS'] = weighted_ratings(ratings, goals_wgt, xG_wgt, asian_wgt)
    ratings['DS_avg'] = ratings['weight_DS'].mean()
    return ratings


# Predicted match goals and supremacy, with the home advantage factor applied
def predict_goals(home_AS, home_DS, away_AS, away_DS, DS_avg, HA):
    home_pred = (home_AS * away_DS) / DS_avg
    away_pred = (away_AS * home_DS) / DS_avg

    # Moves HA x total goals from the away side to the home side, keeping the total unchanged
    total_HA = (home_pred + away_pred) * HA
    home_pred = home_pred + total_HA / 2
    away_pred = away_pred - total_HA / 2

    return home_pred, away_pred, home_pred + away_pred, home_pred - away_pred

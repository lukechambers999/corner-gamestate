"""
Finding the window length and metric weights that best predict the market.

For every combination of window length n and goals / xG / Asian weights
(summing to 1), each past match is predicted from ratings built only on the
matches before it. The score is the mean absolute error against the market's
own total and supremacy for that match, with supremacy weighted at 0.75
because the model is meant for finding supremacy differences.
Selection uses seasons before 2023/24; the chosen combination is then scored
on 2023/24 onwards. From LT_model.ipynb (Finding optimal parameters).
"""

from itertools import product

import numpy as np
import pandas as pd

from ratings import COLS_TO_AVG_TEAM, predict_goals

TRAIN_TEST_SPLIT = '2023-08-01'
N_MATCHES_RANGE = range(5, 30)
WGT_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


# Weight combinations that sum to 1 (rounded, so float sums like 0.1 + 0.2 + 0.7 aren't dropped)
def weight_combinations():
    return [w for w in product(WGT_RANGE, WGT_RANGE, WGT_RANGE) if round(sum(w), 5) == 1]


# Pre-match rolling averages: the mean of each team's n matches strictly before each match
def pre_match_rolls(team_strength, n_matches, n_teams_div):
    ts = team_strength.copy()
    for col in COLS_TO_AVG_TEAM:
        ts[f'{col}_roll'] = ts.groupby('Team', group_keys=False)[col].transform(
            lambda x: x.shift(-1)[::-1].rolling(window=n_matches, min_periods=n_matches).mean()[::-1])

    # League goals average over the previous n_matches * n_teams_div team-rows
    w = n_matches * n_teams_div
    ts['total_goals_roll'] = ts['total_goals'].shift(-1)[::-1].rolling(window=w, min_periods=w).mean()[::-1]
    return ts


# Joins home and away pre-match rolls onto each fixture
def fixtures_with_rolls(fixtures, ts):
    roll_cols = [f'{c}_roll' for c in COLS_TO_AVG_TEAM]
    home = ts[ts['home_or_away'] == 'Home'][['matchid', 'total_goals_roll'] + roll_cols]
    away = ts[ts['home_or_away'] == 'Away'][['matchid'] + roll_cols]
    df = fixtures.merge(home.add_prefix('Home_').rename(columns={'Home_matchid': 'matchid',
                                                                 'Home_total_goals_roll': 'total_goals_roll'}),
                        on='matchid', how='left')
    df = df.merge(away.add_prefix('Away_').rename(columns={'Away_matchid': 'matchid'}), on='matchid', how='left')
    return df.dropna()


# MAE of total and supremacy predictions against the asian lines for one weight combination
def score(df, goals_wgt, xG_wgt, asian_wgt, HA):
    w = {'GoalsScored': goals_wgt, 'GoalsConceded': goals_wgt, 'xG_for': xG_wgt,
         'xG_conc': xG_wgt, 'Asian_for': asian_wgt, 'Asian_conc': asian_wgt}

    def blend(side, cols):
        return sum(df[f'{side}_{c}_roll'] * w[c] for c in cols)

    home_AS, home_DS = blend('Home', ['GoalsScored', 'xG_for', 'Asian_for']), blend('Home', ['GoalsConceded', 'xG_conc', 'Asian_conc'])
    away_AS, away_DS = blend('Away', ['GoalsScored', 'xG_for', 'Asian_for']), blend('Away', ['GoalsConceded', 'xG_conc', 'Asian_conc'])
    avg_ds_goals = df['total_goals_roll'] / 2

    _, _, total_pred, sup_pred = predict_goals(home_AS, home_DS, away_AS, away_DS, avg_ds_goals, HA)
    mae_total = np.mean(np.abs(total_pred - df['asian_total_goals']))
    mae_sup = np.mean(np.abs(sup_pred - df['asian_sup']))
    return mae_total, mae_sup


# Grid search on the training seasons, then scores the best combination on the test seasons
def find_optimal_parameters(fixtures_past, team_strength, n_teams_div, HA):
    fixtures_past = fixtures_past.copy()
    fixtures_past['asian_total_goals'] = fixtures_past['AsianHomeGoals'] + fixtures_past['AsianAwayGoals']
    fixtures_past['asian_sup'] = fixtures_past['AsianHomeGoals'] - fixtures_past['AsianAwayGoals']
    train = fixtures_past[fixtures_past['Date'] < TRAIN_TEST_SPLIT]
    test = fixtures_past[fixtures_past['Date'] > TRAIN_TEST_SPLIT]

    results = []
    for n_matches in N_MATCHES_RANGE:
        df = fixtures_with_rolls(train, pre_match_rolls(team_strength, n_matches, n_teams_div))
        for goals_wgt, xG_wgt, asian_wgt in weight_combinations():
            mae_total, mae_sup = score(df, goals_wgt, xG_wgt, asian_wgt, HA)
            results.append({'n_matches': n_matches, 'goals_wgt': goals_wgt, 'xG_wgt': xG_wgt,
                            'asian_wgt': asian_wgt, 'mae_total': mae_total, 'mae_sup': mae_sup})

    results_df = pd.DataFrame(results)

    # Preference for finding sup diffs
    results_df['comb_mae'] = results_df['mae_total'] * 0.25 + results_df['mae_sup'] * 0.75
    best = results_df.sort_values('comb_mae').iloc[0]

    df_test = fixtures_with_rolls(test, pre_match_rolls(team_strength, int(best['n_matches']), n_teams_div))
    mae_total_test, mae_sup_test = score(df_test, best['goals_wgt'], best['xG_wgt'], best['asian_wgt'], HA)

    return best, (mae_total_test, mae_sup_test), results_df

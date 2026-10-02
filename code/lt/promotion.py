"""
Promoted teams: carrying second-division form into the top flight.

A newly promoted team has few or no top-flight matches to rate it on. Its
second-division matches are used instead, scaled to top-flight level:
  - every past promotion is found (second division one season, top flight
    the next)
  - for each one, the team's average Asian goals for and against in its
    first top-flight season are compared with its promotion season
  - the average ratios (for and against) become the promotion factors, and
    second-division Asian goals in a promotion season are multiplied by them
Only Asian goals are adjusted, since Understat has no second-division goals
or xG. Adapted from LT_model_master_v2.py (Finding promoted teams,
Promoted teams adjustment factor).
"""

import numpy as np
import pandas as pd


# Season a match belongs to, by its starting year (August onwards)
def season_year(dates):
    dates = pd.to_datetime(dates)
    return np.where(dates.dt.month >= 8, dates.dt.year, dates.dt.year - 1)


# Team-seasons in the second division that were followed by promotion to the top flight
def find_promotions(team_strength, league_top, league_below):
    team_season_league = team_strength[['Team', 'season_year', 'League']].drop_duplicates()
    top = team_season_league[team_season_league['League'] == league_top][['Team', 'season_year']]
    below = team_season_league[team_season_league['League'] == league_below][['Team', 'season_year']]

    promoted = top.merge(below.assign(season_year=below['season_year'] + 1), on=['Team', 'season_year'])
    promoted['below_season'] = promoted['season_year'] - 1
    return promoted.rename(columns={'season_year': 'top_season'}).reset_index(drop=True)


# Average ratio of top-flight to second-division Asian goals across past promotions
def promotion_factors(team_strength, promotions, league_top, league_below):
    def season_means(league, season_col):
        rows = team_strength[team_strength['League'] == league].merge(
            promotions[['Team', season_col]].rename(columns={season_col: 'season_year'}), on=['Team', 'season_year'])
        return rows.groupby(['Team', 'season_year'])[['Asian_for', 'Asian_conc']].mean()

    below = season_means(league_below, 'below_season').reset_index()
    below['season_year'] += 1  # aligns each promotion season with the top-flight season after it
    top = season_means(league_top, 'top_season').reset_index()

    comparison = top.merge(below, on=['Team', 'season_year'], suffixes=('_top', '_below'))
    comparison['adj_factor_for'] = comparison['Asian_for_top'] / comparison['Asian_for_below']
    comparison['adj_factor_conc'] = comparison['Asian_conc_top'] / comparison['Asian_conc_below']

    return comparison['adj_factor_for'].mean(), comparison['adj_factor_conc'].mean(), comparison


# Scales second-division Asian goals in promotion seasons, and drops other second-division rows
def apply_promotion_adjustment(team_strength, promotions, league_below, promo_for_adj, promo_conc_adj):
    df = team_strength.merge(
        promotions[['Team', 'below_season']].rename(columns={'below_season': 'season_year'}).assign(Next_Promoted=True),
        on=['Team', 'season_year'], how='left')
    df['Next_Promoted'] = df['Next_Promoted'].fillna(False).astype(bool)

    adjust = (df['League'] == league_below) & df['Next_Promoted']
    df.loc[adjust, 'Asian_for'] *= promo_for_adj
    df.loc[adjust, 'Asian_conc'] *= promo_conc_adj

    return df[(df['League'] != league_below) | adjust].reset_index(drop=True)

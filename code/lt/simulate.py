"""
Monte Carlo simulation of the rest of the season.

Each remaining fixture is drawn as a home win, draw or away win from its 1X2
probabilities, the points are added to the current table, and the final
positions are recorded. Repeated N times, the share of simulations in which a
team finishes 1st, top 4, top 6 or bottom 3 is its probability, and 1 /
probability is the fair price. Teams level on points are ordered at random.
A vectorised version of LT_model.ipynb (Monte Carlo simulation of rest of season).
"""

import numpy as np
import pandas as pd


def simulate_season(table, fixtures, n_sims=10000, seed=None):
    rng = np.random.default_rng(seed)
    teams = list(table.index)
    idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    home = fixtures['Home'].map(idx).to_numpy()
    away = fixtures['Away'].map(idx).to_numpy()
    p_home = fixtures['home_pc'].to_numpy()
    p_draw = fixtures['draw_pc'].to_numpy()

    # One uniform draw per fixture per simulation decides the outcome
    u = rng.random((n_sims, len(fixtures)))
    home_win = u < p_home
    draw = (u >= p_home) & (u < p_home + p_draw)
    away_win = ~home_win & ~draw

    points = np.tile(table['Pts'].to_numpy(dtype=float), (n_sims, 1))
    rows = np.arange(n_sims)[:, None]
    np.add.at(points, (rows, home[None, :]), 3 * home_win + draw)
    np.add.at(points, (rows, away[None, :]), 3 * away_win + draw)

    # Random tie-break: a tiny jitter that can't overturn a whole point
    order = np.argsort(-(points + rng.random(points.shape) * 0.01), axis=1)
    position = np.empty_like(order)
    position[rows, order] = np.arange(n_teams)

    results = pd.DataFrame({
        '1st': (position == 0).mean(axis=0),
        'Top 4': (position < 4).mean(axis=0),
        'Top 6': (position < 6).mean(axis=0),
        'Bottom 3': (position >= n_teams - 3).mean(axis=0),
    }, index=pd.Index(teams, name='Team'))

    for col in ['1st', 'Top 4', 'Top 6', 'Bottom 3']:
        results[f'{col} odds'] = (1 / results[col]).replace(np.inf, np.nan)
    return results

"""
Scoreline probabilities and 1X2 prices, with a Dixon-Coles low-score correction.

Two independent Poissons give each scoreline's probability from the home and
away expected goals. That under-prices low-scoring draws, so Dixon and Coles'
tau factor reweights 0-0, 0-1, 1-0 and 1-1 using one parameter, rho, fitted by
maximum likelihood on past matches' actual scorelines (with the market's
expected goals as the inputs). From LT_model.ipynb (Calculating a draw
adjustment factor, End of season table prediction).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 11
LOW_SCORELINES = [(0, 0), (0, 1), (1, 0), (1, 1)]


# Dixon-Coles low-score correction factor
def dixon_coles_tau(home_goals, away_goals, home_exp, away_exp, rho):
    tau = np.ones_like(np.asarray(home_exp, dtype=float))
    tau = np.where((home_goals == 0) & (away_goals == 0), 1 - (home_exp * away_exp * rho), tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1 + (home_exp * rho), tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1 + (away_exp * rho), tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, tau)
    return tau


# Negative log-likelihood of the actual scorelines for a given rho
def dixon_coles_neg_log_likelihood(rho, home_goals, away_goals, home_exp, away_exp):
    tau = dixon_coles_tau(home_goals, away_goals, home_exp, away_exp, rho)
    probs = tau * poisson.pmf(home_goals, home_exp) * poisson.pmf(away_goals, away_exp)
    return -np.log(np.clip(probs, 1e-10, None)).sum()


# Fits rho by maximising the likelihood of the training set's actual scorelines
def fit_rho(train):
    args = (train['goals_h'].to_numpy(), train['goals_a'].to_numpy(),
            train['AsianHomeGoals'].to_numpy(), train['AsianAwayGoals'].to_numpy())
    return minimize_scalar(dixon_coles_neg_log_likelihood, bounds=(-1, 1), method='bounded', args=args).x


# Scoreline probability matrix (rows = home goals, cols = away goals) with Dixon-Coles adjustments
def score_matrix(home_exp, away_exp, rho):
    home_probs = poisson.pmf(range(MAX_GOALS), home_exp)
    away_probs = poisson.pmf(range(MAX_GOALS), away_exp)

    matrix = np.outer(home_probs, away_probs)
    for home_goals, away_goals in LOW_SCORELINES:
        matrix[home_goals, away_goals] *= dixon_coles_tau(home_goals, away_goals, home_exp, away_exp, rho)

    return matrix / matrix.sum()


# Home / draw / away probabilities from the adjusted scoreline matrix
def match_probabilities(home_exp, away_exp, rho):
    matrix = score_matrix(home_exp, away_exp, rho)
    home = np.sum(np.tril(matrix, -1))
    draw = np.sum(np.diag(matrix))
    away = np.sum(np.triu(matrix, 1))
    return home, draw, away

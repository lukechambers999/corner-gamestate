"""
Predicting corners from ratings and supremacy.

Two OLS equations, one per side:
    home corners ~ adjsup + home attack rating + away defence rating
    away corners ~ adjsup + away attack rating + home defence rating

The target is always raw corners, whichever metric built the ratings. Each
(metric, window) rating set is fitted on the earliest 80% of matches by date
and scored by R² on the latest 20%.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

TRAIN_FRACTION = 0.8
SIDES = {  # side -> (target, attack rating, opponent defence rating)
    "home": ("home_corners_raw", "home_corner_attack_rating", "away_corner_defence_rating"),
    "away": ("away_corners_raw", "away_corner_attack_rating", "home_corner_defence_rating"),
}


def fit_side(train: pd.DataFrame, side: str):
    target, attack, defence = SIDES[side]
    X = sm.add_constant(train[["adjsup", attack, defence]])
    return sm.OLS(train[target], X).fit()


def predict_side(model, df: pd.DataFrame, side: str) -> pd.Series:
    _, attack, defence = SIDES[side]
    return model.predict(sm.add_constant(df[["adjsup", attack, defence]], has_constant="add"))


def out_of_sample_r2(df: pd.DataFrame, side: str) -> float:
    """Chronological train/test split; R² of test-set predictions."""
    split = df["Date"].quantile(TRAIN_FRACTION)
    train, test = df[df["Date"] <= split], df[df["Date"] > split]
    model = fit_side(train, side)
    target = SIDES[side][0]
    resid = test[target] - predict_side(model, test, side)
    return 1 - (resid ** 2).sum() / ((test[target] - test[target].mean()) ** 2).sum()


def predict_supremacy(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """
    Predicted corner supremacy (home - away) for the betting backtest. The OLS
    intercept absorbs the training era's average corner level, so each side's
    prediction is rescaled by current league baseline / training-era baseline.
    """
    preds = {}
    for side, baseline in (("home", "league_home_baseline"), ("away", "league_away_baseline")):
        raw = predict_side(fit_side(train, side), test, side)
        preds[side] = raw * test[baseline] / train[baseline].mean()
    return preds["home"] - preds["away"]

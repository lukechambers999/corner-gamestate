"""
Builds the JSON snapshots in ../snapshots/lt/ that the Long-term model page runs on.

Run from the repo root:  .venv/Scripts/python _build/make_lt_snapshots.py [--src ../LT_Model_App]

Sources (not committed; default path is relative to this repo's parent folder):
  LT_Model_App/fixtures_data_past_filt.csv  - league history: goals, xG and Asian (odds-implied) goals
  LT_Model_App/currentseason_past.csv       - this season's results
  LT_Model_App/fixtures_data_future.csv     - remaining fixtures

The browser recomputes ratings, prices and simulations from the per-team match
history in the JSON. The values fitted here (home advantage, rho and the
optimised window/weights) are the page's fixed inputs and defaults.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "lt"))
from home_advantage import remove_covid_period, home_advantage  # noqa: E402
from ratings import COLS_TO_AVG_TEAM, team_strength_frame, current_ratings, predict_goals  # noqa: E402
from optimise import TRAIN_TEST_SPLIT, find_optimal_parameters  # noqa: E402
from dixon_coles import fit_rho, match_probabilities  # noqa: E402
from tables import current_table, expected_table  # noqa: E402

OUT = ROOT / "snapshots" / "lt"
WINDOW_CAP = 30

# One entry per league; more are added here as the live data pipeline comes online
LEAGUES = {
    "serie_a": {"name": "Serie A", "league_tc": "ItalySerieA", "n_teams_div": 20, "as_of": "2026-04-01"},
}


def r(x, nd=6):
    return None if pd.isna(x) else round(float(x), nd)


def build(key, cfg, src):
    past = pd.read_csv(src / "fixtures_data_past_filt.csv", parse_dates=["Date"])
    results = pd.read_csv(src / "currentseason_past.csv")
    fixtures = pd.read_csv(src / "fixtures_data_future.csv")

    # Past matches with full data, closed-doors period removed
    past = remove_covid_period(past.dropna(subset=["goals_h", "goals_a", "xG_h", "xG_a"]))
    past["matchid"] = past["Date"].dt.strftime("%Y-%m-%d") + "-" + past["Home"] + "-" + past["Away"]

    HA = home_advantage(past)
    rho = float(fit_rho(past[past["Date"] < TRAIN_TEST_SPLIT]))
    team_strength = team_strength_frame(past)

    best, (mae_total_test, mae_sup_test), _ = find_optimal_parameters(past, team_strength, cfg["n_teams_div"], HA)
    n_best = int(best["n_matches"])
    w_best = (float(best["goals_wgt"]), float(best["xG_wgt"]), float(best["asian_wgt"]))
    print(f"{key}: HA={HA:.4f} rho={rho:.4f} best n={n_best} weights={w_best} "
          f"test MAE total={mae_total_test:.4f} sup={mae_sup_test:.4f}")

    teams = sorted(set(results["Home"]) | set(results["Away"]))
    history = team_strength[team_strength["Team"].isin(teams)]
    window_max = int(min(WINDOW_CAP, history.groupby("Team").size().min()))

    team_json = []
    for team in teams:
        rows = history[history["Team"] == team].head(window_max)
        team_json.append({"team": team, **{c: [r(v, 5) for v in rows[c]] for c in COLS_TO_AVG_TEAM}})

    # Python reference outputs at the defaults, for checking the browser port
    ratings = current_ratings(team_strength, teams, n_best, *w_best).set_index("Team")
    fx = fixtures[["Date", "Home", "Away"]].copy()
    hp, ap, tot, sup = predict_goals(ratings.loc[fx["Home"], "weight_AS"].to_numpy(), ratings.loc[fx["Home"], "weight_DS"].to_numpy(),
                                     ratings.loc[fx["Away"], "weight_AS"].to_numpy(), ratings.loc[fx["Away"], "weight_DS"].to_numpy(),
                                     ratings["DS_avg"].iloc[0], HA)
    fx["Home_Pred_Goals"], fx["Away_Pred_Goals"] = hp, ap
    fx[["home_pc", "draw_pc", "away_pc"]] = [match_probabilities(h, a, rho) for h, a in zip(hp, ap)]
    table = current_table(teams, results)
    exp_table = expected_table(table, fx)

    data = {
        "meta": {
            "key": key, "league": cfg["name"], "as_of": cfg["as_of"], "HA": r(HA), "rho": r(rho),
            "window_max": window_max, "window_min": 5,
            "defaults": {"n_matches": n_best, "goals_wgt": w_best[0], "xG_wgt": w_best[1], "asian_wgt": w_best[2]},
            "test_mae": {"total": r(mae_total_test, 4), "sup": r(mae_sup_test, 4), "split": TRAIN_TEST_SPLIT},
            "n_history_matches": int(len(past)),
            "history_from": past["Date"].min().strftime("%Y-%m-%d"),
        },
        "teams": team_json,
        "results": results[["Date", "Home", "Away", "goals_h", "goals_a"]].to_dict("records"),
        "fixtures": fixtures[["Date", "Home", "Away"]].to_dict("records"),
        "check": {
            "ratings": {t: [r(ratings.loc[t, "weight_AS"]), r(ratings.loc[t, "weight_DS"])] for t in ratings.index},
            "fixtures": [[r(x) for x in row] for row in fx[["Home_Pred_Goals", "Away_Pred_Goals", "home_pc", "draw_pc", "away_pc"]].head(10).to_numpy()],
            "expected_pts": {t: r(exp_table.loc[t, "Pts"]) for t in exp_table.index},
        },
    }
    (OUT / f"{key}.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return {"key": key, "league": cfg["name"], "file": f"{key}.json"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT.parent / "LT_Model_App")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = [build(key, cfg, args.src) for key, cfg in LEAGUES.items()]
    (OUT / "index.json").write_text(json.dumps({"leagues": index}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

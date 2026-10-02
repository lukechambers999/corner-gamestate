"""
Builds the JSON snapshots in ../snapshots/lt/ that the Long-term model page runs on.

Run from the repo root:  .venv/Scripts/python _build/make_lt_snapshots.py [--src ../LT_Model_App]

Sources (not committed; default path is relative to this repo's parent folder):
  LT_Model_App/fixtures_data_past_filt.csv  - league history: goals, xG and Asian (odds-implied) goals
  LT_Model_App/currentseason_past.csv       - this season's results
  LT_Model_App/fixtures_data_future.csv     - remaining fixtures
  LT_Model_App/<league_tc_below>.csv        - second-division totalcorner data, for promoted teams
  LT_Model_App/sup_conversion.csv, goallines.csv - line -> expected goals lookups for that data

The browser recomputes ratings, prices and simulations from the per-team match
history in the JSON. The values fitted here (home advantage, rho and the
optimised window/weights) are the page's fixed inputs and defaults.

Promoted teams' histories include their promotion season in the division
below, with Asian goals scaled by the promotion factors (code/lt/promotion.py).
The grid search still runs on top-flight matches only, as in v2.
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
from asian_lines import add_asian_goals_lookup  # noqa: E402
from promotion import season_year, find_promotions, promotion_factors, apply_promotion_adjustment  # noqa: E402

OUT = ROOT / "snapshots" / "lt"
WINDOW_CAP = 38  # one season

# One entry per league; more are added here as the live data pipeline comes online
LEAGUES = {
    "serie_a": {"name": "Serie A", "league_tc": "ItalySerieA", "league_tc_below": "ItalySerieB",
                "n_teams_div": 20, "as_of": "2026-04-01"},
}


def r(x, nd=6):
    return None if pd.isna(x) else round(float(x), nd)


# Second-division matches with Asian goals from the totalcorner lines (no goals or xG available)
def load_below(src, cfg):
    tc = pd.read_csv(src / f"{cfg['league_tc_below']}.csv",
                     usecols=["Date", "Home", "Away", "AH.Line", "AH.Home.Odds", "AH.Away.Odds", "Goal.Line", "Goal.O.Odds"])
    tc["Date"] = pd.to_datetime(tc["Date"], format="%d.%m.%Y")
    tc["matchid"] = tc["Date"].dt.strftime("%Y-%m-%d") + "-" + tc["Home"] + "-" + tc["Away"]
    tc = tc.drop_duplicates(subset="matchid")
    tc = tc[tc["Date"] < cfg["as_of"]]

    sup_conv = pd.read_csv(src / "sup_conversion.csv", encoding="utf-8-sig")
    goals_conv = pd.read_csv(src / "goallines.csv", encoding="utf-8-sig")
    tc = add_asian_goals_lookup(tc, sup_conv, goals_conv).dropna(subset=["AsianHomeGoals", "AsianAwayGoals"])
    for col in ["goals_h", "goals_a", "xG_h", "xG_a"]:
        tc[col] = np.nan
    tc["League"] = cfg["league_tc_below"]
    return tc[["Date", "matchid", "League", "Home", "Away", "goals_h", "goals_a", "xG_h", "xG_a",
               "AsianHomeGoals", "AsianAwayGoals"]]


# Team histories across both divisions, with promotion seasons scaled to top-flight level
def promotion_adjusted_history(past_all, below, cfg):
    ts = team_strength_frame(pd.concat([past_all, below], ignore_index=True))
    ts["season_year"] = season_year(ts["Date"])
    promotions = find_promotions(ts, cfg["league_tc"], cfg["league_tc_below"])
    promo_for, promo_conc, comparison = promotion_factors(ts, promotions, cfg["league_tc"], cfg["league_tc_below"])
    ts = apply_promotion_adjustment(ts, promotions, cfg["league_tc_below"], promo_for, promo_conc)
    ts = remove_covid_period(ts).sort_values(["Date", "matchid"], ascending=False).reset_index(drop=True)
    return ts, promo_for, promo_conc, len(comparison)


def build(key, cfg, src):
    past = pd.read_csv(src / "fixtures_data_past_filt.csv", parse_dates=["Date"])
    results = pd.read_csv(src / "currentseason_past.csv")
    fixtures = pd.read_csv(src / "fixtures_data_future.csv")

    # Past matches with full data, closed-doors period removed
    past_all = past.dropna(subset=["goals_h", "goals_a", "xG_h", "xG_a"]).copy()
    past_all["matchid"] = past_all["Date"].dt.strftime("%Y-%m-%d") + "-" + past_all["Home"] + "-" + past_all["Away"]
    past_all["League"] = cfg["league_tc"]
    past = remove_covid_period(past_all)

    HA = home_advantage(past)
    rho = float(fit_rho(past[past["Date"] < TRAIN_TEST_SPLIT]))
    team_strength = team_strength_frame(past)

    best, (mae_total_test, mae_sup_test), _ = find_optimal_parameters(past, team_strength, cfg["n_teams_div"], HA)
    n_best = int(best["n_matches"])
    w_best = (float(best["goals_wgt"]), float(best["xG_wgt"]), float(best["asian_wgt"]))
    print(f"{key}: HA={HA:.4f} rho={rho:.4f} best n={n_best} weights={w_best} "
          f"test MAE total={mae_total_test:.4f} sup={mae_sup_test:.4f}")

    history_all, promo_for, promo_conc, n_promotions = promotion_adjusted_history(past_all, load_below(src, cfg), cfg)
    print(f"{key}: promotion factors for={promo_for:.4f} conc={promo_conc:.4f} from {n_promotions} promotions")

    teams = sorted(set(results["Home"]) | set(results["Away"]))
    history = history_all[history_all["Team"].isin(teams)]
    window_max = int(min(WINDOW_CAP, history.groupby("Team").size().min()))

    team_json = []
    for team in teams:
        rows = history[history["Team"] == team].head(window_max)
        team_json.append({"team": team, **{c: [r(v, 5) for v in rows[c]] for c in COLS_TO_AVG_TEAM},
                          "below": [int(lg == cfg["league_tc_below"]) for lg in rows["League"]]})

    # Python reference outputs at the defaults, for checking the browser port
    ratings = current_ratings(history_all, teams, n_best, *w_best).set_index("Team")
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
            "league_below": cfg["league_tc_below"],
            "promotion": {"for": r(promo_for, 4), "conc": r(promo_conc, 4), "n": n_promotions},
        },
        "teams": team_json,
        "results": results[["Date", "Home", "Away", "goals_h", "goals_a"]].to_dict("records"),
        "fixtures": fixtures[["Date", "Home", "Away"]].to_dict("records"),
        "check": {
            "ratings": {t: [r(ratings.loc[t, "weight_AS"]), r(ratings.loc[t, "weight_DS"])] for t in ratings.index},
            # Longest window, where promoted teams' windows reach into the division below
            "ratings_window_max": {t: [r(row.weight_AS), r(row.weight_DS)] for t, row in
                                   current_ratings(history_all, teams, window_max, *w_best).set_index("Team").iterrows()},
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

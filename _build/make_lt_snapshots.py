"""
Builds the JSON snapshots in ../snapshots/lt/ that the Long-term model page runs on,
one per top-5 league, from live data.

Run from the repo root:
  .venv/Scripts/python _build/make_lt_snapshots.py --tc <dir> [--us-cache <dir>] [--leagues EPL Serie_A]

Sources:
  Understat (scraped here)      - every match's goals and xG since 2015/16, and this season's fixtures
  <tc>/<League>.csv             - totalcorner_data_finished_master CSVs from Google Drive
                                  (gdrive:/TC_Scraper/totalcorner_data_finished_master/), for the
                                  pre-match Asian handicap and goal line, top flight and division below

The scheduled workflow (.github/workflows/update_lt_data.yml) runs this daily.

The browser recomputes ratings, prices and simulations from the per-team match
history in the JSON. The values fitted here (home advantage, rho and the
optimised window/weights) are the page's fixed inputs and defaults.

Promoted teams' histories include their promotion season in the division
below, with Asian goals scaled by the promotion factors (code/lt/promotion.py).
The grid search and rho use top-flight matches with betting lines only, as in
the notebook. Recent matches whose lines haven't been scraped yet stay in the
rating history: their goals and xG count, and the odds average skips them.
"""

import argparse
import json
import sys
from datetime import date
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
from asian_lines import add_asian_goals  # noqa: E402
from promotion import season_year, find_promotions, promotion_factors, apply_promotion_adjustment  # noqa: E402
import understat  # noqa: E402

OUT = ROOT / "snapshots" / "lt"
WINDOW_CAP = 38  # one season
WINDOW_MIN = 5
TC_COLS = ["Date", "Home", "Away", "AH.Line", "AH.Home.Odds", "AH.Away.Odds", "Goal.Line", "Goal.O.Odds", "Goal.U.Odds"]


def r(x, nd=6):
    return None if pd.isna(x) else round(float(x), nd)


# One row per totalcorner match with the market's expected goals for each side
def load_tc(tc_dir, league_file):
    tc = pd.read_csv(tc_dir / f"{league_file}.csv", usecols=TC_COLS)
    tc["Date"] = pd.to_datetime(tc["Date"], format="%d.%m.%Y")
    tc = add_asian_goals(tc.drop_duplicates(subset=["Date", "Home", "Away"]))
    tc["League"] = league_file
    return tc[["Date", "Home", "Away", "League", "AsianHomeGoals", "AsianAwayGoals"]]


# Joins each Understat match to its totalcorner lines: same teams, kick-off dates within two days
# (the two sites can put a late kick-off either side of midnight)
def attach_lines(matches, tc):
    lines = tc.dropna(subset=["AsianHomeGoals", "AsianAwayGoals"]).sort_values("Date")
    merged = pd.merge_asof(matches.sort_values("Date"), lines.drop(columns="League"), on="Date", by=["Home", "Away"],
                           tolerance=pd.Timedelta(days=2), direction="nearest")
    return merged.sort_values(["Date", "Home"]).reset_index(drop=True)


# Understat matches for one league, scraped or read from a local cache
def understat_matches(us_league, cache):
    today = date.today()
    last_season = today.year if today.month >= 7 else today.year - 1
    path = cache / f"{us_league}.pkl" if cache else None
    if path and path.exists():
        raw = pd.read_pickle(path)
    else:
        raw = understat.scrape_league(us_league, last_season)
        if path:
            cache.mkdir(parents=True, exist_ok=True)
            raw.to_pickle(path)
    return understat.clean(raw)


# Second-division matches with Asian goals only (Understat has no second-division goals or xG)
def load_below(tc_dir, cfg, as_of):
    tc = load_tc(tc_dir, cfg["league_tc_below"]).dropna(subset=["AsianHomeGoals", "AsianAwayGoals"])
    tc = tc[tc["Date"] <= as_of].copy()
    for col in ["goals_h", "goals_a", "xG_h", "xG_a"]:
        tc[col] = np.nan
    tc["matchid"] = tc["Date"].dt.strftime("%Y-%m-%d") + "-" + tc["Home"] + "-" + tc["Away"]
    return tc


# Team histories across both divisions, with promotion seasons scaled to top-flight level
def promotion_adjusted_history(past_all, below, cfg, current):
    ts = team_strength_frame(pd.concat([past_all, below], ignore_index=True))
    ts["season_year"] = season_year(ts["Date"])
    promotions = find_promotions(ts, cfg["league_tc"], cfg["league_tc_below"], current=current)
    promo_for, promo_conc, comparison = promotion_factors(ts, promotions, cfg["league_tc"], cfg["league_tc_below"])
    ts = apply_promotion_adjustment(ts, promotions, cfg["league_tc_below"], promo_for, promo_conc)
    ts = remove_covid_period(ts).sort_values(["Date", "matchid"], ascending=False).reset_index(drop=True)
    return ts, promo_for, promo_conc, len(comparison)


def build(us_league, cfg, tc_dir, cache):
    key = us_league.lower()
    as_of = pd.Timestamp(date.today())
    us = understat_matches(us_league, cache)

    # Current season: the latest one Understat has fixtures or results for
    season = us["season"].max()
    current = us[us["season"] == season]
    teams = sorted(set(current["Home"]) | set(current["Away"]))
    results = current[current["isResult"]].sort_values("Date")
    fixtures = current[~current["isResult"]].sort_values(["Date", "Home"])

    # Every played match with its betting lines where totalcorner has them
    tc = load_tc(tc_dir, cfg["league_tc"])
    played = us[us["isResult"]].dropna(subset=["goals_h", "goals_a", "xG_h", "xG_a"])
    past_all = attach_lines(played, tc)
    past_all["matchid"] = past_all["Date"].dt.strftime("%Y-%m-%d") + "-" + past_all["Home"] + "-" + past_all["Away"]
    past_all["League"] = cfg["league_tc"]

    # Home advantage on every non-COVID result; rho and the grid search on matches with lines
    past = remove_covid_period(past_all)
    lined = past.dropna(subset=["AsianHomeGoals", "AsianAwayGoals"])
    HA = home_advantage(past)
    rho = float(fit_rho(lined[lined["Date"] < TRAIN_TEST_SPLIT]))
    best, (mae_total_test, mae_sup_test), _ = find_optimal_parameters(lined, team_strength_frame(lined), cfg["n_teams_div"], HA)
    n_best = int(best["n_matches"])
    w_best = (float(best["goals_wgt"]), float(best["xG_wgt"]), float(best["asian_wgt"]))

    no_lines = results.merge(past_all[past_all["AsianHomeGoals"].isna()][["Date", "Home", "Away"]], on=["Date", "Home", "Away"])
    print(f"{key}: {season} - {len(results)} results ({len(no_lines)} without lines), {len(fixtures)} fixtures. "
          f"{len(lined)}/{len(past)} non-COVID matches matched to lines. HA={HA:.4f} rho={rho:.4f} "
          f"best n={n_best} weights={w_best} test MAE total={mae_total_test:.4f} sup={mae_sup_test:.4f}")

    season_start = int(season[:4])
    history_all, promo_for, promo_conc, n_promotions = promotion_adjusted_history(
        past_all, load_below(tc_dir, cfg, as_of), cfg, current=(season_start, teams))
    print(f"{key}: promotion factors for={promo_for:.4f} conc={promo_conc:.4f} from {n_promotions} promotions")

    history = history_all[history_all["Team"].isin(teams)]
    counts = history.groupby("Team").size().reindex(teams, fill_value=0)
    short = counts[counts < WINDOW_MIN]
    if len(short):
        raise SystemExit(f"{key}: too little history for {short.to_dict()} - check team names in understat.NAME_CONV")
    window_max = int(min(WINDOW_CAP, counts.min()))
    n_best = min(n_best, window_max)

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
    fx[["home_pc", "draw_pc", "away_pc"]] = [match_probabilities(h, a, rho) for h, a in zip(hp, ap)] if len(fx) else np.empty((0, 3))
    exp_table = expected_table(current_table(teams, results), fx)

    as_records = lambda df, cols: df[cols].assign(Date=df["Date"].dt.strftime("%Y-%m-%d")).to_dict("records")
    data = {
        "meta": {
            "key": key, "league": cfg["name"], "season": season,
            "lines_to": tc["Date"].max().strftime("%Y-%m-%d"), "n_results_no_lines": int(len(no_lines)),
            "HA": r(HA), "rho": r(rho), "window_max": window_max, "window_min": WINDOW_MIN,
            "defaults": {"n_matches": n_best, "goals_wgt": w_best[0], "xG_wgt": w_best[1], "asian_wgt": w_best[2]},
            "test_mae": {"total": r(mae_total_test, 4), "sup": r(mae_sup_test, 4), "split": TRAIN_TEST_SPLIT},
            "n_history_matches": int(len(past)), "n_lined_matches": int(len(lined)),
            "history_from": past["Date"].min().strftime("%Y-%m-%d"),
            "league_below": cfg["league_tc_below"],
            "promotion": {"for": r(promo_for, 4), "conc": r(promo_conc, 4), "n": n_promotions},
        },
        "teams": team_json,
        "results": as_records(results, ["Date", "Home", "Away", "goals_h", "goals_a"]),
        "fixtures": as_records(fixtures, ["Date", "Home", "Away"]),
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
    ap.add_argument("--tc", type=Path, required=True, help="folder holding the totalcorner_data_finished_master CSVs")
    ap.add_argument("--us-cache", type=Path, help="reuse / save Understat scrapes here instead of always scraping")
    ap.add_argument("--leagues", nargs="+", default=list(understat.LEAGUES), choices=list(understat.LEAGUES))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    index = [build(lg, understat.LEAGUES[lg], args.tc, args.us_cache) for lg in args.leagues]
    # The build date lives only in index.json, so the league files change only when the data does
    if set(args.leagues) == set(understat.LEAGUES):
        (OUT / "index.json").write_text(json.dumps({"updated": date.today().isoformat(), "leagues": index}, indent=1),
                                        encoding="utf-8")


if __name__ == "__main__":
    main()

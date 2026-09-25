"""
Builds the small CSV snapshots in ../snapshots/ that the site renders from.

Run from the repo root:  .venv/Scripts/python _build/make_snapshots.py

Sources (not committed; paths are relative to this repo's parent folder):
  portfolio/data/corner_gamestate.csv, portfolio/data/mins_conv.csv  - notebook-era event data
  tc_scraper/*.csv, tc_scraper/corner_matches_adjusted/            - full-model results
"""

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from gamestate import process  # noqa: E402
from repeaters import flag_repeaters, repeater_rates  # noqa: E402
from adjusted_values import supremacy_buckets, adjusted_values  # noqa: E402

sys.path.insert(0, str(ROOT))
from _charts import SHOWCASE_LEAGUES  # noqa: E402

RESEARCH = ROOT.parent
PORTFOLIO_DATA = RESEARCH / "portfolio" / "data"
TC = RESEARCH / "tc_scraper"
OUT = ROOT / "snapshots"
OUT.mkdir(exist_ok=True)

SAMPLE_MATCH = "2025-02-12-Everton-Liverpool"
CASE_STUDY_MATCH = "2024-08-27-Rayo Vallecano-Barcelona"
# Chosen for lopsided corner splits: most of a team's corners came in one short spell of winning or losing
TEAM_SAMPLE_MATCHES = ["2020-06-29-Crystal Palace-Burnley", "2024-10-19-Tottenham-West Ham"]
REPEATER_MATCH = "2025-02-15-Aston Villa-Ipswich"
TC_SAMPLE_URL ="https://www.totalcorner.com/match/corner-stats/190784578"  # Aston Villa v West Ham, 22 Mar 2026
PIN_SAMPLE_DATES = ("2026-02-07", "2026-02-08")  # one Premier League weekend
# Spread from the highest to the lowest raw home attack ratings; all have a full rating on every metric
RATINGS_SAMPLE_TEAMS = ["Man City", "Liverpool", "Arsenal", "Chelsea", "Newcastle", "Bournemouth",
                        "Everton", "Crystal Palace"]
RATINGS_WINDOW, RATINGS_MIN_PERIODS = 25, 10  # the adj_nr / 25-game setting carried forward in section 5
# One weekend of live predictions (13-15 Sep 2026, from the 18 Sep daily run) with Pinnacle prices attached:
# a mix of flagged value bets and matches where model and market agree, across the big five leagues
PREDICTIONS_SAMPLE = [("Leeds", "Newcastle"), ("Man Utd", "Man City"), ("Coventry", "Brighton"),
                      ("Levante", "Barcelona"), ("Elche", "Real Madrid"), ("Celta Vigo", "Malaga"),
                      ("Napoli", "Bologna"), ("Inter Milan", "Udinese"), ("Elversberg", "Bayern Munich"),
                      ("Brest", "PSG"), ("Le Mans", "Lens")]


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"  {name:40s} {len(df):>6} rows")


def team_repeater_rates(min_matches=100):
    """Repeater share per team across the 12 showcase leagues, from the full model's per-match files."""
    rows = []
    for lg in SHOWCASE_LEAGUES:
        m = pd.read_csv(TC / "corner_matches_adjusted" / f"{lg}_matches.csv")
        for side in ["home", "away"]:
            rows.append(pd.DataFrame({"League": lg, "Team": m[side.capitalize()],
                                      "corners": m[f"{side}_corners_raw"], "repeaters": m[f"{side}_reps"]}))
    rt = (pd.concat(rows).groupby(["League", "Team"])
          .agg(matches=("corners", "size"), corners=("corners", "sum"), repeaters=("repeaters", "sum"))
          .reset_index())
    rt["repeater_rate"] = rt["repeaters"] / rt["corners"]
    return rt[rt["matches"] >= min_matches]


def epl_current_ratings():
    """Current Premier League team ratings on all four metrics, as in the tc_scraper dashboard's Team ratings tab."""
    sys.path.insert(0, str(TC))
    sys.path.insert(0, str(TC / "dashboards"))
    from rating_backtest_web import load_all_matches
    from predictions.current_ratings import compute_current_team_ratings
    from build_current_ratings import METRICS

    cwd = os.getcwd()
    os.chdir(TC)  # load_all_matches reads corner_matches_adjusted/ relative to tc_scraper
    try:
        all_matches = load_all_matches()
    finally:
        os.chdir(cwd)

    rows = []
    for metric, (home_col, away_col, allow_cross_league) in METRICS.items():
        home, away = compute_current_team_ratings(
            all_matches, window_size=RATINGS_WINDOW, min_periods=RATINGS_MIN_PERIODS,
            home_col=home_col, away_col=away_col, allow_cross_league=allow_cross_league)
        # raw/nr keep one row per league a team has played in; keep only the Premier League one
        home = home[home["home_current_league"] == "EnglandPremierLeague"]
        away = away[away["away_current_league"] == "EnglandPremierLeague"]
        m = home.merge(away, on=["Team", "Country"])
        m = m[m["Team"].isin(RATINGS_SAMPLE_TEAMS)]
        m["metric"] = metric
        rows.append(m)
    out = pd.concat(rows)
    out["order"] = out["Team"].map({t: i for i, t in enumerate(RATINGS_SAMPLE_TEAMS)})
    assert out.groupby("Team").size().eq(len(METRICS)).all() and out["Team"].nunique() == len(RATINGS_SAMPLE_TEAMS)
    return out.sort_values(["order", "metric"])[
        ["Team", "metric", "home_corner_attack_rating", "home_corner_defense_rating",
         "away_corner_attack_rating", "away_corner_defence_rating", "home_rating_as_of", "away_rating_as_of"]]


def predictions_sample():
    """A sample of the daily predictions file behind the tc_scraper dashboard's Predictions tab."""
    pr = pd.read_csv(TC / "predictions" / "current_predictions.csv")
    pr["order"] = pd.Series(list(zip(pr["Home"], pr["Away"]))).map({m: i for i, m in enumerate(PREDICTIONS_SAMPLE)})
    pr = pr[pr["order"].notna() & pr["matched"]].sort_values("order")
    assert len(pr) == len(PREDICTIONS_SAMPLE), "sample fixtures missing from current_predictions.csv"
    return pr[["Date", "League", "Home", "Away", "adjsup", "predicted_home_corners", "predicted_away_corners",
               "predicted_supremacy", "corner_supremacy", "diff_sup", "Corner Handicap",
               "Home Handicap Corners Odds", "Away Handicap Corners Odds", "p_cover_home", "p_cover_away",
               "ev_home", "ev_away", "value_bet"]]


def main():
    if sys.argv[1:] == ["ratings"]:  # rebuild only this snapshot
        save(epl_current_ratings(), "epl_current_ratings.csv")
        return
    if sys.argv[1:] == ["predictions"]:
        save(predictions_sample(), "predictions_sample.csv")
        return

    print("Loading event data...")
    raw = pd.read_csv(PORTFOLIO_DATA / "corner_gamestate.csv")
    raw_events = raw[raw["matchid"] == SAMPLE_MATCH][["Minute", "Team", "Event"]]

    df = flag_repeaters(process(raw))
    print(f"  {df['matchid'].nunique()} matches after validation")

    # 1. Worked example: one match's events with running score/state
    ex = df[df["matchid"] == SAMPLE_MATCH]
    save(ex[["Minute", "Team", "Event", "home_goals", "away_goals", "gamestate"]], "example_match_events.csv")
    m = ex.iloc[0]
    save(pd.DataFrame([{
        "Home": m["Home"], "Away": m["Away"],
        "Home leading": m["homeleadmins"], "Level": m["drawingmins"], "Away leading": m["awayleadmins"],
        "Home corners": ex[(ex["Event"] == "Corner") & (ex["Team"] == m["Home"])].shape[0],
        "Away corners": ex[(ex["Event"] == "Corner") & (ex["Team"] == m["Away"])].shape[0],
    }]), "example_match_minutes.csv")
    assert len(raw_events) == len(ex), "sample match was dropped by validation"

    # Team-level minutes and corners per state, for a few sample matches
    tm = df[df["matchid"].isin(TEAM_SAMPLE_MATCHES)].groupby("teammatchid").first().reset_index()
    tm["order"] = tm["matchid"].map({m: i for i, m in enumerate(TEAM_SAMPLE_MATCHES)})
    tm["is_away"] = tm["Team"] == tm["Away"]
    tm = tm.sort_values(["order", "is_away"])
    save(tm[["Date", "Home", "Away", "Team", "teamsup", "teamleadingmins", "teamdrawingmins", "teamlosingmins",
             "team_leading_corners", "team_drawing_corners", "team_losing_corners"]], "team_state_sample.csv")

    # Repeater example: every corner in one match, with its repeater flag
    rp = df[(df["matchid"] == REPEATER_MATCH) & (df["Event"] == "Corner")]
    save(rp[["Minute", "half", "Team", "Event", "Repeater"]], "repeater_example.csv")

    # 2. Data coverage
    matches = df.groupby("matchid").first()
    cov = matches.groupby("League").agg(
        matches=("Home", "size"), first=("Date", "min"), last=("Date", "max")
    ).reset_index()
    cov["corners"] = df[df["Event"] == "Corner"].groupby("League").size().values
    save(cov, "coverage.csv")

    # 3. Expected vs actual state minutes, EPL 2023/24 first 9 rounds
    team = df.groupby("teammatchid").first().reset_index()
    epl = team[(team["League"] == "England Premier League")
               & (team["Date"] > "2023-07-01") & (team["Date"] < "2023-11-01")].copy()
    mins_conv = pd.read_csv(PORTFOLIO_DATA / "mins_conv.csv")
    epl["hcaplevel"] = -epl["hcaplevel"]
    epl = epl.merge(mins_conv, on=["hcaplevel", "goallevel"], how="left")
    is_home = epl["Team"] == epl["Home"]
    epl["exp_lead"] = epl["home_leading"].where(is_home, epl["away_leading"])
    epl["exp_lose"] = epl["away_leading"].where(is_home, epl["home_leading"])
    epl["exp_draw"] = epl["drawing_mins"]
    rows = []
    for basis, cols in {"Expected": ["exp_lead", "exp_draw", "exp_lose"],
                        "Actual": ["teamleadingmins", "teamdrawingmins", "teamlosingmins"]}.items():
        g = epl.groupby("Team")[cols].sum()
        g = g.div(g.sum(axis=1), axis=0)
        g.columns = ["Leading", "Level", "Trailing"]
        g["basis"] = basis
        rows.append(g.reset_index())
    save(pd.concat(rows), "epl_2324_state_minutes.csv")

    # 4. Repeaters
    save(repeater_rates(df, "League"), "repeaters_by_league.csv")
    save(team_repeater_rates(), "repeaters_by_team.csv")

    # 5. La Liga supremacy x state buckets, and the Rayo-Barcelona case study
    liga_events = df[df["League"] == "Spain La Liga"]
    liga_team = team[(team["League"] == "Spain La Liga")
                     & ((team["Date"] > "2021-07-01") | (team["Date"] < "2020-04-01"))]  # exclude closed-doors period
    home_b = supremacy_buckets(liga_team[liga_team["Team"] == liga_team["Home"]]).assign(side="Home")
    away_b = supremacy_buckets(liga_team[liga_team["Team"] == liga_team["Away"]]).assign(side="Away")

    valued = adjusted_values(liga_events, home_b, away_b)
    case = valued[valued["matchid"] == CASE_STUDY_MATCH][
        ["Minute", "Team", "teamsup", "team_state", "Repeater", "expected_rate", "avg_rate", "adj_corner_value"]]
    save(case, "case_study_corners.csv")

    # 6. Raw scraper output samples, one match / matchday each
    tc_raw = pd.read_csv(PORTFOLIO_DATA / "EnglandPremierLeague.csv", dtype=str)
    tc_match = tc_raw[tc_raw["Game.URL"] == TC_SAMPLE_URL]
    save(tc_match, "sample_totalcorner_events.csv")

    pin_all = pd.read_excel(RESEARCH / "pin_scraper" / "pinnacle_corners_historical.xlsx")
    pin = pin_all[(pin_all["League"] == "EnglandPremierLeague")
                  & pin_all["Date"].between(*PIN_SAMPLE_DATES)].sort_values(["Date", "Home"])
    save(pin, "sample_pinnacle_corners.csv")
    save(pd.DataFrame([{"matches": len(pin_all), "leagues": pin_all["League"].nunique(),
                        "first": pin_all["Date"].min().date(), "last": pin_all["Date"].max().date()}]),
         "pinnacle_coverage.csv")

    # 7. Full-model results from tc_scraper (copied as-is or lightly reduced)
    print("Copying tc_scraper results...")
    for name in ["gs_home_corner_share.csv", "gs_coefficients.csv", "corner_adj_qa_summary.csv",
                 "bet_backtest_summary.csv", "bet_backtest_results.csv", "ev_variant_peak_pnl_curves.csv"]:
        shutil.copy(TC / name, OUT / name)
        print(f"  {name}")

    buckets = pd.read_csv(TC / "gs_coefficient_buckets.csv")
    save(buckets[buckets["League"].isin(SHOWCASE_LEAGUES)], "gs_buckets_showcase.csv")

    rb = pd.read_csv(TC / "rating_backtest_results_all_leagues.csv")
    save(rb[rb["scope"] == "global"][["side", "metric", "window_size", "n_train", "n_test",
                                     "r2_test", "rmse_test", "mae_test"]], "rating_backtest_global.csv")


if __name__ == "__main__":
    main()

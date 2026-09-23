"""
Builds the small CSV snapshots in ../snapshots/ that the site renders from.

Run from the repo root:  .venv/Scripts/python _build/make_snapshots.py

Sources (not committed; paths are relative to this repo's parent folder):
  portfolio/data/corner_gamestate.csv, portfolio/data/mins_conv.csv  - notebook-era event data
  tc_scraper/*.csv                                                    - full-model results
"""

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from gamestate import process  # noqa: E402
from repeaters import flag_repeaters, repeater_rates  # noqa: E402
from adjusted_values import supremacy_buckets, adjusted_values  # noqa: E402

RESEARCH = ROOT.parent
PORTFOLIO_DATA = RESEARCH / "portfolio" / "data"
TC = RESEARCH / "tc_scraper"
OUT = ROOT / "snapshots"
OUT.mkdir(exist_ok=True)

SAMPLE_MATCH = "2025-02-12-Everton-Liverpool"
CASE_STUDY_MATCH = "2024-08-27-Rayo Vallecano-Barcelona"


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(f"  {name:40s} {len(df):>6} rows")


def main():
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
    rt = repeater_rates(df, ["League", "Team"])
    save(rt[rt["matches"] > 50], "repeaters_by_team.csv")

    # 5. La Liga supremacy x state buckets, and the Rayo-Barcelona case study
    liga_events = df[df["League"] == "Spain La Liga"]
    liga_team = team[(team["League"] == "Spain La Liga")
                     & ((team["Date"] > "2021-07-01") | (team["Date"] < "2020-04-01"))]  # exclude closed-doors period
    home_b = supremacy_buckets(liga_team[liga_team["Team"] == liga_team["Home"]]).assign(side="Home")
    away_b = supremacy_buckets(liga_team[liga_team["Team"] == liga_team["Away"]]).assign(side="Away")
    save(pd.concat([home_b, away_b]), "laliga_sup_buckets.csv")

    valued = adjusted_values(liga_events, home_b, away_b)
    case = valued[valued["matchid"] == CASE_STUDY_MATCH][
        ["Minute", "Team", "teamsup", "team_state", "Repeater", "expected_rate", "avg_rate", "adj_corner_value"]]
    save(case, "case_study_corners.csv")

    # 6. Full-model results from tc_scraper (copied as-is or lightly reduced)
    print("Copying tc_scraper results...")
    for name in ["gs_home_corner_share.csv", "gs_coefficients.csv", "corner_adj_qa_summary.csv",
                 "bet_backtest_summary.csv", "bet_backtest_results.csv", "ev_variant_peak_pnl_curves.csv"]:
        shutil.copy(TC / name, OUT / name)
        print(f"  {name}")

    rb = pd.read_csv(TC / "rating_backtest_results_all_leagues.csv")
    save(rb[rb["scope"] == "global"][["side", "metric", "window_size", "n_train", "n_test",
                                     "r2_test", "rmse_test", "mae_test"]], "rating_backtest_global.csv")


if __name__ == "__main__":
    main()

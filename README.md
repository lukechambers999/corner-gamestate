# Corners & Game State

Source for the write-up at **https://lukechambers999.github.io/corner-gamestate/**,
plus an interactive Long-term season model page (`long-term-model.qmd`).
It covers building corner ratings that correct for game state, supremacy and
repeater corners, and backtesting them against Pinnacle's corner markets.

## Layout

| Path | What it is |
|---|---|
| `index.qmd` | The article: prose, tables and charts (code hidden) |
| `methodology.qmd` | Definitions and backtest design |
| `snapshots/` | Small CSVs that every chart and table reads from |
| `code/` | Processing code for the original study, linked from the article |
| `_build/make_snapshots.py` | Regenerates `snapshots/` from the raw data (needs the private data folders next to this repo) |
| `_charts.py` | Shared Plotly styling |
| `long-term-model.qmd` | Long-term model page: write-up plus the embedded interactive app |
| `code/lt/` | Long-term model code (adapted from `portfolio/LT_model.ipynb`), linked from that page |
| `lt_app/` | The in-browser app: `model.js` mirrors `code/lt/`, `app.js` is the UI |
| `snapshots/lt/` | Per-league JSON the app runs on (`index.json` lists the leagues) |
| `_build/make_lt_snapshots.py` | Regenerates `snapshots/lt/` for the top-5 leagues: scrapes Understat, joins the totalcorner lines, fits HA, rho, the default window/weights and the promotion factors |
| `.github/workflows/update_lt_data.yml` | Daily Action: pulls the totalcorner CSVs from Google Drive, runs the build, commits `snapshots/lt/` if it changed and republishes |

## Editing

```sh
py -3.13 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
set QUARTO_PYTHON=.venv\Scripts\python.exe
quarto preview           # live-reloading local preview
```

Edit the text in `index.qmd`, then commit and push. The GitHub Action re-renders
the site and publishes it to the `gh-pages` branch.

## Long-term model data

The `Update Long-term model data` Action rebuilds `snapshots/lt/` every day at
07:23 UTC, after tc_scraper's daily pipeline has updated
`gdrive:/TC_Scraper/totalcorner_data_finished_master/`. It needs the
`RCLONE_CONFIG` repo secret (base64 of an `rclone.conf` with a `gdrive` remote,
the same as tc_scraper's). Run it by hand with `gh workflow run update_lt_data.yml`.

To build locally, pull the ten top-flight and second-division CSVs and run:

```sh
.venv/Scripts/pip install -r _build/requirements_lt.txt
rclone copy gdrive:/TC_Scraper/totalcorner_data_finished_master/ tc_data/ --include "{EnglandPremierLeague,EnglandChampionship,SpainLaLiga,SpainSegunda,ItalySerieA,ItalySerieB,GermanyBundesligaI,GermanyBundesligaII,FranceLigue1,FranceLigue2}.csv"
.venv/Scripts/python _build/make_lt_snapshots.py --tc tc_data [--us-cache <dir>] [--leagues EPL Serie_A]
```

`--us-cache` saves the Understat scrapes so repeat runs skip them. If the build
stops with "too little history" for a team, its Understat spelling needs adding
to `NAME_CONV` in `code/lt/understat.py`. In the browser console on the page,
`ltCheck()` reports the largest difference between the JS and Python outputs.

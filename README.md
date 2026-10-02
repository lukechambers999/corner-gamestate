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
| `_build/make_lt_snapshots.py` | Regenerates `snapshots/lt/` from `../LT_Model_App/` CSVs, fitting HA, rho and the default window/weights |

## Editing

```sh
py -3.13 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
set QUARTO_PYTHON=.venv\Scripts\python.exe
quarto preview           # live-reloading local preview
```

Edit the text in `index.qmd`, then commit and push. The GitHub Action re-renders
the site and publishes it to the `gh-pages` branch.

## Long-term model data

```sh
.venv/Scripts/python _build/make_lt_snapshots.py --src ../LT_Model_App
```

The current snapshot is a fixed Serie A demo (season cut off at 2026-04-01).
To add a league, add it to `LEAGUES` in the build script; the page's league
picker appears automatically once `index.json` lists more than one. The plan
for live data is a scheduled Action that scrapes Understat + totalcorner,
re-runs the build script for all top-5 leagues, and commits `snapshots/lt/`
(which triggers the publish workflow). In the browser console on the page,
`ltCheck()` reports the largest difference between the JS and Python outputs.

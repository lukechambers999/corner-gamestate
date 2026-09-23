# Corners & Game State

Source for the write-up at **https://lukechambers999.github.io/corner-gamestate/**.
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

## Editing

```sh
py -3.13 -m venv .venv && .venv/Scripts/pip install -r requirements.txt
set QUARTO_PYTHON=.venv\Scripts\python.exe
quarto preview           # live-reloading local preview
```

Edit the text in `index.qmd`, then commit and push. The GitHub Action re-renders
the site and publishes it to the `gh-pages` branch.

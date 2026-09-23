"""Shared chart styling and helpers for the site (imported by the .qmd pages)."""

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import HTML, Markdown, display

SNAP = Path(__file__).resolve().parent / "snapshots"
REPO = "https://github.com/lukechambers999/corner-gamestate/blob/main"

# Game state -> colour. Fixed everywhere on the site.
STATE_COLORS = {"Leading": "#1baf7a", "Level": "#2a78d6", "Trailing": "#eb6834"}
ACCENT = "#2a78d6"
MUTED = "#8a8984"
POS, NEG = "#2a78d6", "#e34948"  # diverging poles (positive / negative)

TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"

pio.templates["site"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=TEXT_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="", automargin=True),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, ticks="", automargin=True),
        legend=dict(orientation="h", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font_color=TEXT, bordercolor=GRID),
        bargap=0.25,
        title=dict(font=dict(color=TEXT, size=15), x=0, xanchor="left"),
    )
)
pio.templates.default = "plotly_white+site"
CONFIG = {"displaylogo": False, "responsive": True,
          "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def show(fig):
    """Render a figure, lifting its title out into an HTML caption so it can't collide with the legend."""
    title = fig.layout.title.text
    if title:
        display(HTML(f'<p class="chart-title">{title}</p>'))
        fig.update_layout(title=None)
    has_legend = fig.layout.showlegend is not False and sum(t.showlegend is not False for t in fig.data) > 1
    has_subtitles = any(a.yref == "paper" and a.y == 1 for a in fig.layout.annotations)
    if has_legend and fig.layout.legend.y is None:
        # Pin the legend to the top of the container, above any subplot titles
        fig.update_layout(legend=dict(yref="container", y=0.995, yanchor="top"))
    fig.update_layout(margin_t=(36 if has_legend else 8) + (26 if has_subtitles else 0)
                      + (44 if fig.layout.updatemenus else 0))
    fig.show(config=CONFIG)


LEAGUE_OVERRIDES = {"PolandILiga": "Poland I Liga", "Turkiye1Lig": "Turkiye 1. Lig"}


def pretty_league(name):
    """'PortugalSegundaLiga' -> 'Portugal Segunda Liga', 'Austria2.Liga' -> 'Austria 2. Liga'."""
    if name in LEAGUE_OVERRIDES:
        return LEAGUE_OVERRIDES[name]
    s = re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=\.)(?=[A-Z])", " ", name)
    return s.replace("HNL", " HNL").replace("NBII", "NB II").replace("NBI", "NB I").replace("  ", " ").strip()


def load(name, **kw):
    return pd.read_csv(SNAP / name, **kw)


def table(df, floatfmt=".2f"):
    """Render a DataFrame as a Markdown table (Quarto styles it)."""
    return Markdown(df.to_markdown(index=False, floatfmt=floatfmt, intfmt=","))


def code_link(path, label=None):
    return Markdown(f"[{label or 'Code'} &rarr; `{path}`]({REPO}/{path}){{.code-link}}")

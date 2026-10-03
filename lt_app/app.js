// Interactive Season Outrights Model: controls, tabs and tables. The model itself is in model.js.
// model.js inherits this script's ?v= version tag, so a deploy refreshes both files together.
const { teamRatings, priceFixtures, currentTable, expectedTable, simulateSeason, checkAgainstPython } =
  await import("./model.js" + new URL(import.meta.url).search);

const REPO = "https://github.com/lukechambers999/portfolio_2026/blob/main";
const SNAP = "snapshots/lt/";
const SEED = 1;

// Code links shown under each tab: [label, path with line range]
const CODE = {
  ratings: [
    ["Team performance frame and rolling ratings", "code/lt/ratings.py#L19-L70"],
    ["Promoted teams: promotion factors", "code/lt/promotion.py"],
    ["Window and weight grid search", "code/lt/optimise.py#L73-L98"],
    ["Browser version", "lt_app/model.js#L6-L32"],
  ],
  fixtures: [
    ["Expected goals with home advantage", "code/lt/ratings.py#L73-L83"],
    ["Dixon-Coles scorelines and 1X2", "code/lt/dixon_coles.py#L20-L62"],
    ["Odds-implied (Asian) goals", "code/lt/asian_lines.py"],
  ],
  table: [
    ["Current and expected tables", "code/lt/tables.py"],
  ],
  season: [
    ["Monte Carlo simulation", "code/lt/simulate.py"],
    ["Browser version", "lt_app/model.js#L141-L183"],
  ],
};

const fmt = {
  n: (v, d = 2) => (v == null || !isFinite(v) ? "–" : v.toFixed(d)),
  signed: (v, d = 2) => (v > 0 ? "+" : "") + v.toFixed(d),
  pct: (p) => (p <= 0 ? "–" : p < 0.001 ? "<0.1%" : (p * 100).toFixed(1) + "%"),
  odds: (p) => (p <= 0 ? "–" : 1 / p > 1000 ? "1000+" : (1 / p).toFixed(2)),
  date: (s) => new Date(s + "T12:00:00").toLocaleDateString("en-GB", { weekday: "short", day: "numeric", month: "short" }),
};

const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k instanceof Node ? k : document.createTextNode(k));
  return e;
};

function codeLinks(key) {
  return el("div", { class: "lt-code" },
    el("span", { class: "lt-code-lbl" }, "Code"),
    CODE[key].map(([label, path]) =>
      el("a", { href: `${REPO}/${path}`, target: "_blank", rel: "noopener" }, label, " → ", el("code", {}, path.split("#")[0]))));
}

function table(headers, rows, opts = {}) {
  const thead = el("thead", {}, el("tr", {}, headers.map((h, i) => {
    const th = el("th", { class: h.cls || "" }, h.label);
    if (opts.onSort && h.key) {
      th.classList.add("sortable");
      if (opts.sortKey === h.key) th.classList.add(opts.sortDir > 0 ? "asc" : "desc");
      th.addEventListener("click", () => opts.onSort(h.key));
    }
    return th;
  })));
  return el("div", { class: "table-wrap" + (opts.scroll ? " scroll" : "") }, el("table", { class: "lt-table" }, thead, el("tbody", {}, rows)));
}

// ── State ──────────────────────────────────────────────────────────────
const state = { data: null, params: null, tab: "table", sims: 10000, simResult: null, simParams: null, ratingSort: ["AS", -1] };

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const out = {};
  for (const k of ["n", "g", "x"]) if (h.has(k)) out[k] = parseFloat(h.get(k));
  if (h.has("tab")) out.tab = h.get("tab");
  if (h.has("lg")) out.lg = h.get("lg");
  return out;
}

function writeHash() {
  const p = state.params;
  history.replaceState(null, "", `#lg=${state.data.meta.key}&n=${p.n}&g=${p.g}&x=${p.x}&tab=${state.tab}`);
}

const round2 = (v) => Math.round(v * 100) / 100;
const paramsKey = (p) => `${p.n}|${p.g}|${p.x}`;

// ── Compute ────────────────────────────────────────────────────────────
function compute() {
  const { data, params } = state;
  const o = round2(1 - params.g - params.x);
  const ratings = teamRatings(data.teams, params.n, params.g, params.x, o);
  const priced = priceFixtures(data.fixtures, ratings, data.meta.HA, data.meta.rho);
  const current = currentTable(data.teams.map((t) => t.team), data.results);
  const expected = expectedTable(current, priced);
  return { o, ratings, priced, current, expected };
}

// ── Tabs ───────────────────────────────────────────────────────────────
function ratingsTab(c) {
  const [key, dir] = state.ratingSort;
  const rows = [...c.ratings.rows].map((r) => ({ ...r, net: r.AS - r.DS })).sort((a, b) => dir * (a[key] - b[key]));
  const maxAS = Math.max(...rows.map((r) => r.AS)), maxDS = Math.max(...rows.map((r) => r.DS));
  const bar = (v, max, cls) => el("span", { class: `lt-bar ${cls}`, style: `width:${(v / max) * 100}%` });
  const sortBy = (k) => {
    state.ratingSort = state.ratingSort[0] === k ? [k, -state.ratingSort[1]] : [k, k === "DS" ? 1 : -1];
    render();
  };
  return el("div", {},
    el("p", { class: "lt-note" },
      `Attack = expected goals scored per match, defence = expected goals conceded. Each is a weighted average for the given weights for each metric over the team's last ${state.params.n} league matches.` +
      (c.ratings.rows.some((r) => r.below)
        ? " Teams tagged P are newly promoted. The number shows how many matches in their window come from the division below, derived using odds based ratings adjusted to top-flight level."
        : "")),
    table(
      [{ label: "#" }, { label: "Team" }, { label: "Attack", key: "AS", cls: "num" }, { label: "", cls: "barcol" },
       { label: "Defence", key: "DS", cls: "num" }, { label: "", cls: "barcol" }, { label: "Net", key: "net", cls: "num" }],
      rows.map((r, i) => el("tr", {},
        el("td", { class: "muted" }, i + 1), el("td", { class: "team" }, r.team,
          r.below ? el("span", { class: "lt-tag", title: `${r.below} of the last ${state.params.n} matches are from the division below, scaled by the promotion factors` }, `P ${r.below}`) : null),
        el("td", { class: "num" }, fmt.n(r.AS)), el("td", { class: "barcol" }, bar(r.AS, maxAS, "att")),
        el("td", { class: "num" }, fmt.n(r.DS)), el("td", { class: "barcol" }, bar(r.DS, maxDS, "def")),
        el("td", { class: "num" }, fmt.signed(r.net)))),
      { onSort: sortBy, sortKey: key, sortDir: dir }),
    codeLinks("ratings"));
}

function fixturesTab(c) {
  const rows = c.priced.map((f) => {
    const fav = Math.max(f.p.home, f.p.draw, f.p.away);
    const cell = (p) => el("td", { class: "num odds" + (p === fav ? " fav" : "") }, fmt.odds(p));
    return el("tr", {},
      el("td", { class: "muted" }, fmt.date(f.Date)),
      el("td", { class: "team" }, f.Home), el("td", { class: "team" }, f.Away),
      el("td", { class: "num" }, fmt.n(f.home)), el("td", { class: "num" }, fmt.n(f.away)),
      el("td", { class: "num" }, fmt.n(f.total)), el("td", { class: "num" }, fmt.signed(f.sup)),
      el("td", { class: "num muted" }, fmt.pct(f.p.home)), el("td", { class: "num muted" }, fmt.pct(f.p.draw)),
      el("td", { class: "num muted" }, fmt.pct(f.p.away)),
      cell(f.p.home), cell(f.p.draw), cell(f.p.away));
  });
  return el("div", {},
    el("p", { class: "lt-note" },
      "Predicted goals for each side, total match goals and goal supremacy (the difference in predicted goals), the 1X2 probabilities and unmargined decimal odds."),
    table([{ label: "Date" }, { label: "Home" }, { label: "Away" }, { label: "H xG", cls: "num" }, { label: "A xG", cls: "num" },
           { label: "Total", cls: "num" }, { label: "Sup", cls: "num" }, { label: "H", cls: "num" }, { label: "D", cls: "num" },
           { label: "A", cls: "num" }, { label: "1", cls: "num" }, { label: "X", cls: "num" }, { label: "2", cls: "num" }], rows,
          { scroll: true }),
    codeLinks("fixtures"));
}

function tableTab(c) {
  const curPos = Object.fromEntries(c.current.map((r, i) => [r.team, i + 1]));
  return el("div", {},
    el("p", { class: "lt-note" },
      `Current points plus the expected points from the ${c.priced.length} remaining fixtures. The arrow shows the change from the current position.`),
    table([{ label: "#" }, { label: "Team" }, { label: "W", cls: "num" },
           { label: "D", cls: "num" }, { label: "L", cls: "num" }, { label: "GF", cls: "num" }, { label: "GA", cls: "num" },
           { label: "GD", cls: "num" }, { label: "So far", cls: "num" }, { label: "Pred pts", cls: "num" }],
      c.expected.map((r, i) => {
        const move = curPos[r.team] - (i + 1);
        return el("tr", {},
          el("td", { class: "muted" }, i + 1,
            move ? el("span", { class: move > 0 ? "lt-up" : "lt-down" }, move > 0 ? ` ▲${move}` : ` ▼${-move}`) : null),
          el("td", { class: "team" }, r.team),
          el("td", { class: "num" }, fmt.n(r.W, 1)), el("td", { class: "num" }, fmt.n(r.D, 1)),
          el("td", { class: "num" }, fmt.n(r.L, 1)), el("td", { class: "num" }, fmt.n(r.GF, 1)), el("td", { class: "num" }, fmt.n(r.GA, 1)),
          el("td", { class: "num" }, fmt.signed(r.GD, 1)), el("td", { class: "num muted" }, r.cur.Pts),
          el("td", { class: "num strong" }, fmt.n(r.Pts, 1)));
      })),
    codeLinks("table"));
}

function seasonTab(c) {
  const stale = state.simResult && state.simParams !== paramsKey(state.params);
  const run = () => {
    const btn = document.querySelector("#lt-run");
    btn.disabled = true;
    btn.textContent = "Running…";
    setTimeout(() => {
      const t0 = performance.now();
      state.simResult = simulateSeason(c.current, c.priced, state.sims, SEED);
      state.simMs = performance.now() - t0;
      state.lastSims = state.sims;
      state.simParams = paramsKey(state.params);
      render();
    }, 20);
  };
  const controls = el("div", { class: "lt-simbar" },
    el("label", {}, "Simulations ",
      el("select", { onchange: (e) => (state.sims = +e.target.value) },
        [1000, 5000, 10000, 20000].map((n) => {
          const o = el("option", { value: n }, n.toLocaleString());
          if (n === state.sims) o.selected = true;
          return o;
        }))),
    el("button", { id: "lt-run", class: "lt-btn primary", onclick: run }, state.simResult ? "Re-run" : "Run simulations"),
    stale ? el("span", { class: "lt-badge" }, "Parameters changed. Re-run to update") : null);

  let body;
  if (!state.simResult) {
    body = el("p", { class: "lt-empty" }, "Run the simulations to price each team's chances of winning the league, finishing in the top 4 or top 6, or being relegated (bottom 3).");
  } else {
    const avgPos = (r) => r.positions.reduce((a, p, i) => a + p * (i + 1), 0);
    const rows = [...state.simResult].sort((a, b) => avgPos(a) - avgPos(b));
    const pair = (p) => [el("td", { class: "num muted" }, fmt.pct(p)), el("td", { class: "num odds" }, fmt.odds(p))];
    body = table(
      [{ label: "Team" }, { label: "Avg pos", cls: "num" }, { label: "Winner", cls: "num grp" }, { label: "odds", cls: "num" },
       { label: "Top 4", cls: "num grp" }, { label: "odds", cls: "num" }, { label: "Top 6", cls: "num grp" }, { label: "odds", cls: "num" },
       { label: "Bottom 3", cls: "num grp" }, { label: "odds", cls: "num" }],
      rows.map((r) => el("tr", { class: stale ? "stale" : "" },
        el("td", { class: "team" }, r.team), el("td", { class: "num" }, fmt.n(avgPos(r), 1)),
        pair(r.first), pair(r.top4), pair(r.top6), pair(r.bottom3))));
  }
  return el("div", {},
    el("p", { class: "lt-note" },
      "Each remaining fixture is drawn as a home win, draw or away win from its 1X2 probabilities. Each team's probability is the share of simulated seasons where it finishes in that position, and its fair odds are 1 / probability. Teams level on points are ordered at random. A fixed seed means the same parameters always give the same result."),
    controls,
    state.simResult && !stale ? el("p", { class: "muted small" }, `${state.lastSims.toLocaleString()} seasons simulated in ${Math.round(state.simMs)} ms.`) : null,
    body,
    codeLinks("season"));
}

const TABS = [
  ["table", "Predicted table", tableTab],
  ["ratings", "Team ratings", ratingsTab],
  ["fixtures", "Fixtures & prices", fixturesTab],
  ["season", "Season odds", seasonTab],
];

// ── Controls ───────────────────────────────────────────────────────────
// The sliders are built once per league and updated in place, so dragging stays smooth.
let ui = {};

function slider(label, min, max, step, display) {
  const out = el("output");
  const input = el("input", { type: "range", min, max, step });
  const wrap = el("label", { class: "lt-slider" }, el("span", { class: "lt-slider-lbl" }, label, out), input);
  return { wrap, input, out, display };
}

function setSlider(s, value, max) {
  if (max !== undefined) s.input.max = max;
  s.input.value = value;
  s.out.textContent = s.display(value);
}

function buildControls() {
  const { meta } = state.data;
  const d = meta.defaults;
  ui.n = slider("Window (matches)", meta.window_min, meta.window_max, 1, (v) => v);
  ui.g = slider("Goals weight", 0, 1, 0.05, (v) => v.toFixed(2));
  ui.x = slider("xG weight", 0, 1, 0.05, (v) => v.toFixed(2));
  ui.n.input.addEventListener("input", () => update({ n: +ui.n.input.value }));
  ui.g.input.addEventListener("input", () => update({ g: +ui.g.input.value }));
  ui.x.input.addEventListener("input", () => update({ x: +ui.x.input.value }));

  ui.oOut = el("output");
  ui.bar = { g: el("span", { class: "g", title: "Goals" }), x: el("span", { class: "x", title: "xG" }), o: el("span", { class: "o", title: "Odds" }) };
  ui.reset = el("button", { class: "lt-btn", onclick: () => update({ n: d.n_matches, g: d.goals_wgt, x: d.xG_wgt }) }, "Reset to optimised");

  return el("div", { class: "lt-controls" },
    el("div", { class: "lt-controls-grid" },
      ui.n.wrap, ui.g.wrap, ui.x.wrap,
      el("div", { class: "lt-slider auto" },
        el("span", { class: "lt-slider-lbl" }, "Odds weight (auto)", ui.oOut),
        el("div", { class: "lt-weightbar", "aria-hidden": "true" }, ui.bar.g, ui.bar.x, ui.bar.o),
        el("div", { class: "lt-legend muted small" },
          el("span", { class: "k g" }), "Goals ", el("span", { class: "k x" }), "xG ", el("span", { class: "k o" }), "Odds"))),
    el("div", { class: "lt-controls-foot" },
      ui.reset,
      el("span", { class: "muted small" },
        `Optimised: ${d.n_matches} matches, weights ${d.goals_wgt} / ${d.xG_wgt} / ${d.asian_wgt} (goals / xG / odds), ` +
        "found by a grid search for the window and weights that best predicted the market's totals and supremacy in previous seasons.")));
}

function syncControls() {
  const p = state.params, d = state.data.meta.defaults;
  const o = round2(1 - p.g - p.x);
  setSlider(ui.n, p.n);
  setSlider(ui.g, p.g);
  setSlider(ui.x, p.x, round2(1 - p.g));
  ui.oOut.textContent = o.toFixed(2);
  ui.bar.g.style.width = `${p.g * 100}%`;
  ui.bar.x.style.width = `${p.x * 100}%`;
  ui.bar.o.style.width = `${o * 100}%`;
  ui.reset.disabled = p.n === d.n_matches && p.g === d.goals_wgt && p.x === d.xG_wgt;
}

// ── Render ─────────────────────────────────────────────────────────────
let root, controlsHost, panelHost, tabHost;

function update(patch) {
  const p = { ...state.params, ...patch };
  p.g = round2(p.g);
  p.x = round2(Math.min(p.x, 1 - p.g));
  state.params = p;
  writeHash();
  render();
}

function render() {
  syncControls();
  tabHost.querySelectorAll("button").forEach((b) => {
    const on = b.dataset.tab === state.tab;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on);
  });
  const [, , fn] = TABS.find(([k]) => k === state.tab);
  panelHost.replaceChildren(fn(compute()));
}

// Loads a league's snapshot. Parameters from the link apply on first load only;
// switching league starts from that league's optimised defaults.
async function loadLeague(file, fromHash) {
  state.data = await (await fetch(SNAP + file)).json();
  const d = state.data.meta.defaults;
  const h = fromHash ? readHash() : {};
  const n = Math.min(state.data.meta.window_max, Math.max(state.data.meta.window_min, h.n ?? d.n_matches));
  const g = round2(Math.min(1, Math.max(0, h.g ?? d.goals_wgt)));
  const x = round2(Math.min(1 - g, Math.max(0, h.x ?? d.xG_wgt)));
  state.params = { n, g, x };
  if (TABS.some(([k]) => k === h.tab)) state.tab = h.tab;
  state.simResult = null;
  window.ltCheck = () => checkAgainstPython(state.data);
}

function subtitle() {
  const { meta, results, fixtures } = state.data;
  const last = results.reduce((m, r) => (r.Date > m ? r.Date : m), "");
  const parts = [
    meta.season.replace("/20", "/"),
    results.length ? `${results.length} results to ${fmt.date(last)}` : "no results yet",
    `${fixtures.length} fixtures remaining`,
  ];
  if (state.updated) parts.push(`updated ${fmt.date(state.updated)}`);
  return parts.join(" · ");
}

function linesNote() {
  const n = state.data.meta.n_results_no_lines;
  return n ? `${n} recent result${n > 1 ? "s have" : " has"} no betting lines yet, so ${n > 1 ? "they count" : "it counts"} for goals and xG only.` : "";
}

async function init() {
  root = document.getElementById("lt-app");
  if (!root) return;
  try {
    const index = await (await fetch(SNAP + "index.json")).json();
    state.updated = index.updated;
    const wanted = readHash().lg;
    const first = index.leagues.find((l) => l.key === wanted) || index.leagues[0];
    await loadLeague(first.file, true);

    const title = el("div", { class: "lt-title" }, state.data.meta.league);
    const sub = el("div", { class: "muted small" }, subtitle());
    const note = el("div", { class: "muted small" }, linesNote());
    const picker = el("div", { class: "lt-leagues", role: "radiogroup", "aria-label": "League" },
      index.leagues.map((l) => el("button", {
        class: "lt-league-btn", role: "radio", "data-key": l.key,
        onclick: async () => {
          if (l.key === state.data.meta.key) return;
          await loadLeague(l.file, false);
          title.textContent = state.data.meta.league;
          sub.textContent = subtitle();
          note.textContent = linesNote();
          controlsHost.replaceChildren(buildControls());
          syncPicker();
          writeHash();
          render();
        } }, l.league)));
    const syncPicker = () => picker.querySelectorAll("button").forEach((b) => {
      const on = b.dataset.key === state.data.meta.key;
      b.classList.toggle("active", on);
      b.setAttribute("aria-checked", on);
    });
    syncPicker();
    const head = el("div", { class: "lt-head" }, el("div", {}, title, sub, note));

    controlsHost = el("div", {}, buildControls());
    tabHost = el("div", { class: "lt-tabs", role: "tablist" },
      TABS.map(([k, label]) => el("button", { "data-tab": k, role: "tab", onclick: () => { state.tab = k; writeHash(); render(); } }, label)));
    panelHost = el("div", { class: "lt-panel", role: "tabpanel" });
    root.replaceChildren(picker, head, controlsHost, tabHost, panelHost);
    render();
  } catch (err) {
    root.replaceChildren(el("p", { class: "lt-empty" }, "The model data couldn't be loaded. " + err.message));
    console.error(err);
  }
}

init();

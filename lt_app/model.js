// Browser implementation of the Long-term model. Mirrors code/lt/*.py so the page can
// recompute everything live as the parameters change. No DOM code in this file.

const MAX_GOALS = 11;

// Team ratings: mean of each metric over the team's most recent n matches, then weighted
// (code/lt/ratings.py: current_ratings, weighted_ratings). Promoted teams' second-division
// matches have Asian goals only, so goals and xG average over the matches that have them,
// and fall back on the Asian average if none do.
export function teamRatings(teams, n, goalsWgt, xgWgt, asianWgt) {
  const mean = (arr) => {
    const v = arr.slice(0, n).filter((x) => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  };
  const rows = teams.map((t) => {
    const af = mean(t.Asian_for), aa = mean(t.Asian_conc);
    const m = {
      gf: mean(t.GoalsScored) ?? af, ga: mean(t.GoalsConceded) ?? aa,
      xgf: mean(t.xG_for) ?? af, xga: mean(t.xG_conc) ?? aa,
      af, aa,
    };
    return {
      team: t.team, ...m,
      below: (t.below || []).slice(0, n).reduce((a, b) => a + b, 0),
      AS: m.gf * goalsWgt + m.xgf * xgWgt + m.af * asianWgt,
      DS: m.ga * goalsWgt + m.xga * xgWgt + m.aa * asianWgt,
    };
  });
  const dsAvg = rows.reduce((a, r) => a + r.DS, 0) / rows.length;
  return { rows, dsAvg, byTeam: Object.fromEntries(rows.map((r) => [r.team, r])) };
}

// Expected goals for a fixture with the home advantage shift (code/lt/ratings.py: predict_goals)
export function predictGoals(home, away, dsAvg, HA) {
  let h = (home.AS * away.DS) / dsAvg;
  let a = (away.AS * home.DS) / dsAvg;
  const totalHA = (h + a) * HA;
  h += totalHA / 2;
  a -= totalHA / 2;
  return { home: h, away: a, total: h + a, sup: h - a };
}

function poissonPmfs(lam) {
  const out = new Float64Array(MAX_GOALS);
  out[0] = Math.exp(-lam);
  for (let k = 1; k < MAX_GOALS; k++) out[k] = (out[k - 1] * lam) / k;
  return out;
}

// Scoreline matrix with Dixon-Coles tau on 0-0, 0-1, 1-0, 1-1 (code/lt/dixon_coles.py: score_matrix)
export function scoreMatrix(lh, la, rho) {
  const ph = poissonPmfs(lh), pa = poissonPmfs(la);
  const m = [];
  let sum = 0;
  for (let i = 0; i < MAX_GOALS; i++) {
    m.push(new Float64Array(MAX_GOALS));
    for (let j = 0; j < MAX_GOALS; j++) {
      let p = ph[i] * pa[j];
      if (i === 0 && j === 0) p *= 1 - lh * la * rho;
      else if (i === 0 && j === 1) p *= 1 + lh * rho;
      else if (i === 1 && j === 0) p *= 1 + la * rho;
      else if (i === 1 && j === 1) p *= 1 - rho;
      m[i][j] = p;
      sum += p;
    }
  }
  for (const row of m) for (let j = 0; j < MAX_GOALS; j++) row[j] /= sum;
  return m;
}

// Home / draw / away probabilities (code/lt/dixon_coles.py: match_probabilities)
export function oneXTwo(lh, la, rho) {
  const m = scoreMatrix(lh, la, rho);
  let home = 0, draw = 0, away = 0;
  for (let i = 0; i < MAX_GOALS; i++)
    for (let j = 0; j < MAX_GOALS; j++) {
      if (i > j) home += m[i][j];
      else if (i === j) draw += m[i][j];
      else away += m[i][j];
    }
  return { home, draw, away };
}

// Prices every remaining fixture from the ratings
export function priceFixtures(fixtures, ratings, HA, rho) {
  return fixtures.map((f) => {
    const g = predictGoals(ratings.byTeam[f.Home], ratings.byTeam[f.Away], ratings.dsAvg, HA);
    return { ...f, ...g, p: oneXTwo(g.home, g.away, rho) };
  });
}

const sortTable = (rows) => rows.sort((a, b) => b.Pts - a.Pts || b.GD - a.GD || b.GF - a.GF);

// Current table from results (code/lt/tables.py: current_table)
export function currentTable(teamNames, results) {
  const t = Object.fromEntries(teamNames.map((n) => [n, { team: n, MP: 0, W: 0, D: 0, L: 0, GF: 0, GA: 0, GD: 0, Pts: 0 }]));
  for (const r of results) {
    const h = t[r.Home], a = t[r.Away];
    h.MP++; a.MP++;
    if (r.goals_h > r.goals_a) { h.W++; h.Pts += 3; a.L++; }
    else if (r.goals_h < r.goals_a) { a.W++; a.Pts += 3; h.L++; }
    else { h.D++; a.D++; h.Pts++; a.Pts++; }
    h.GF += r.goals_h; h.GA += r.goals_a; a.GF += r.goals_a; a.GA += r.goals_h;
  }
  const rows = Object.values(t);
  rows.forEach((r) => (r.GD = r.GF - r.GA));
  return sortTable(rows);
}

// Expected final table: current table plus each fixture's expected points and goals
// (code/lt/tables.py: expected_table)
export function expectedTable(table, priced) {
  const t = Object.fromEntries(table.map((r) => [r.team, { ...r, cur: r }]));
  for (const f of priced) {
    const h = t[f.Home], a = t[f.Away];
    h.MP++; a.MP++;
    h.W += f.p.home; h.D += f.p.draw; h.L += f.p.away;
    a.W += f.p.away; a.D += f.p.draw; a.L += f.p.home;
    h.Pts += 3 * f.p.home + f.p.draw;
    a.Pts += 3 * f.p.away + f.p.draw;
    h.GF += f.home; h.GA += f.away; a.GF += f.away; a.GA += f.home;
  }
  const rows = Object.values(t);
  rows.forEach((r) => (r.GD = r.GF - r.GA));
  return sortTable(rows);
}

// Small seeded PRNG so a run can be reproduced
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Monte Carlo of the remaining fixtures (code/lt/simulate.py: simulate_season).
// Teams level on points are ordered at random.
export function simulateSeason(table, priced, nSims, seed = 1) {
  const rand = mulberry32(seed);
  const names = table.map((r) => r.team);
  const idx = Object.fromEntries(names.map((n, i) => [n, i]));
  const nT = names.length;
  const base = Float64Array.from(table, (r) => r.Pts);
  const fh = Int32Array.from(priced, (f) => idx[f.Home]);
  const fa = Int32Array.from(priced, (f) => idx[f.Away]);
  const ph = Float64Array.from(priced, (f) => f.p.home);
  const pd = Float64Array.from(priced, (f) => f.p.home + f.p.draw);

  const first = new Float64Array(nT), top4 = new Float64Array(nT), top6 = new Float64Array(nT), bottom3 = new Float64Array(nT);
  const posCount = Array.from({ length: nT }, () => new Float64Array(nT));
  const pts = new Float64Array(nT);
  const order = Array.from({ length: nT }, (_, i) => i);

  for (let s = 0; s < nSims; s++) {
    pts.set(base);
    for (let k = 0; k < fh.length; k++) {
      const u = rand();
      if (u < ph[k]) pts[fh[k]] += 3;
      else if (u < pd[k]) { pts[fh[k]] += 1; pts[fa[k]] += 1; }
      else pts[fa[k]] += 3;
    }
    for (let i = 0; i < nT; i++) pts[i] += rand() * 0.01; // random tie-break
    order.sort((x, y) => pts[y] - pts[x]);
    for (let p = 0; p < nT; p++) {
      const i = order[p];
      posCount[i][p]++;
      if (p === 0) first[i]++;
      if (p < 4) top4[i]++;
      if (p < 6) top6[i]++;
      if (p >= nT - 3) bottom3[i]++;
    }
  }
  return names.map((team, i) => ({
    team,
    first: first[i] / nSims, top4: top4[i] / nSims, top6: top6[i] / nSims, bottom3: bottom3[i] / nSims,
    positions: Array.from(posCount[i], (c) => c / nSims),
  }));
}

// Compares the browser outputs at the default parameters with the Python reference in the JSON
export function checkAgainstPython(data) {
  const d = data.meta.defaults;
  const ratings = teamRatings(data.teams, d.n_matches, d.goals_wgt, d.xG_wgt, d.asian_wgt);
  const priced = priceFixtures(data.fixtures, ratings, data.meta.HA, data.meta.rho);
  const exp = expectedTable(currentTable(data.teams.map((t) => t.team), data.results), priced);
  let maxRating = 0, maxFixture = 0, maxPts = 0;
  for (const [team, [as, ds]] of Object.entries(data.check.ratings)) {
    const r = ratings.byTeam[team];
    maxRating = Math.max(maxRating, Math.abs(r.AS - as), Math.abs(r.DS - ds));
  }
  data.check.fixtures.forEach((c, k) => {
    const f = priced[k];
    [f.home, f.away, f.p.home, f.p.draw, f.p.away].forEach((v, j) => (maxFixture = Math.max(maxFixture, Math.abs(v - c[j]))));
  });
  for (const r of exp) maxPts = Math.max(maxPts, Math.abs(r.Pts - data.check.expected_pts[r.team]));
  const longest = teamRatings(data.teams, data.meta.window_max, d.goals_wgt, d.xG_wgt, d.asian_wgt);
  let maxRatingWindowMax = 0;
  for (const [team, [as, ds]] of Object.entries(data.check.ratings_window_max || {})) {
    const r = longest.byTeam[team];
    maxRatingWindowMax = Math.max(maxRatingWindowMax, Math.abs(r.AS - as), Math.abs(r.DS - ds));
  }
  return { maxRating, maxFixture, maxPts, maxRatingWindowMax };
}

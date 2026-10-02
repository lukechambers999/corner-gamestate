"""
League tables: the current table from results, and the expected final table.

The expected table adds each remaining fixture's outcome probabilities to the
current one: a home win probability of 0.5 is worth 1.5 expected points, and
expected goals are added to goals for and against.
From LT_model.ipynb (Current season table, End of season table prediction).
"""

import pandas as pd

TABLE_COLS = ['MP', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']


# Current table from played results
def current_table(teams, results):
    table = pd.DataFrame(0.0, index=pd.Index(teams, name='Team'), columns=TABLE_COLS)

    for row in results.itertuples():
        home, away, goals_h, goals_a = row.Home, row.Away, row.goals_h, row.goals_a
        table.loc[[home, away], 'MP'] += 1

        if goals_h > goals_a:
            table.loc[home, ['W', 'Pts']] += [1, 3]
            table.loc[away, 'L'] += 1
        elif goals_h < goals_a:
            table.loc[away, ['W', 'Pts']] += [1, 3]
            table.loc[home, 'L'] += 1
        else:
            table.loc[[home, away], ['D', 'Pts']] += 1

        table.loc[home, ['GF', 'GA', 'GD']] += [goals_h, goals_a, goals_h - goals_a]
        table.loc[away, ['GF', 'GA', 'GD']] += [goals_a, goals_h, goals_a - goals_h]

    return table.sort_values(['Pts', 'GD', 'GF'], ascending=False)


# Adds expected points, results and goals from each remaining fixture to the current table
def expected_table(table, fixtures):
    table = table.copy()

    for row in fixtures.itertuples():
        home, away = row.Home, row.Away
        table.loc[[home, away], 'MP'] += 1

        table.loc[home, ['W', 'D', 'L']] += [row.home_pc, row.draw_pc, row.away_pc]
        table.loc[away, ['W', 'D', 'L']] += [row.away_pc, row.draw_pc, row.home_pc]
        table.loc[home, 'Pts'] += 3 * row.home_pc + row.draw_pc
        table.loc[away, 'Pts'] += 3 * row.away_pc + row.draw_pc

        exp_hg, exp_ag = row.Home_Pred_Goals, row.Away_Pred_Goals
        table.loc[home, ['GF', 'GA', 'GD']] += [exp_hg, exp_ag, exp_hg - exp_ag]
        table.loc[away, ['GF', 'GA', 'GD']] += [exp_ag, exp_hg, exp_ag - exp_hg]

    return table.sort_values(['Pts', 'GD', 'GF'], ascending=False)

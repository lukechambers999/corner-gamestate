"""
Understat match data: goals and xG for every played match, plus the remaining fixtures.

Each league-season is scraped with understatapi, flattened from Understat's
nested JSON into one row per match, and team names are converted to the
totalcorner.com spellings so the two sources can be joined.
From LT_model.ipynb (Understat fixture scraper, Data clean up, Dataset merging).
"""

import pandas as pd
import understatapi

# Understat league name -> totalcorner league file (top flight and the division below)
LEAGUES = {
    'EPL': {'name': 'Premier League', 'league_tc': 'EnglandPremierLeague', 'league_tc_below': 'EnglandChampionship', 'n_teams_div': 20},
    'La_Liga': {'name': 'La Liga', 'league_tc': 'SpainLaLiga', 'league_tc_below': 'SpainSegunda', 'n_teams_div': 20},
    'Serie_A': {'name': 'Serie A', 'league_tc': 'ItalySerieA', 'league_tc_below': 'ItalySerieB', 'n_teams_div': 20},
    'Bundesliga': {'name': 'Bundesliga', 'league_tc': 'GermanyBundesligaI', 'league_tc_below': 'GermanyBundesligaII', 'n_teams_div': 18},
    'Ligue_1': {'name': 'Ligue 1', 'league_tc': 'FranceLigue1', 'league_tc_below': 'FranceLigue2', 'n_teams_div': 18},
}

FIRST_SEASON = 2015

# Understat name -> totalcorner name, where they differ
NAME_CONV = {
    'Parma Calcio 1913': 'Parma', 'Inter': 'Inter Milan', 'SPAL 2013': 'Spal',
    'Wolverhampton Wanderers': 'Wolverhampton', 'West Bromwich Albion': 'West Brom', 'Sheffield United': 'Sheff Utd',
    'Manchester City': 'Man City', 'Manchester United': 'Man Utd', 'Newcastle United': 'Newcastle', 'Nottingham Forest': 'Nottm Forest',
    'Alaves': 'CD Alaves', 'Real Valladolid': 'Valladolid', 'SD Huesca': 'Huesca', 'Deportivo La Coruna': 'Deportivo A Coruna',
    'Borussia M.Gladbach': 'Borussia M\'gladbach', 'FC Cologne': 'Cologne', 'FC Heidenheim': 'Heidenheim', 'Freiburg': 'SC Freiburg',
    'Greuther Fuerth': 'Greuther Furth',
    'Hamburger SV': 'Hamburg', 'Hoffenheim': 'TSG Hoffenheim', 'Ingolstadt': 'FC Ingolstadt', 'Mainz 05': 'Mainz',
    'Nuernberg': 'Nurnberg', 'RasenBallsport Leipzig': 'RB Leipzig', 'Schalke 04': 'Schalke', 'St. Pauli': 'St Pauli',
    'Fortuna Duesseldorf': 'Fortuna Dusseldorf', 'Ajaccio': 'AC Ajaccio', 'GFC Ajaccio': 'Ajaccio GFCA',
    'Paris Saint Germain': 'PSG', 'Saint-Etienne': 'St Etienne',
}


# Scrapes every season of a league from FIRST_SEASON up to the current one
def scrape_league(league, last_season):
    client = understatapi.UnderstatClient()
    raw_data = []
    for season in range(FIRST_SEASON, last_season + 1):
        df = pd.DataFrame(client.league(league=league).get_match_data(season=str(season)))
        df['season'] = f"{season}/{season + 1}"
        raw_data.append(df)
    return pd.concat(raw_data, ignore_index=True)


# Flattens the nested columns, converts types and harmonises team names with totalcorner
def clean(fixtures_data):
    df = fixtures_data.copy()
    for col in ['h', 'a', 'goals', 'xG']:
        flat = pd.json_normalize(df[col])
        flat.columns = [f"{col}_{key}" for key in flat.columns]
        df = pd.concat([df.drop(columns=col), flat], axis=1)

    df = df.rename(columns={'h_title': 'Home', 'a_title': 'Away', 'datetime': 'Date'})
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df['isResult'] = df['isResult'].astype(bool)
    for col in ['goals_h', 'goals_a', 'xG_h', 'xG_a']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Home'] = df['Home'].replace(NAME_CONV)
    df['Away'] = df['Away'].replace(NAME_CONV)
    return df[['Date', 'season', 'isResult', 'Home', 'Away', 'goals_h', 'goals_a', 'xG_h', 'xG_a']]

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3  # SQLite

# Create connection
database_connection = sqlite3.connect("database.sqlite")
data = pd.read_sql("""SELECT Match.id, Match.home_team_api_id, Match.away_team_api_id,
                                        Country.name AS country_name, 
                                        League.name AS league_name, 
                                        season, 
                                        stage, 
                                        date,
                                        HT.team_long_name AS  home_team,
                                        AT.team_long_name AS away_team,
                                        home_team_goal, 
                                        away_team_goal                                        
                                FROM Match
                                JOIN Country on Country.id = Match.country_id
                                JOIN League on League.id = Match.league_id
                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id                          
                                ORDER by date
                                ;""", database_connection)
data1=data[["home_team","away_team","season","home_team_goal","away_team_goal"]]

dataset=pd.DataFrame({"HomeTeamAPI":data['home_team_api_id'] ,"HomeTeam":data1.home_team+data1.season,'AwayTeamAPI':data['away_team_api_id'],
                   "AwayTeam":data1.away_team+data1.season,"FTHG":data1.home_team_goal,"FTAG":data1.away_team_goal})
print(dataset.head())

tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table'; """, database_connection)

# Player Attributes Table with only the unique player_api_id feature and the overall_rating feature
player_attributes = pd.read_sql_query("""SELECT DISTINCT player_api_id, overall_rating 
                                         FROM Player_Attributes
                                         GROUP BY player_api_id                         
                                         """, database_connection)
# set the index to be the player_api_id field
player_attributes.set_index('player_api_id')

# Matches Table
match = pd.read_sql("""SELECT *
                       FROM Match
                       WHERE home_player_1 IS NOT NULL AND 
                       home_player_2 IS NOT NULL AND 
                       home_player_3 IS NOT NULL AND 
                       home_player_4 IS NOT NULL AND
                       home_player_5 IS NOT NULL AND
                       home_player_6 IS NOT NULL AND
                       home_player_7 IS NOT NULL AND
                       home_player_8 IS NOT NULL AND
                       home_player_9 IS NOT NULL AND
                       home_player_10 IS NOT NULL AND
                       home_player_11 IS NOT NULL AND
                       away_player_1 IS NOT NULL  AND
                       away_player_2 IS NOT NULL AND
                       away_player_3 IS NOT NULL AND
                       away_player_4 IS NOT NULL AND
                       away_player_5 IS NOT NULL AND
                       away_player_6 IS NOT NULL AND
                       away_player_7 IS NOT NULL AND
                       away_player_8 IS NOT NULL AND
                       away_player_9 IS NOT NULL AND
                       away_player_10 IS NOT NULL AND
                       away_player_11 IS NOT NULL
                       """, database_connection)

# Team Attributes Table
team_attribute = pd.read_sql("""SELECT *
                                 FROM Team_Attributes
                                 """, database_connection)



home_team_ids = dataset['HomeTeamAPI'].dropna().tolist()
away_team_ids = dataset['AwayTeamAPI'].dropna().tolist()




# home_players_features = [f"home_player_{feat}" for feat in range(1,11)]
teams_players = {}

for home_team in home_team_ids:
    teams_players[home_team] = []
    df = match.loc[match['home_team_api_id'] == home_team] # Get the dataframe of each home team
    home_team_lineup = df.loc[:, 'home_player_1':'home_player_11']  # Get the lineup of players id of the home team
    for column in home_team_lineup.columns:
        #  Remove duplicates for each player_X feature
        players = home_team_lineup.drop_duplicates(column)[column].tolist()
        teams_players[home_team] += players

    # Remove duplicates where player 1 was player 2 for example
    teams_players[home_team] = set(teams_players[home_team])

# for away_team in home_team_ids:
#     if away_team not in teams_players: # If the away_team never played as an home team
#         teams_players[away_team] = []
#
#     df = match.loc[match['away_team_api_id'] == away_team] # Get the dataframe of each away team
#     away_team_lineup = df.loc[:, 'away_player_1':'away_player_11']  # Get the lineup of players id of the away team
#     for column in away_team_lineup.columns:
#         #  Remove duplicates for each player_X feature
#         players = away_team_lineup.dropna().drop_duplicates(column)[column].tolist()
#         teams_players[away_team] += players
#
#     # Remove duplicates where player 1 was player 2 for example
#     teams_players[away_team] = set(teams_players[away_team])

team_average_players_ratings = {}

for team, players in teams_players.items():
    players_ratings = player_attributes.iloc[players]
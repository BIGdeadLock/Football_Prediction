import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3  # SQLite
import xml.etree.ElementTree as ET


class Preprocesses(object):

    def __init__(self, database_path):

        self._database_connection = sqlite3.connect(database_path)
        self._team_attributes_data = None
        self._player_attributes_data = None
        self._match_data = None
        self._dataset: pd.DataFrame = None
        self.__load_data()

    def preprocess(self) -> pd.DataFrame:
        """
        The main method that start the preprocess flow.

        :return: DataFrame object containing the preprocessed dataset
        """
        # self.__add_team_rankings()
        # self.__add_team_stats()
        # self.__add_classification()
        self.__add_bets_ods_features()
        self._database_connection.close()

        return self._dataset

    def __load_data(self):
        """
        The method will be responsible for loading the data from the database.
        """
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
                                       LIMIT 5000

                                        ;""", self._database_connection)
        data1 = data[["home_team", "away_team", "season", "home_team_goal", "away_team_goal"]]

        self._dataset = pd.DataFrame(
            {"HomeTeamAPI": data['home_team_api_id'], "HomeTeam": data1.home_team + data1.season,
             'AwayTeamAPI': data['away_team_api_id'],
             "AwayTeam": data1.away_team + data1.season, "HomeTeamGaols": data1.home_team_goal,
             "AwayTeamGaols": data1.away_team_goal})

        # Player Attributes Table with only the unique player_api_id feature and the overall_rating feature
        self._player_attributes_data = pd.read_sql_query("""SELECT DISTINCT player_api_id, overall_rating 
                                                 FROM Player_Attributes
                                                 GROUP BY player_api_id                         
                                                 """, self._database_connection)
        # set the index to be the player_api_id field
        self._player_attributes_data.set_index('player_api_id', inplace=True, drop=True)

        # Matches Table
        self._match_data = pd.read_sql("""SELECT *
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
                               LIMIT 5000
                               """, self._database_connection)  # TODO: Remove the limit

        # Team Attributes Table
        self._team_attributes_data = pd.read_sql("""SELECT team_api_id, buildUpPlaySpeed, chanceCreationShooting, defencePressure  
                                         FROM Team_Attributes
                                         GROUP BY team_api_id
                                         """, self._database_connection)

    def __add_team_rankings(self):
        """
        The method will be responsible for creating the Team Rankings features in the dataset.
        The team rankings features include the HomeTeamRanking and AwayTeamRanking which are based on the
        overall_rating of the players in each team's lineup.

        :return:
        """
        home_team_ids = self._dataset['HomeTeamAPI'].drop_duplicates().dropna().tolist()
        away_team_ids = self._dataset['AwayTeamAPI'].drop_duplicates().dropna().tolist()

        home_players_features = [f"home_player_{feat}" for feat in range(1, 11)]
        away_players_features = [f"away_player_{feat}" for feat in range(1, 11)]
        teams_players = {}

        for home_team in home_team_ids:
            teams_players[home_team] = []
            df = self._match_data.loc[
                self._match_data['home_team_api_id'] == home_team]  # Get the dataframe of each home team
            home_team_lineup = df.loc[:,
                               'home_player_1':'home_player_11']  # Get the lineup of players id of the home team
            home_team_lineup.drop_duplicates(subset=home_players_features, keep=False, inplace=True)
            for column in home_team_lineup.columns:
                #  Remove duplicates for each player_X feature
                players = home_team_lineup.drop_duplicates(column)[column].tolist()
                teams_players[home_team] += players

            # Remove duplicates where player 1 was player 2 for example
            teams_players[home_team] = set(teams_players[home_team])

        for away_team in away_team_ids:
            teams_players[away_team] = []
            df = self._match_data.loc[
                self._match_data['away_team_api_id'] == away_team]  # Get the dataframe of each home team
            away_team_lineup = df.loc[:,
                               'away_player_1':'away_player_11']  # Get the lineup of players id of the home team
            away_team_lineup.drop_duplicates(subset=away_players_features, keep=False, inplace=True)
            for column in away_team_lineup.columns:
                #  Remove duplicates for each player_X feature
                players = away_team_lineup.drop_duplicates(column)[column].tolist()
                teams_players[away_team] += players

        team_average_players_ratings = {}

        for team, players in teams_players.items():
            if players:
                players_ratings = self._player_attributes_data.loc[list(players)]  # Get the team players ratings
                team_average_players_ratings[team] = players_ratings.mean().at['overall_rating']

        home_team_average_players_ratings = pd.DataFrame({"HomeTeamAPI": list(team_average_players_ratings.keys()),
                                                          "HomeTeamRatings": list(
                                                              team_average_players_ratings.values())})
        away_team_average_players_ratings = pd.DataFrame({"AwayTeamAPI": list(team_average_players_ratings.keys()),
                                                          "AwayTeamRatings": list(
                                                              team_average_players_ratings.values())})

        self._dataset = pd.merge(self._dataset, home_team_average_players_ratings, how="inner", on="HomeTeamAPI")
        self._dataset = pd.merge(self._dataset, away_team_average_players_ratings, how="inner", on="AwayTeamAPI")

    def __add_team_stats(self):
        """
        The method will be responsible for creating the Team stats features in the dataset.
        The team stats features include the buildUpPlaySpeed, chanceCreationShooting and defencePressure of each team
        in each match.

        :return:
        """
        self._dataset = pd.merge(self._dataset, self._team_attributes_data, how="inner", left_on="HomeTeamAPI",
                                 right_on="team_api_id"). \
            rename(
            columns={'buildUpPlaySpeed': 'HomeTeamPlaySpeed', "chanceCreationShooting": "HomeTeamCreatonShooting",
                     "defencePressure": "HomeTeamDefencePressure"})
        self._dataset = pd.merge(self._dataset, self._team_attributes_data, how="inner", left_on="AwayTeamAPI",
                                 right_on="team_api_id"). \
            rename(
            columns={'buildUpPlaySpeed': 'AwayTeamPlaySpeed', "chanceCreationShooting": "AwayTeamCreatonShooting",
                     "defencePressure": "AwayTeamDefencePressure"})
        self._dataset = self._dataset.drop(columns={'team_api_id_x', 'team_api_id_y'})

    def __add_classification(self):
        """
        The method will be responsible for creating the labels for each match.
        0 - Draw
        1 - Home team wins
        2 - Home team loose (Away team win)
        :return:
        """
        win = []
        for l in range(0, len(self._dataset)):
            if self._dataset.HomeTeamGaols[l] > self._dataset.AwayTeamGaols[l]:
                k1 = 1
                win.append(k1)
            elif self._dataset.HomeTeamGaols[l] == self._dataset.AwayTeamGaols[l]:
                k1 = 0
                win.append(k1)
            else:
                k1 = 2
                win.append(k1)

        self._dataset['win'] = win

    def __parse_cards_xml(self, xml_document, home_team, away_team, card_type='y'):
        stat_home_team = 0
        stat_away_team = 0
        tree = ET.fromstring(xml_document)

        if tree.tag == 'card':
            for child in tree.iter('value'):
                # Some xml docs have no card_type element in the tree. comment section seems to have that information
                try:
                    if child.find('comment').text == card_type:
                        if int(child.find('team').text) == home_team:
                            stat_home_team += 1
                    else:
                        stat_away_team += 1

                except AttributeError:
                    # Some values in the xml doc don't have team values, so there isn't much we can do at this stage
                    pass

                return stat_home_team, stat_away_team

    def __add_bets_ods_features(self):
        match_betting_ods = {"Label": [], "HomeTeamsOdds": [], "AwayTeamOdds": []}

        for label, row in self._dataset.iterrows():
            match_betting_ods["Label"] += [label]


            away_team ,home_team = row.at['HomeTeamAPI'], row.at['AwayTeamAPI']
            #  Get all the matches of the away_team and the home_team
            match = self._match_data.loc[(self._match_data['home_team_api_id'] == home_team) & (self._match_data['away_team_api_id'] == away_team)]

            if match.shape[0] == 0:
                self.__remove_row(label)
                continue

            betting_ods = match.loc[:, "B365H": "BSA"]

            # TODO: think about something other than a flag

            for bet, column in zip(['H', 'A'], ['HomeTeamsOdds', "AwayTeamOdds"]): # TODO: Think if Draw is needed
                betting_odds = betting_ods.loc[:, betting_ods.columns.str.endswith(bet)]
                betting_odd = betting_odds.fillna(0).values.mean()
                if not betting_odd:  # If the observation does have all nulls delete it from the dataset
                    self.__remove_row(label)
                else:
                    #  For each match calculate the mean of all betting ods and that will be the match bet odd.
                    match_betting_ods[column] += [betting_odd]

                    if bet == 'H':
                        match_betting_ods["Label"] += [label] # TODO: Need to add only once

        bets_df = pd.DataFrame(match_betting_ods) #TODO: Fix unqeual columns length
        self._dataset = pd.merge(self._dataset, bets_df, left_index=True, right_index=True)

    def __remove_row(self, row_index):
        self._dataset = self._dataset[self._dataset.index != row_index]


p = Preprocesses("database.sqlite")
data = p.preprocess()
x = p._match_data
#
# # Create connection

database_connection = sqlite3.connect("database.sqlite")
# data = pd.read_sql("""SELECT Match.id, Match.home_team_api_id, Match.away_team_api_id,
#                                         Country.name AS country_name,
#                                         League.name AS league_name,
#                                         season,
#                                         stage,
#                                         date,
#                                         HT.team_long_name AS  home_team,
#                                         AT.team_long_name AS away_team,
#                                         home_team_goal,
#                                         away_team_goal
#                                 FROM Match
#                                 JOIN Country on Country.id = Match.country_id
#                                 JOIN League on League.id = Match.league_id
#                                 LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id
#                                 LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id
#                                 ORDER by date
#                                 ;""", database_connection)
# data1=data[["home_team","away_team","season","home_team_goal","away_team_goal"]]
#
# dataset=pd.DataFrame({"HomeTeamAPI":data['home_team_api_id'] ,"HomeTeam":data1.home_team+data1.season,'AwayTeamAPI':data['away_team_api_id'],
#                    "AwayTeam":data1.away_team+data1.season,"HomeTeamGaols":data1.home_team_goal,"AwayTeamGaols":data1.away_team_goal})
#
# tables = pd.read_sql("""SELECT *
#                         FROM sqlite_master
#                         WHERE type='table'; """, database_connection)
#
# # Player Attributes Table with only the unique player_api_id feature and the overall_rating feature
# player_attributes = pd.read_sql_query("""SELECT DISTINCT player_api_id, overall_rating
#                                          FROM Player_Attributes
#                                          GROUP BY player_api_id
#                                          """, database_connection)
# # set the index to be the player_api_id field
# player_attributes.set_index('player_api_id', inplace=True, drop=True)
#
# # Matches Table
# match = pd.read_sql("""SELECT *
#                        FROM Match
#                        WHERE home_player_1 IS NOT NULL AND
#                        home_player_2 IS NOT NULL AND
#                        home_player_3 IS NOT NULL AND
#                        home_player_4 IS NOT NULL AND
#                        home_player_5 IS NOT NULL AND
#                        home_player_6 IS NOT NULL AND
#                        home_player_7 IS NOT NULL AND
#                        home_player_8 IS NOT NULL AND
#                        home_player_9 IS NOT NULL AND
#                        home_player_10 IS NOT NULL AND
#                        home_player_11 IS NOT NULL AND
#                        away_player_1 IS NOT NULL  AND
#                        away_player_2 IS NOT NULL AND
#                        away_player_3 IS NOT NULL AND
#                        away_player_4 IS NOT NULL AND
#                        away_player_5 IS NOT NULL AND
#                        away_player_6 IS NOT NULL AND
#                        away_player_7 IS NOT NULL AND
#                        away_player_8 IS NOT NULL AND
#                        away_player_9 IS NOT NULL AND
#                        away_player_10 IS NOT NULL AND
#                        away_player_11 IS NOT NULL
#                        LIMIT 5000
#                        """, database_connection) # TODO: Remove the limit
#
# # Team Attributes Table
# team_attribute = pd.read_sql("""SELECT team_api_id, buildUpPlaySpeed, chanceCreationShooting, defencePressure
#                                  FROM Team_Attributes
#                                  GROUP BY team_api_id
#                                  """, database_connection)
#
# del data1
# del data
# database_connection.close()
#
#
# home_team_ids = dataset['HomeTeamAPI'].drop_duplicates().dropna().tolist()
# away_team_ids = dataset['AwayTeamAPI'].drop_duplicates().dropna().tolist()
#
#
#
#
# home_players_features = [f"home_player_{feat}" for feat in range(1,11)]
# away_players_features = [f"away_player_{feat}" for feat in range(1,11)]
# teams_players = {}
#
# for home_team in home_team_ids:
#     teams_players[home_team] = []
#     df = match.loc[match['home_team_api_id'] == home_team] # Get the dataframe of each home team
#     home_team_lineup = df.loc[:, 'home_player_1':'home_player_11']  # Get the lineup of players id of the home team
#     home_team_lineup.drop_duplicates(subset=home_players_features, keep=False, inplace=True)
#     for column in home_team_lineup.columns:
#         #  Remove duplicates for each player_X feature
#         players = home_team_lineup.drop_duplicates(column)[column].tolist()
#         teams_players[home_team] += players
#
#     # Remove duplicates where player 1 was player 2 for example
#     teams_players[home_team] = set(teams_players[home_team])
#
# for away_team in home_team_ids:
#     teams_players[away_team] = []
#     df = match.loc[match['away_team_api_id'] == away_team]  # Get the dataframe of each home team
#     away_team_lineup = df.loc[:, 'away_player_1':'away_player_11']  # Get the lineup of players id of the home team
#     away_team_lineup.drop_duplicates(subset=away_players_features, keep=False, inplace=True)
#     for column in away_team_lineup.columns:
#         #  Remove duplicates for each player_X feature
#         players = away_team_lineup.drop_duplicates(column)[column].tolist()
#         teams_players[away_team] += players
#
#
# team_average_players_ratings = {}
#
# for team, players in teams_players.items():
#     if players:
#         players_ratings = player_attributes.loc[list(players)]  # Get the team players ratings
#         team_average_players_ratings[team] = players_ratings.mean().at['overall_rating']
#
# home_team_average_players_ratings = pd.DataFrame({"HomeTeamAPI":list(team_average_players_ratings.keys()), "HomeTeamRatings":list(team_average_players_ratings.values())})
# away_team_average_players_ratings = pd.DataFrame({"AwayTeamAPI":list(team_average_players_ratings.keys()), "AwayTeamRatings":list(team_average_players_ratings.values())})
#
# dataset = pd.merge(dataset, home_team_average_players_ratings, how="inner", on="HomeTeamAPI")
# dataset = pd.merge(dataset, away_team_average_players_ratings, how="inner", on="AwayTeamAPI")
# dataset = pd.merge(dataset, team_attribute, how="inner", left_on="HomeTeamAPI",right_on="team_api_id").\
#     rename(columns={'buildUpPlaySpeed': 'HomeTeamPlaySpead', "chanceCreationShooting": "HomeTeamCreatonShooting",
#                     "defencePressure": "HomeTeamDefencePressure"})
# dataset = pd.merge(dataset, team_attribute, how="inner", left_on="AwayTeamAPI",right_on="team_api_id").\
#     rename(columns={'buildUpPlaySpeed': 'AwayTeamPlaySpead', "chanceCreationShooting": "AwayTeamCreatonShooting",
#                     "defencePressure": "AwayTeamDefencePressure"})
# dataset = dataset.drop(columns={'team_api_id_x', 'team_api_id_y'})

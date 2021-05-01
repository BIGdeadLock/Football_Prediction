import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3  # SQLite
import xml.etree.ElementTree as ET


class FootballPreprocessesor(object):
    """
    The object that will wrap all the football data cleaning and manipulation functionalities.
    """
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
        self.__clear_null_from_match()
        self.__shrink_match_data_dimension()
        self.__add_team_rankings()
        self.__add_team_stats()
        self.__add_bets_ods_features()
        # self.__add_classification()
        self._database_connection.close()

        return self._dataset

    def __load_data(self):
        """
        The method will be responsible for loading the data from the database.
        """
        self.__load_match_table()
        self.__load_team_attr_table()
        self.__load_player_attr_table()
        self.__create_init_dataset()

    def __create_init_dataset(self):
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

                                            ;""", self._database_connection)
        data1 = data[["home_team", "away_team", "season", "home_team_goal", "away_team_goal"]]

        self._dataset = pd.DataFrame(
            {"HomeTeamAPI": data['home_team_api_id'], "HomeTeam": data1.home_team + data1.season,
             'AwayTeamAPI': data['away_team_api_id'],
             "AwayTeam": data1.away_team + data1.season, "HomeTeamGaols": data1.home_team_goal,
             "AwayTeamGaols": data1.away_team_goal})

    def __load_player_attr_table(self):
        self._player_attributes_data = pd.read_sql_query("""SELECT DISTINCT player_api_id, overall_rating 
                                                       FROM Player_Attributes
                                                       GROUP BY player_api_id                         
                                                       """, self._database_connection)
        # set the index to be the player_api_id field
        self._player_attributes_data.set_index('player_api_id', inplace=True, drop=True)

    def __load_team_attr_table(self):
        self._team_attributes_data = pd.read_sql("""SELECT team_api_id, buildUpPlaySpeed, chanceCreationShooting, defencePressure  
                                                FROM Team_Attributes
                                                GROUP BY team_api_id
                                                """, self._database_connection)

    def __load_match_table(self):
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
                                       """, self._database_connection)  # TODO: Remove the limit

    def __unique_value_exctraction(self, df: pd.DataFrame, columns: list) -> set:
        """
        The method will be used to extract unique values of each column set in the columns param.
        :param dataframe: DataFrame - the data which the columns belongs to
        :param columns: list of columns to which we need to extract unique values.
        :return: list of unique values
        """
        unique_values = []
        for col in columns:
            values = df.drop_duplicates(col)[col].tolist()
            unique_values += values

        return set(unique_values)

    def __clear_null_from_match(self):
        """
        The method will be responsible for deleting nulls from the match data based on rules.
        :return:
        """

        # Rule 1 - Clear rows which contains nulls in all the home bets odds columns or away team bets odds columns
        cols = self._match_data.loc[:,"B365H":"BSA"]
        self._bets_columns = {"all": [c for c in cols]}
        self._bets_columns['h'] = [c for c in self._bets_columns['all'] if c[-1] == "H"]
        self._bets_columns['a'] = [c for c in self._bets_columns['all'] if c[-1] == "A"]
        self._bets_columns['a'] = [c for c in self._bets_columns['all'] if c[-1] == "D"]
        # Drop a match observation if all the home team bets have null, like wise to the away_team
        self._match_data.dropna(axis=0, subset=self._bets_columns['h'], how="all", inplace=True)
        self._match_data.dropna(axis=0, subset=self._bets_columns['a'], how="all", inplace=True)

    def __shrink_match_data_dimension(self):
        """
        The method will be responsible for deleting unwanted columns (feature) from the match data.
        :return:
        """
        home_player_X_positions = [f"home_player_X{i}" for i in range(1,12)]
        home_player_Y_positions = [f"home_player_Y{i}" for i in range(1,12)]
        away_player_X_positions = [f"away_player_X{i}" for i in range(1,12)]
        away_player_Y_positions = [f"away_player_Y{i}" for i in range(1,12)]
        for col in [home_player_X_positions, home_player_Y_positions, away_player_X_positions, away_player_Y_positions]:
            self._match_data.drop(col, axis=1, inplace=True)

    def __add_team_rankings(self):
        """
        The method will be responsible for creating the Team Rankings features in the dataset.
        The team rankings features include the HomeTeamRanking and AwayTeamRanking which are based on the
        overall_rating of the players in each team's lineup.

        :return:
        """
        home_team_ids = self._dataset['HomeTeamAPI'].drop_duplicates().dropna().tolist()
        away_team_ids = self._dataset['AwayTeamAPI'].drop_duplicates().dropna().tolist()
        teams_players = {}

        for home_team, away_team in zip(home_team_ids, away_team_ids):
            df = self._match_data.loc[
                self._match_data['home_team_api_id'] == home_team]  # Get the dataframe of each home team
            home_team_lineup = df.loc[:,
                               'home_player_1':'home_player_11']  # Get the lineup of players id of the home team

            if home_team_lineup.shape[0] != 0: # If loc result were 0 continue
                teams_players[home_team] = self.__unique_value_exctraction(home_team_lineup,
                                                                           list(home_team_lineup.columns))

            df = self._match_data.loc[
                self._match_data['away_team_api_id'] == away_team]  # Get the dataframe of each away team
            away_team_lineup = df.loc[:,
                               'away_player_1':'away_player_11']  # Get the lineup of players id of the away team

            if away_team_lineup.shape[0] != 0:  # If loc result were 0 continue
                teams_players[away_team] = self.__unique_value_exctraction(away_team_lineup,
                                                                           list(away_team_lineup.columns))


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
        new_df = pd.DataFrame()

        for label, row in self._dataset.iterrows():


            away_team ,home_team = row.at['HomeTeamAPI'], row.at['AwayTeamAPI']
            #  Get all the matches of the away_team and the home_team
            match = self._match_data.loc[(self._match_data['home_team_api_id'] == home_team) & (self._match_data['away_team_api_id'] == away_team)]

            if match.shape[0] == 0:
                self.__remove_row(label)
                continue

            betting_ods = match.loc[:, self._bets_columns['all'][0]: self._bets_columns['all'][-1]]

            for bet, column in zip(['h', 'a'], ['HomeTeamsOdds', "AwayTeamOdds"]): # TODO: Think if Draw is needed

                home_or_away_bets_odds = betting_ods.loc[:, self._bets_columns[bet]]
                #  For each match calculate the mean of all betting ods and that will be the match bet odd.
                betting_odd = home_or_away_bets_odds.fillna(0).values.mean()
                if not betting_odd: # TODO: Show guy
                    print("a")
                    continue
                row[column] = betting_odd

            #  Create a new dataframe with the new Odss feature
            new_df = new_df.append(row)

        # update the new df
        del self._dataset
        self._dataset = new_df


    def __remove_row(self, row_index):
        self._dataset = self._dataset[self._dataset.index != row_index]


p = FootballPreprocessesor("database.sqlite")
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

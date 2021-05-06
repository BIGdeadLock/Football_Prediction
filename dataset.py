"""
The script will wrap the functionalities for loading of the Data from the selected database
"""
import sqlite3
import definition
import pandas as pd

class Dataset(object):

    def __init__(self, path_to_database):
        """
        The Constructor.
        :param path_to_database: path to the database which contains the football Data

        """
        self._database_connection = sqlite3.connect(path_to_database)
        self.__load_data()


    def get_train_dataset(self):
        """
        Getter for the train Data. The Data contain games with various seasons excluding 2015, 2016
        :return: List of 4 dataframe represent the train data
        """

        return self._dataset, self._match_data, self._team_attributes_data, self._player_attributes_data

    def get_test_dataset(self):
        """
        Getter for the train Data. The Data contain games from 2015 and 2016 only.
        :return: List of 4 dataframe represent the test data
        """
        return self._test_set, self._match_testdata, self._team_attributes_testdata, self._player_attributes_testset

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
                                                WHERE  season <> '2015/2016'
                                                ORDER by date
    
                                                ;""", self._database_connection)
        data1 = data[[definition.TOKEN_HOME_TEAM_NAME, definition.TOKEN_AWAY_TEAM_NAME,
                      definition.TOKEN_MATCH_SEASON, definition.TOKEN_MATCH_HOME_TEAM_GOAL,
                      definition.TOKEN_MATCH_AWAY_TEAM_GOAL]]

        self._dataset = pd.DataFrame(
            {definition.TOKEN_MATCH_ID: data[definition.TOKEN_MATCH_ID],
             definition.TOKEN_DS_HOME_TEAM_ID: data[definition.TOKEN_MATCH_HOME_TEAM_ID],
             definition.TOKEN_DS_HOME_TEAM_NAME: data1.home_team + data1.season,
             definition.TOKEN_DS_AWAY_TEAM_ID: data[definition.TOKEN_MATCH_AWAY_TEAM_ID],
             definition.TOKEN_DS_AWAY_TEAM_NAME: data1.away_team + data1.season,
             definition.TOKEN_DS_HOME_TEAM_GOALS: data1.home_team_goal,
             definition.TOKEN_DS_AWAY_TEAM_GOALS: data1.away_team_goal})


        test_data = pd.read_sql("""SELECT Match.id, Match.home_team_api_id, Match.away_team_api_id,
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
                                                    WHERE  season = '2015/2016'
                                                     ORDER by date
                                                     ;""", self._database_connection)
        data1 = test_data[["home_team", "away_team", "season", "home_team_goal", "away_team_goal"]]

        self._test_set = pd.DataFrame(
            {definition.TOKEN_MATCH_ID: test_data[definition.TOKEN_MATCH_ID],
             definition.TOKEN_DS_HOME_TEAM_ID: test_data[definition.TOKEN_MATCH_HOME_TEAM_ID],
             definition.TOKEN_DS_HOME_TEAM_NAME: data1.home_team + data1.season,
             definition.TOKEN_DS_AWAY_TEAM_ID: test_data[definition.TOKEN_MATCH_AWAY_TEAM_ID],
             definition.TOKEN_DS_AWAY_TEAM_NAME: data1.away_team + data1.season,
             definition.TOKEN_DS_HOME_TEAM_GOALS: data1.home_team_goal,
             definition.TOKEN_DS_AWAY_TEAM_GOALS: data1.away_team_goal})

    def __load_player_attr_table(self):
        self._player_attributes_data = pd.read_sql_query("""SELECT DISTINCT player_api_id, avg(overall_rating) as overall_rating 
                                                           FROM Player_Attributes
                                                           WHERE strftime('%Y',date)<>'2015' or strftime('%Y',date)<>'2016'
                                                           GROUP BY player_api_id                         
                                                           """, self._database_connection)
        # set the index to be the player_api_id field
        self._player_attributes_data.set_index(definition.TOKEN_PLAYER_ID, inplace=True, drop=True)

        self._player_attributes_testset = pd.read_sql_query("""SELECT player_api_id,  avg(overall_rating) as overall_rating  
                                                              FROM Player_Attributes
                                                              WHERE strftime('%Y',date)='2015' or strftime('%Y',date)='2016'
                                                              GROUP BY player_api_id
                                                              """, self._database_connection)
        # set the index to be the player_api_id field
        self._player_attributes_testset.set_index(definition.TOKEN_PLAYER_ID, inplace=True, drop=True)

    def __load_team_attr_table(self):
        self._team_attributes_data = pd.read_sql("""SELECT team_api_id, avg(buildUpPlaySpeed) as buildUpPlaySpeed,
                                                    avg(chanceCreationShooting) as chanceCreationShooting, 
                                                    avg(defencePressure) as defencePressure  
                                                    FROM Team_Attributes
                                                    WHERE strftime('%Y',date)<>'2015' or strftime('%Y',date)<>'2016'
                                                    GROUP BY team_api_id
                                                    """, self._database_connection)

        self._team_attributes_testdata = pd.read_sql("""SELECT team_api_id, 
                                              avg(buildUpPlaySpeed) as buildUpPlaySpeed,
                                              avg(chanceCreationShooting) as chanceCreationShooting, 
                                              avg(defencePressure) as defencePressure 
                                                  FROM Team_Attributes
                                                  WHERE strftime('%Y',date)='2015' or strftime('%Y',date)='2016'
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
                                           away_player_11 IS NOT NULL AND
                                           season <> '2015/2016'
                                           """, self._database_connection)

        self._match_testdata = pd.read_sql("""SELECT *
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
                                     away_player_11 IS NOT NULL AND
                                     season = '2015/2016'
                                     """, self._database_connection)






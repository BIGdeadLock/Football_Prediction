import pandas as pd
import definition
import sqlite3  # SQLite
import xml.etree.ElementTree as ET
from copy import deepcopy


class TestData(object):

    def __init__(self, database_path):
        self._database_connection = sqlite3.connect(database_path)
        self._team_attributes_testdata = None
        self._player_attributes_testset = None
        self._match_testdata = None
        self._test_set: pd.DataFrame = None
        self.__load_data()

    def __load_data(self):
        """
        The method will be responsible for loading the data from the database.
        """
        self.__load_match_table()
        self.__load_team_attr_table()
        self.__load_player_attr_table()
        self.__create_init_dataset()

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
        cols = self._match_testdata.loc[:, "B365H":"BSA"]
        self._bets_columns = {"all": [c for c in cols]}
        self._bets_columns['h'] = [c for c in self._bets_columns['all'] if c[-1] == "H"]
        self._bets_columns['a'] = [c for c in self._bets_columns['all'] if c[-1] == "A"]
        self._bets_columns['a'] = [c for c in self._bets_columns['all'] if c[-1] == "D"]
        # -----Test Data----
        self._match_testdata.dropna(axis=0, subset=self._bets_columns['h'], how="all", inplace=True)
        self._match_testdata.dropna(axis=0, subset=self._bets_columns['a'], how="all", inplace=True)

    def __shrink_match_data_dimension(self):
        """
        The method will be responsible for deleting unwanted columns (feature) from the match data.
        :return:
        """
        home_player_X_positions = [f"home_player_X{i}" for i in range(1, 12)]
        home_player_Y_positions = [f"home_player_Y{i}" for i in range(1, 12)]
        away_player_X_positions = [f"away_player_X{i}" for i in range(1, 12)]
        away_player_Y_positions = [f"away_player_Y{i}" for i in range(1, 12)]

        # -----Test Data----
        for col in [home_player_X_positions, home_player_Y_positions, away_player_X_positions,
                    away_player_Y_positions]:
            self._match_testdata.drop(col, axis=1, inplace=True)

    def __add_team_rankings(self):
        """
        The method will be responsible for creating the Team Rankings features in the dataset.
        The team rankings features include the HomeTeamRanking and AwayTeamRanking which are based on the
        overall_rating of the players in each team's lineup.

        :return:
        """
        home_team_ids = self._test_set['HomeTeamAPI'].drop_duplicates().dropna().tolist()
        away_team_ids = self._test_set['AwayTeamAPI'].drop_duplicates().dropna().tolist()
        teams_players = {}

        for home_team, away_team in zip(home_team_ids, away_team_ids):
            df = self._match_testdata.loc[
                self._match_testdata['home_team_api_id'] == home_team]  # Get the dataframe of each home team
            home_team_lineup = df.loc[:,
                               'home_player_1':'home_player_11']  # Get the lineup of players id of the home team

            if home_team_lineup.shape[0] != 0:  # If loc result were 0 continue
                teams_players[home_team] = self.__unique_value_exctraction(home_team_lineup,
                                                                           list(home_team_lineup.columns))

            df = self._match_testdata.loc[
                self._match_testdata['away_team_api_id'] == away_team]  # Get the dataframe of each away team
            away_team_lineup = df.loc[:,
                               'away_player_1':'away_player_11']  # Get the lineup of players id of the away team

            if away_team_lineup.shape[0] != 0:  # If loc result were 0 continue
                teams_players[away_team] = self.__unique_value_exctraction(away_team_lineup,
                                                                           list(away_team_lineup.columns))

        team_average_players_ratings = {}

        for team, players in teams_players.items():
            if players:
                try:
                    players_ratings = self._player_attributes_testset.loc[list(players)]  # Get the team players ratings
                    team_average_players_ratings[team] = players_ratings.mean().at['overall_rating']
                except KeyError:
                    continue #TODO: Players from the lineup in the match table are not included in the _player_attributes_testset

        home_team_average_players_ratings = pd.DataFrame({"HomeTeamAPI": list(team_average_players_ratings.keys()),
                                                          "HomeTeamRatings": list(
                                                              team_average_players_ratings.values())})
        away_team_average_players_ratings = pd.DataFrame({"AwayTeamAPI": list(team_average_players_ratings.keys()),
                                                          "AwayTeamRatings": list(
                                                              team_average_players_ratings.values())})

        self._test_set = pd.merge(self._test_set, home_team_average_players_ratings, how="inner", on="HomeTeamAPI")
        self._test_set = pd.merge(self._test_set, away_team_average_players_ratings, how="inner", on="AwayTeamAPI")

        # ------- Test Data ------------

    def __add_team_stats(self):
        """
        The method will be responsible for creating the Team stats features in the dataset.
        The team stats features include the buildUpPlaySpeed, chanceCreationShooting and defencePressure of each team
        in each match.

        :return:
        """
        self._test_set = pd.merge(self._test_set, self._team_attributes_testdata, how="inner", left_on="HomeTeamAPI",
                                 right_on="team_api_id"). \
            rename(
            columns={'buildUpPlaySpeed': 'HomeTeamPlaySpeed', "chanceCreationShooting": "HomeTeamCreatonShooting",
                     "defencePressure": "HomeTeamDefencePressure"})
        self._test_set = pd.merge(self._test_set, self._team_attributes_testdata, how="inner", left_on="AwayTeamAPI",
                                 right_on="team_api_id"). \
            rename(
            columns={'buildUpPlaySpeed': 'AwayTeamPlaySpeed', "chanceCreationShooting": "AwayTeamCreatonShooting",
                     "defencePressure": "AwayTeamDefencePressure"})
        self._test_set = self._test_set.drop(columns={'team_api_id_x', 'team_api_id_y'})

    def __add_classification(self):
        """
        The method will be responsible for creating the labels for each match.
        0 - Draw
        1 - Home team wins
        2 - Home team loose (Away team win)
        :return:
        """
        win = []
        for l in range(0, len(self._test_set)):
            if self._test_set.HomeTeamGoals[l] > self._test_set.AwayTeamGoals[l]:
                k1 = 1
                win.append(k1)
            elif self._test_set.HomeTeamGoals[l] == self._test_set.AwayTeamGoals[l]:
                k1 = 0
                win.append(k1)
            else:
                k1 = 2
                win.append(k1)

        self._test_set['win'] = win

    def __parse_xml(self):
        self._match_testdata[['on_target_shot_home_team', 'on_target_shot_away_team']] = self._match_testdata[
            ['shoton', 'home_team_api_id', 'away_team_api_id']].apply(
            lambda x: self.__calculate_stats_both_teams(x['shoton'], x['home_team_api_id'], x['away_team_api_id']),
            axis=1,
            result_type="expand")
        # self.__mean_for_team_for_feat('on_target_shot_home_team', 'home')

        self._match_testdata[['yellow_card_home_team', 'yellow_card_away_team']] = self._match_testdata[
            ['card', 'home_team_api_id', 'away_team_api_id']].apply(
            lambda x: self.__calculate_stats_both_teams(x['card'], x['home_team_api_id'], x['away_team_api_id']),
            axis=1,
            result_type="expand")
        self._match_testdata[['red_card_home_team', 'red_card_away_team']] = self._match_testdata[
            ['card', 'home_team_api_id', 'away_team_api_id']].apply(
            lambda x: self.__calculate_stats_both_teams(x['card'], x['home_team_api_id'], x['away_team_api_id'],
                                                        card_type='r'), axis=1, result_type="expand")
        self._match_testdata[['crosses_home_team', 'crosses_away_team']] = self._match_testdata[
            ['cross', 'home_team_api_id', 'away_team_api_id']].apply(
            lambda x: self.__calculate_stats_both_teams(x['cross'], x['home_team_api_id'], x['away_team_api_id']),
            axis=1,
            result_type="expand")
        self._match_testdata[['corner_home_team', 'corner_away_team']] = self._match_testdata[
            ['corner', 'home_team_api_id', 'away_team_api_id']].apply(
            lambda x: self.__calculate_stats_both_teams(x['corner'], x['home_team_api_id'], x['away_team_api_id']),
            axis=1,
            result_type="expand")
        self._match_testdata[['possession_home_team', 'possession_away_team']] = self._match_testdata[
            ['possession', 'home_team_api_id', 'away_team_api_id']].apply(
            lambda x: self.__calculate_stats_both_teams(x['possession'], x['home_team_api_id'], x['away_team_api_id']),
            axis=1,
            result_type="expand")

    def __fill_with_mean(self, feature, home_or_away):
        for team in self._team_attributes_testdata['team_api_id'].tolist():
            team_matches = self._match_testdata.loc[self._match_testdata[f'{home_or_away}_team_api_id'] == team]

            if team_matches.shape[0] != 0:
                not_null = team_matches[~team_matches[feature].isna()]
                nulls = team_matches[team_matches[feature].isna()]
                matches_indexes = nulls.index.tolist()

                if not_null.shape[0] != 0 and nulls.shape[0] != 0:
                    avg = not_null[feature].mean()
                    if avg == 0:
                        #  If the average is 0 - all the rows have 0 value. Delete them
                        self._match_testdata.drop(matches_indexes, inplace=True)
                    else:
                        self._match_testdata.at[matches_indexes, feature] = avg

                else:
                    #  If all the values of the feature are null - delete it
                    self._match_testdata.drop(matches_indexes, inplace=True)

    def __calculate_stats_both_teams(self, xml_document, home_team, away_team, card_type='y'):
        if not xml_document:
            return None, None

        tree = ET.fromstring(xml_document)
        stat_home_team = 0
        stat_away_team = 0

        # Dealing with card type using the root element & the card type argument
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

        # Lets take the last possession stat which is available from the xml doc
        if tree.tag == 'possession':
            try:
                last_value = [child for child in tree.iter('value')][-1]
                return int(last_value.find('homepos').text), int(last_value.find('awaypos').text)
            except:
                return None, None

        # Taking care of all other stats by extracting based on the home team & away team api id's
        for team in [int(stat.text) for stat in tree.findall('value/team')]:
            if team == home_team:
                stat_home_team += 1
            else:
                stat_away_team += 1
        return stat_home_team, stat_away_team

    def __add_bets_ods_features(self):
        new_df = pd.DataFrame()

        for label, row in self._test_set.iterrows():
            away_team, home_team = row.at['HomeTeamAPI'], row.at['AwayTeamAPI']
            #  Get all the matches of the away_team and the home_team
            match = self._match_testdata.loc[(self._match_testdata['home_team_api_id'] == home_team) & (
                        self._match_testdata['away_team_api_id'] == away_team)]

            if match.shape[0] == 0:
                self.__remove_row(label)
                continue

            betting_ods = match.loc[:, self._bets_columns['all'][0]: self._bets_columns['all'][-1]]

            for bet, column in zip(['h', 'a'], ['HomeTeamsOdds', "AwayTeamOdds"]):  # TODO: Think if Draw is needed

                home_or_away_bets_odds = betting_ods.loc[:, self._bets_columns[bet]]
                #  For each match calculate the mean of all betting ods and that will be the match bet odd.
                betting_odd = home_or_away_bets_odds.fillna(0).values.mean()

                row[column] = betting_odd

            #  Create a new dataframe with the new Odss feature
            new_df = new_df.append(row)

        # update the new df
        del self._test_set
        self._test_set = new_df

    def __add_team_goals_avg(self):
        home_new_data = {"HomeTeamAPI": [], "HomeTeamAvgGoals": []}
        away_new_data = {"AwayTeamAPI": [], "AwayTeamAvgGoals": []}
        for label in self._team_attributes_testdata['team_api_id'].tolist():
            home_team_games = self._test_set.loc[(self._test_set['HomeTeamAPI'] == label)]
            home_team_goals_avg = home_team_games['HomeTeamGoals'].mean()
            home_new_data['HomeTeamAPI'] += [label]
            home_new_data['HomeTeamAvgGoals'] += [home_team_goals_avg]

            away_team_games = self._test_set.loc[(self._test_set['AwayTeamAPI'] == label)]
            away_team_goals_avg = away_team_games['AwayTeamGoals'].mean()
            away_new_data['AwayTeamAPI'] += [label]
            away_new_data['AwayTeamAvgGoals'] += [away_team_goals_avg]

        new_home_df = pd.DataFrame(home_new_data)
        new_away_df = pd.DataFrame(away_new_data)

        self._test_set = pd.merge(self._test_set, new_home_df, how="left", on="HomeTeamAPI")
        self._test_set = pd.merge(self._test_set, new_away_df, how="left", on="AwayTeamAPI")
        return

    def __add_goals_difference(self):
        """
        The method will be responsible for adding the goals difference between teams features.
        For each match the home team and away team will be taken into account in the goals difference
        calculation.
        :return:
        """
        copy_df = deepcopy(self._test_set)  # Create a copy of the dataset to not change it
        new_data = {"HomeTeamAPI": [], "AwayTeamAPI": [], "GoalDiff": []}

        #  Iterate over the data set until there are no more matches
        while copy_df.shape[0] > 0:
            match = copy_df.iloc[0]  # Take the first match each iteration

            away_team, home_team = match.at['HomeTeamAPI'], match.at['AwayTeamAPI']

            #  Get all the matches of the away_team against the home_team and vice versa
            matches1 = copy_df.loc[(copy_df['HomeTeamAPI'] == home_team) & (
                    copy_df['AwayTeamAPI'] == away_team)]
            matches2 = copy_df.loc[(copy_df['AwayTeamAPI'] == home_team) & (
                    copy_df['HomeTeamAPI'] == away_team)]

            home_goals = matches1["HomeTeamGoals"].sum()
            away_goals = matches2["AwayTeamGoals"].sum()

            total_home_team_goals = home_goals + away_goals

            home_goals = matches2["HomeTeamGoals"].sum()
            away_goals = matches1["AwayTeamGoals"].sum()

            total_away_team_goals = home_goals + away_goals

            diff = total_home_team_goals - total_away_team_goals

            new_data["HomeTeamAPI"] += [home_team]
            new_data["AwayTeamAPI"] += [away_team]
            new_data["GoalDiff"] += [diff]

            new_data["HomeTeamAPI"] += [away_team]
            new_data["AwayTeamAPI"] += [home_team]
            new_data["GoalDiff"] += [-diff]

            #  Delete the matches from the copy df
            copy_df.drop(list(matches1.index), axis="index", inplace=True)
            copy_df.drop(list(matches2.index), axis="index", inplace=True)

        new_data_df = pd.DataFrame(new_data)
        self._test_set = pd.merge(self._test_set, new_data_df, how="inner", on=["HomeTeamAPI", "AwayTeamAPI"])

    def __remove_row(self, row_index):
        self._test_set = self._test_set[self._test_set.index != row_index]

    def __join_match_table(self):
        to_join = self._match_testdata.loc[:, 'on_target_shot_home_team': 'possession_away_team']
        ids = self._match_testdata.loc[:, 'id']
        to_join = pd.concat([to_join, ids], axis=1)
        self._test_set = pd.merge(self._test_set, to_join, how="inner", on="id")

    def __remove_uneeded_features(self):
        self._test_set.drop(columns=['id'], inplace=True)
        self._test_set.drop(columns=['AwayTeam'], inplace=True)
        self._test_set.drop(columns=['HomeTeam'], inplace=True)


    def get_testdata(self):
        self.__clear_null_from_match()
        self.__shrink_match_data_dimension()

        self.__parse_xml()
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_SHOTON, definition.TOKEN_HOME_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_SHOTON, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_YELLOWCARD, definition.TOKEN_HOME_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_YELLOWCARD, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_REDCARD, definition.TOKEN_HOME_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_REDCARD, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_CROSSES, definition.TOKEN_HOME_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_CROSSES, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_CORNERS, definition.TOKEN_HOME_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_CORNERS, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_POSS, definition.TOKEN_HOME_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_POSS, definition.TOKEN_AWAY_TEAM_NAME)
        self.__join_match_table()
        self.__add_team_stats()
        self.__add_team_goals_avg()
        self.__add_goals_difference()
        self.__add_bets_ods_features()
        self.__add_team_rankings()
        self.__add_classification()

        self.__remove_uneeded_features()
        self._database_connection.close()

        return self._test_set

    def __create_init_dataset(self):
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
                                                   WHERE season = '2015/2016'                         
                                                   ORDER by date
                                                   ;""", self._database_connection)

        data1 = test_data[["home_team", "away_team", "season", "home_team_goal", "away_team_goal"]]

        self._test_set = pd.DataFrame(
            {"id": test_data["id"],
             "HomeTeamAPI": test_data['home_team_api_id'], "HomeTeam": data1.home_team + data1.season,
             'AwayTeamAPI': test_data['away_team_api_id'],
             "AwayTeam": data1.away_team + data1.season, "HomeTeamGoals": data1.home_team_goal,
             "AwayTeamGoals": data1.away_team_goal})

    def __load_player_attr_table(self):
        self._player_attributes_testset = pd.read_sql_query("""SELECT DISTINCT player_api_id, overall_rating 
                                                            FROM Player_Attributes
                                                            WHERE strftime('%Y',date)='2015' or strftime('%Y',date)='2016'

                                                            """, self._database_connection)
        # set the index to be the player_api_id field
        self._player_attributes_testset.set_index('player_api_id', inplace=True, drop=True)

    def __load_team_attr_table(self):
        self._team_attributes_testdata = pd.read_sql("""SELECT team_api_id, buildUpPlaySpeed, chanceCreationShooting, defencePressure  
                                                    FROM Team_Attributes
                                                    WHERE strftime('%Y',date)='2015' or strftime('%Y',date)='2016'
                                                    """, self._database_connection)

    def __load_match_table(self):
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


t = TestData("database.sqlite")
data = t.get_testdata()
data.to_csv("testset.csv", index=False)
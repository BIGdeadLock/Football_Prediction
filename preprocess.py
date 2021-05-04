import pandas as pd
import definition
import sqlite3  # SQLite
import xml.etree.ElementTree as ET
from copy import deepcopy

definition.TOKEN_DS_HOME_TEAM_AVG_GOALS = "HomeTeamAvgGoals"

definition.TOKEN_LEFT_JOIN = "left"

definition.TOKEN_TEAM_ATTR_ID = 'team_api_id'

definition.TOKEN_INNER_JOIN = "inner"

definition.TOKEN_TEAM_DEF_PRESS = "defencePressure"

definition.TOKEN_TEAM_SPEED = 'buildUpPlaySpeed'

definition.TOKEN_MATCH_AWAY_TEAM_ID = 'away_team_api_id'

definition.TOKEN_MATCH_HOME_TEAM_ID = 'home_team_api_id'

defintion.TOKEN_MATCH_CARD = 'card'

definiton.TOKEN_MATCH_AWAY_TEAM_REDCARD = 'red_card_away_team'

definition.TOKEN_MATCH_HOME_TEAM_REDCARD = 'red_card_home_team'

definition.TOKEN_MATCH_HOME_TEAM_REDCARD = 'red_card_home_team'

definition.TOKEN_MATCH_CARD = 'card'

definition.TOKEN_MATCH_HOME_TEAM_ID = 'home_team_api_id'

definition.TOKEN_MATCH_HOME_TEAM_ID = 'home_team_api_id'


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

        self.__parse_xml()
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_SHOTON, definition.TOKEN_HOME_TEAM)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_SHOTON, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_YELLOWCARD, definition.TOKEN_HOME_TEAM)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_YELLOWCARD, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_REDCARD, definition.TOKEN_HOME_TEAM)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_REDCARD, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_CROSSES, definition.TOKEN_HOME_TEAM)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_CROSSES, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_CORNERS, definition.TOKEN_HOME_TEAM)
        self.__fill_with_mean(definition.TOKEN_MATCH_AWAY_TEAM_CORNERS, definition.TOKEN_AWAY_TEAM_NAME)
        self.__fill_with_mean(definition.TOKEN_MATCH_HOME_TEAM_POSS, definition.TOKEN_HOME_TEAM)
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
             definition.TOKEN_DS_AWAY_TEAM_ID : data[definition.TOKEN_MATCH_AWAY_TEAM_ID],
             definition.TOKEN_DS_AWAY_TEAM_NAME: data1.away_team + data1.season, definition.TOKEN_DS_HOME_TEAM_GOALS: data1.home_team_goal,
             definition.TOKEN_DS_AWAY_TEAM_GOALS: data1.away_team_goal})

    def __load_player_attr_table(self):
        self._player_attributes_data = pd.read_sql_query("""SELECT DISTINCT player_api_id, overall_rating 
                                                       FROM Player_Attributes
                                                       WHERE strftime('%Y',date)<>'2015' or strftime('%Y',date)<>'2016'
                                                       GROUP BY player_api_id                         
                                                       """, self._database_connection)
        # set the index to be the player_api_id field
        self._player_attributes_data.set_index('player_api_id', inplace=True, drop=True)


    def __load_team_attr_table(self):
        self._team_attributes_data = pd.read_sql("""SELECT team_api_id, buildUpPlaySpeed, chanceCreationShooting, defencePressure  
                                                FROM Team_Attributes
                                                WHERE strftime('%Y',date)<>'2015' or strftime('%Y',date)<>'2016'
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
        for col in [definition.TOKEN_MATCH_HOME_PLAYERS_X_POS, definition.TOKEN_MATCH_HOME_PLAYERS_Y_POS,
                    definition.TOKEN_MATCH_AWAY_PLAYERS_X_POS, definition.TOKEN_MATCH_AWAY_PLAYERS_Y_POS,
                    definition.TOKEN_MATCH_GOALS]:
            self._match_data.drop(col, axis=1, inplace=True)

    def __add_team_rankings(self):
        """
        The method will be responsible for creating the Team Rankings features in the dataset.
        The team rankings features include the HomeTeamRanking and AwayTeamRanking which are based on the
        overall_rating of the players in each team's lineup.

        :return:
        """
        home_team_ids = self._dataset[definition.TOKEN_DS_HOME_TEAM_ID].drop_duplicates().dropna().tolist()
        away_team_ids = self._dataset[definition.TOKEN_DS_AWAY_TEAM_ID].drop_duplicates().dropna().tolist()
        teams_players = {}

        for home_team, away_team in zip(home_team_ids, away_team_ids):
            df = self._match_data.loc[
                self._match_data[definition.TOKEN_MATCH_HOME_TEAM_ID] == home_team]  # Get the dataframe of each home team
            home_team_lineup = df.loc[:,
                               definition.TOKEN_MATCH_HOME_PLAYERS_ID[0]:definition.TOKEN_MATCH_HOME_PLAYERS_ID[10]]  # Get the lineup of players id of the home team

            if home_team_lineup.shape[0] != 0: # If loc result were 0 continue
                teams_players[home_team] = self.__unique_value_exctraction(home_team_lineup,
                                                                           list(home_team_lineup.columns))

            df = self._match_data.loc[
                self._match_data[definition.TOKEN_MATCH_AWAY_TEAM_ID] == away_team]  # Get the dataframe of each away team
            away_team_lineup = df.loc[:,
                               definition.TOKEN_MATCH_AWAY_PLAYERS_ID[0]:definition.TOKEN_MATCH_AWAY_PLAYERS_ID[10]]  # Get the lineup of players id of the away team

            if away_team_lineup.shape[0] != 0:  # If loc result were 0 continue
                teams_players[away_team] = self.__unique_value_exctraction(away_team_lineup,
                                                                           list(away_team_lineup.columns))


        team_average_players_ratings = {}

        for team, players in teams_players.items():
            if players:
                players_ratings = self._player_attributes_data.loc[list(players)]  # Get the team players ratings
                team_average_players_ratings[team] = players_ratings.mean().at[definition.TOKEN_PLAYER_ATTRIB_OVERALL]

        home_team_average_players_ratings = pd.DataFrame({definition.TOKEN_DS_HOME_TEAM_ID: list(team_average_players_ratings.keys()),
                                                          definition.TOKEN_DS_HOME_TEAM_Rating: list(
                                                              team_average_players_ratings.values())})
        away_team_average_players_ratings = pd.DataFrame({definition.TOKEN_DS_AWAY_TEAM_ID: list(team_average_players_ratings.keys()),
                                                          definition.TOKEN_DS_AWAY_TEAM_Rating: list(
                                                              team_average_players_ratings.values())})

        self._dataset = pd.merge(self._dataset, home_team_average_players_ratings, how=definition.TOKEN_INNER_JOIN,
                                 on=definition.TOKEN_DS_HOME_TEAM_ID)
        self._dataset = pd.merge(self._dataset, away_team_average_players_ratings, how=definition.TOKEN_INNER_JOIN,
                                 on=definition.TOKEN_DS_AWAY_TEAM_ID)


    def __add_team_stats(self):
        """
        The method will be responsible for creating the Team stats features in the dataset.
        The team stats features include the buildUpPlaySpeed, chanceCreationShooting and defencePressure of each team
        in each match.

        :return:
        """
        self._dataset = pd.merge(self._dataset, self._team_attributes_data, how=definition.TOKEN_INNER_JOIN,
                                 left_on=definition.TOKEN_DS_HOME_TEAM_ID,
                                 right_on=definition.TOKEN_TEAM_ATTR_ID). \
            rename(
            columns={definition.TOKEN_TEAM_SPEED: definition.TOKEN_HOME_TEAM_SPEED, definition.TOKEN_TEAM_CHANES: definition.TOKEN_HOME_TEAM_SHOOT,
                     definition.TOKEN_TEAM_DEF_PRESS: definition.TOKEN_HOME_TEAM_DEF})
        self._dataset = pd.merge(self._dataset, self._team_attributes_data, how=definition.TOKEN_INNER_JOIN,
                                 left_on=definition.TOKEN_DS_AWAY_TEAM_ID,
                                 right_on=definition.TOKEN_TEAM_ATTR_ID). \
            rename(
            columns={definition.TOKEN_TEAM_SPEED: definition.TOKEN_AWAY_TEAM_SPEED,  definition.TOKEN_TEAM_CHANES: definition.TOKEN_AWAY_TEAM_SHOOT,
                     definition.TOKEN_TEAM_DEF_PRESS: definition.TOKEN_AWAY_TEAM_DEF})
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
            if self._dataset.HomeTeamGoals[l] > self._dataset.AwayTeamGoals[l]:
                k1 = 1
                win.append(k1)
            elif self._dataset.HomeTeamGoals[l] == self._dataset.AwayTeamGoals[l]:
                k1 = 0
                win.append(k1)
            else:
                k1 = 2
                win.append(k1)

        self._dataset[definition.TOKEN_CLASS_NAME] = win

    def __parse_xml(self):
        self._match_data[[definition.TOKEN_MATCH_HOME_TEAM_SHOTON, definition.TOKEN_MATCH_AWAY_TEAM_SHOTON]] =\
            self._match_data[
            [definition.TOKEN_MATCH_SHOTON, definition.TOKEN_MATCH_HOME_TEAM_ID,
             definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_SHOTON],
                                                        x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                        x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
            result_type="expand")

        self._match_data[[definition.TOKEN_MATCH_HOME_TEAM_YELLOWCARD, definition.TOKEN_MATCH_AWAY_TEAM_YELLOWCARD]] =\
            self._match_data[
            [definition.TOKEN_MATCH_CARD, definition.TOKEN_MATCH_HOME_TEAM_ID,
             definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CARD],
            x[definition.TOKEN_MATCH_HOME_TEAM_ID],
            x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
            result_type="expand")
        self._match_data[[definition.TOKEN_MATCH_HOME_TEAM_REDCARD, definition.TOKEN_MATCH_AWAY_TEAM_REDCARD]] = \
            self._match_data[[definition.TOKEN_MATCH_CARD, definition.TOKEN_MATCH_HOME_TEAM_ID,
                              definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CARD],
                x[definition.TOKEN_MATCH_HOME_TEAM_ID], x[
                definition.TOKEN_MATCH_AWAY_TEAM_ID],
                                                        card_type='r'), axis=1, result_type="expand")

        self._match_data[[definition.TOKEN_MATCH_HOME_TEAM_CROSSES, definition.TOKEN_MATCH_AWAY_TEAM_CROSSES]] = \
            self._match_data[[definition.TOKEN_MATCH_CROSS, definition.TOKEN_MATCH_HOME_TEAM_ID,
         definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CROSS],
        x[definition.TOKEN_MATCH_HOME_TEAM_ID], x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
            result_type="expand")

        self._match_data[[definition.TOKEN_MATCH_HOME_TEAM_CORNERS, definition.TOKEN_MATCH_AWAY_TEAM_CORNERS]] =\
            self._match_data[[definition.TOKEN_MATCH_CORNERS,
           definition.TOKEN_MATCH_HOME_TEAM_ID,
           definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CORNERS],
            x[definition.TOKEN_MATCH_HOME_TEAM_ID], x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
            result_type="expand")

        self._match_data[[definition.TOKEN_MATCH_HOME_TEAM_POSS, definition.TOKEN_MATCH_AWAY_TEAM_POSS]] = self._match_data[
            [definition.TOKEN_MATCH_POSS, definition.TOKEN_MATCH_HOME_TEAM_ID, definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_POSS], x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                        x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
            result_type="expand")


    def __fill_with_mean(self, feature, home_or_away):
        """
        The method will calculate the mean of row in the feature given until that row on theat feature
        :param feature: String. The feature in the df
        :param home_or_away: String. 'home' or 'away' for indciting if the feature is for the home team or away team
        :return:
        """
        for team in self._team_attributes_data[definition.TOKEN_TEAM_ATTR_ID].tolist():
            team_matches = self._match_data.loc[self._match_data[f'{home_or_away}_{definition.TOKEN_TEAM_ATTR_ID}']== team]

            if team_matches.shape[0] != 0:
                not_null = team_matches[~team_matches[feature].isna()]
                nulls = team_matches[team_matches[feature].isna()]
                matches_indexes = nulls.index.tolist()

                if not_null.shape[0] != 0 and nulls.shape[0] != 0:
                    avg = not_null[feature].mean()
                    if avg == 0:
                        #  If the average is 0 - all the rows have 0 value. Delete them
                        self._match_data.drop(matches_indexes, inplace=True)
                    else:
                        self._match_data.at[matches_indexes, feature] = avg

                else:
                    #  If all the values of the feature are null - delete it
                    self._match_data.drop(matches_indexes, inplace=True)




    def __calculate_stats_both_teams(self, xml_document, home_team, away_team, card_type='y'):
        if not xml_document:
            return None,None

        tree = ET.fromstring(xml_document)
        stat_home_team = 0
        stat_away_team = 0

        # Dealing with card type using the root element & the card type argument
        if tree.tag == definition.TOKEN_MATCH_CARD:
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
        if tree.tag == definition.TOKEN_MATCH_POSS:
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

        for label, row in self._dataset.iterrows():
            away_team ,home_team = row.at[definition.TOKEN_DS_HOME_TEAM_ID], row.at[definition.TOKEN_DS_AWAY_TEAM_ID]
            #  Get all the matches of the away_team and the home_team
            match = self._match_data.loc[(self._match_data[definition.TOKEN_MATCH_HOME_TEAM_ID] == home_team) &
                                         (self._match_data[
            definition.TOKEN_MATCH_AWAY_TEAM_ID] == away_team)]

            if match.shape[0] == 0:
                self.__remove_row(label)
                continue

            betting_ods = match.loc[:, self._bets_columns['all'][0]: self._bets_columns['all'][-1]]

            for bet, column in zip(['h', 'a'], [definition.TOKEN_DS_HOME_TEAM_ODDS, definition.TOKEN_DS_AWAY_TEAM_ODDS]):

                home_or_away_bets_odds = betting_ods.loc[:, self._bets_columns[bet]]
                #  For each match calculate the mean of all betting ods and that will be the match bet odd.
                betting_odd = home_or_away_bets_odds.fillna(0).values.mean()

                row[column] = betting_odd

            #  Create a new dataframe with the new Odds feature
            new_df = new_df.append(row)

        # update the new df
        del self._dataset
        self._dataset = new_df

    def __add_team_goals_avg(self):
        home_new_data = {definition.TOKEN_DS_HOME_TEAM_ID:[], definition.TOKEN_DS_HOME_TEAM_AVG_GOALS: []}
        away_new_data = {definition.TOKEN_DS_AWAY_TEAM_ID:[], definition.TOKEN_DS_AWAY_TEAM_AVG_GOALS: []}
        for label in self._team_attributes_data[definition.TOKEN_TEAM_ATTR_ID].tolist():
            home_team_games = self._dataset.loc[(self._dataset[definition.TOKEN_DS_HOME_TEAM_ID] == label)]
            home_team_goals_avg = home_team_games[definition.TOKEN_DS_HOME_TEAM_GOALS].mean()
            home_new_data[definition.TOKEN_DS_HOME_TEAM_ID] += [label]
            home_new_data[definition.TOKEN_DS_HOME_TEAM_AVG_GOALS] += [home_team_goals_avg]

            away_team_games = self._dataset.loc[(self._dataset[definition.TOKEN_DS_AWAY_TEAM_ID] == label)]
            away_team_goals_avg = away_team_games[definition.TOKEN_DS_AWAY_TEAM_GOALS].mean()
            away_new_data[definition.TOKEN_DS_AWAY_TEAM_ID] += [label]
            away_new_data[definition.TOKEN_DS_AWAY_TEAM_AVG_GOALS] += [away_team_goals_avg]

        new_home_df = pd.DataFrame(home_new_data)
        new_away_df = pd.DataFrame(away_new_data)

        self._dataset = pd.merge(self._dataset, new_home_df, how=definition.TOKEN_LEFT_JOIN, on=definition.TOKEN_DS_HOME_TEAM_ID)
        self._dataset = pd.merge(self._dataset, new_away_df, how=definition.TOKEN_LEFT_JOIN, on=definition.TOKEN_DS_AWAY_TEAM_ID)
        return

    def __add_goals_difference(self):
        """
        The method will be responsible for adding the goals difference between teams features.
        For each match the home team and away team will be taken into account in the goals difference
        calculation.
        :return:
        """
        copy_df = deepcopy(self._dataset) # Create a copy of the dataset to not change it
        new_data = {definition.TOKEN_DS_HOME_TEAM_ID: [], definition.TOKEN_DS_AWAY_TEAM_ID:[],
                    definition.TOKEN_DS_GOALDIFF: []}

        #  Iterate over the data set until there are no more matches
        while copy_df.shape[0] > 0:
            match = copy_df.iloc[0] # Take the first match each iteration

            away_team, home_team = match.at[definition.TOKEN_DS_HOME_TEAM_ID], match.at[definition.TOKEN_DS_AWAY_TEAM_ID]

            #  Get all the matches of the away_team against the home_team and vice versa
            matches1 = copy_df.loc[(copy_df[definition.TOKEN_DS_HOME_TEAM_ID] == home_team) & (
                        copy_df[definition.TOKEN_DS_AWAY_TEAM_ID] == away_team)]
            matches2 = copy_df.loc[(copy_df[definition.TOKEN_DS_AWAY_TEAM_ID] == home_team) & (
                    copy_df[definition.TOKEN_DS_HOME_TEAM_ID] == away_team)]

            home_goals = matches1[definition.TOKEN_DS_HOME_TEAM_GOALS].sum()
            away_goals = matches2[definition.TOKEN_DS_AWAY_TEAM_GOALS].sum()

            total_home_team_goals = home_goals + away_goals

            home_goals = matches2[definition.TOKEN_DS_HOME_TEAM_GOALS].sum()
            away_goals = matches1[definition.TOKEN_DS_AWAY_TEAM_GOALS].sum()

            total_away_team_goals = home_goals + away_goals

            diff = total_home_team_goals - total_away_team_goals

            new_data[definition.TOKEN_DS_HOME_TEAM_ID] += [home_team]
            new_data[definition.TOKEN_DS_AWAY_TEAM_ID] += [away_team]
            new_data[definition.TOKEN_DS_GOALDIFF] += [diff]

            new_data[definition.TOKEN_DS_HOME_TEAM_ID] += [away_team]
            new_data[definition.TOKEN_DS_AWAY_TEAM_ID] += [home_team]
            new_data[definition.TOKEN_DS_GOALDIFF] += [-diff]

            #  Delete the matches from the copy df
            copy_df.drop(list(matches1.index), axis=definition.TOKEN_INDEX_AXIS, inplace=True)
            copy_df.drop(list(matches2.index), axis=definition.TOKEN_INDEX_AXIS, inplace=True)

        new_data_df = pd.DataFrame(new_data)
        self._dataset = pd.merge(self._dataset, new_data_df, how=definition.TOKEN_INNER_JOIN,
                                 on=[definition.TOKEN_DS_HOME_TEAM_ID,definition.TOKEN_DS_AWAY_TEAM_ID ])

    def __remove_row(self, row_index):
            self._dataset = self._dataset[self._dataset.index != row_index]

    def __join_match_table(self):
        to_join = self._match_data.loc[:, definition.TOKEN_MATCH_HOME_TEAM_SHOTON: definition.TOKEN_MATCH_AWAY_TEAM_POSS]
        ids = self._match_data.loc[:, definition.TOKEN_MATCH_ID]
        to_join = pd.concat([to_join, ids], axis=1)
        self._dataset = pd.merge(self._dataset, to_join, how=definition.TOKEN_INNER_JOIN, on=definition.TOKEN_MATCH_ID)

    def __mean_for_features(self):
        pass

    def __remove_uneeded_features(self):
        self._dataset.drop(columns=[definition.TOKEN_MATCH_ID], inplace=True)
        self._dataset.drop(columns=[definition.TOKEN_DS_HOME_TEAM_NAME], inplace=True)
        self._dataset.drop(columns=[definition.TOKEN_DS_AWAY_TEAM_NAME], inplace=True)
        self._dataset.drop(columns=[definition.TOKEN_DS_AWAY_TEAM_ID], inplace=True)
        self._dataset.drop(columns=[definition.TOKEN_DS_HOME_TEAM_ID], inplace=True)
        self._dataset.drop(columns=[definition.TOKEN_DS_AWAY_TEAM_GOALS], inplace=True)
        self._dataset.drop(columns=[definition.TOKEN_DS_HOME_TEAM_GOALS], inplace=True)


p = FootballPreprocessesor("database.sqlite")
data = p.preprocess()
data.to_csv("dataset_no_2015_2016_no_draw.csv", index=False)
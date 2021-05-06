import pandas as pd
import definition
import sqlite3  # SQLite
import xml.etree.ElementTree as ET
from copy import deepcopy
from utils import inner_join, unique_value_exctraction, dimentions_reduction, remove_row
from dataset import Dataset


class FootballPreprocessesor(object):
    """
    The object that will wrap all the football data cleaning and manipulation functionalities.
    """

    def __init__(self, dt: Dataset):

        self._trainset, self._match_data, self._team_attributes_data, self._player_attributes_data = dt.get_train_dataset()
        self._testset, self._match_testdata, self._team_attributes_testdata, self._player_attributes_testdata = dt.get_test_dataset()

        self._matches_data = [self._match_data, self._match_testdata]
        self._teams_data = [self._team_attributes_data, self._team_attributes_testdata]
        self._players_data = [self._player_attributes_data, self._player_attributes_testdata]
        self._dataset = [self._trainset, self._testset]

    def preprocess(self) -> list:
        """
        The main method that start the preprocess flow.

        :return: list of DataFrame object containing the preprocessed Data of the train and test sets
        """
        features = [definition.TOKEN_MATCH_HOME_PLAYERS_X_POS, definition.TOKEN_MATCH_HOME_PLAYERS_Y_POS,
                    definition.TOKEN_MATCH_AWAY_PLAYERS_X_POS, definition.TOKEN_MATCH_AWAY_PLAYERS_Y_POS,
                    definition.TOKEN_MATCH_GOALS]
        for i, data in enumerate(self._matches_data):
            data = self.__clear_null_from_match(data)
            data = dimentions_reduction(data, features)
            data = self.__parse_xml(data)
            for feat, team in [(definition.TOKEN_MATCH_HOME_TEAM_SHOTON, definition.TOKEN_HOME_TEAM),
                               (definition.TOKEN_MATCH_AWAY_TEAM_SHOTON, definition.TOKEN_AWAY_TEAM),
                               (definition.TOKEN_MATCH_HOME_TEAM_YELLOWCARD, definition.TOKEN_HOME_TEAM),
                               (definition.TOKEN_MATCH_AWAY_TEAM_YELLOWCARD, definition.TOKEN_AWAY_TEAM),
                               (definition.TOKEN_MATCH_HOME_TEAM_REDCARD, definition.TOKEN_HOME_TEAM),
                               (definition.TOKEN_MATCH_AWAY_TEAM_REDCARD, definition.TOKEN_AWAY_TEAM),
                               (definition.TOKEN_MATCH_HOME_TEAM_CROSSES, definition.TOKEN_HOME_TEAM),
                               (definition.TOKEN_MATCH_AWAY_TEAM_CROSSES, definition.TOKEN_AWAY_TEAM),
                               (definition.TOKEN_MATCH_HOME_TEAM_CORNERS, definition.TOKEN_HOME_TEAM),
                               (definition.TOKEN_MATCH_AWAY_TEAM_CORNERS, definition.TOKEN_AWAY_TEAM),
                               (definition.TOKEN_MATCH_HOME_TEAM_POSS, definition.TOKEN_HOME_TEAM),
                               (definition.TOKEN_MATCH_AWAY_TEAM_POSS, definition.TOKEN_AWAY_TEAM)]:
                data = self.__fill_with_mean(feat, team, self._teams_data[i], data)


            self._dataset[i] = inner_join(left_df=self._dataset[i], right_df=data,
                              features=[definition.TOKEN_MATCH_HOME_TEAM_SHOTON,
                              definition.TOKEN_MATCH_AWAY_TEAM_POSS],
                              on=definition.TOKEN_MATCH_ID) # Join the match data with the train / test data
            data = self._dataset[i]
            # Start the dataset creating process
            data = self.__add_team_stats(data=data, team_data=self._teams_data[i])
            data = self.__add_team_goals_avg(data=data, team_data=self._teams_data[i])
            data = self.__add_goals_difference(data=data)
            data = self.__add_bets_ods_features(data=data, matches_data=self._matches_data[i])
            data = self.__add_team_rankings(data=data, match_data=self._matches_data[i],
                                            players_data=self._players_data[i])
            data = self.__add_classification(dataset=data) # Add the wins classification
            self.__remove_uneeded_features(data=data)
            self._dataset[i] = data

        return self._dataset

    def __clear_null_from_match(self, match_data: pd.DataFrame) -> pd.DataFrame:
        """
        The method will be responsible for deleting nulls from the match data based on rules.
        :return: DataFrame
        """

        # Rule 1 - Clear rows which contains nulls in all the home bets odds columns or away team bets odds columns
        cols = match_data.loc[:, "B365H":"BSA"]
        self._bets_columns = {"all": [c for c in cols]}
        self._bets_columns['h'] = [c for c in self._bets_columns['all'] if c[-1] == "H"]
        self._bets_columns['a'] = [c for c in self._bets_columns['all'] if c[-1] == "A"]
        self._bets_columns['a'] = [c for c in self._bets_columns['all'] if c[-1] == "D"]
        # Drop a match observation if all the home team bets have null, like wise to the away_team
        match_data.dropna(axis=0, subset=self._bets_columns['h'], how="all", inplace=True)
        match_data.dropna(axis=0, subset=self._bets_columns['a'], how="all", inplace=True)
        return match_data



    def __add_team_rankings(self, data, match_data, players_data) -> pd.DataFrame:
        """
        The method will be responsible for creating the Team Rankings features in the Data.
        The team rankings features include the HomeTeamRanking and AwayTeamRanking which are based on the
        overall_rating of the players in each team's lineup.
        :param data: DataFrame. The data to add the new features
        :param match_data: DataFrame. The matches data
        :param players_data: DataFrame. The players data
        :return: DataFrame.
        """
        home_team_ids = data[definition.TOKEN_DS_HOME_TEAM_ID].drop_duplicates().dropna().tolist()
        away_team_ids = data[definition.TOKEN_DS_AWAY_TEAM_ID].drop_duplicates().dropna().tolist()
        teams_players = {}

        for home_team, away_team in zip(home_team_ids, away_team_ids):
            df = match_data.loc[
                match_data[
                    definition.TOKEN_MATCH_HOME_TEAM_ID] == home_team]  # Get the dataframe of each home team
            home_team_lineup = df.loc[:,
                               definition.TOKEN_MATCH_HOME_PLAYERS_ID[0]:definition.TOKEN_MATCH_HOME_PLAYERS_ID[
                                   10]]  # Get the lineup of players id of the home team

            if home_team_lineup.shape[0] != 0:  # If loc result were 0 continue
                teams_players[home_team] = unique_value_exctraction(home_team_lineup,
                                                                           list(home_team_lineup.columns))

            df = match_data.loc[ match_data[ definition.TOKEN_MATCH_AWAY_TEAM_ID] == away_team]  # Get the dataframe of each away team
            away_team_lineup = df.loc[:,
                               definition.TOKEN_MATCH_AWAY_PLAYERS_ID[0]:definition.TOKEN_MATCH_AWAY_PLAYERS_ID[
                                   10]]  # Get the lineup of players id of the away team

            if away_team_lineup.shape[0] != 0:  # If loc result were 0 continue
                teams_players[away_team] = unique_value_exctraction(away_team_lineup,
                                                                           list(away_team_lineup.columns))

        team_average_players_ratings = {}

        for team, players in teams_players.items():
            if players:
                try:
                    players_ratings = players_data.loc[list(players)]  # Get the team players ratings
                    team_average_players_ratings[team] = players_ratings.mean().at[definition.TOKEN_PLAYER_ATTRIB_OVERALL]
                except KeyError:
                    continue # Some players are not in the players rating. Nothing we can do about them


        home_team_average_players_ratings = pd.DataFrame(
            {definition.TOKEN_DS_HOME_TEAM_ID: list(team_average_players_ratings.keys()),
             definition.TOKEN_DS_HOME_TEAM_Rating: list(
                 team_average_players_ratings.values())})
        away_team_average_players_ratings = pd.DataFrame(
            {definition.TOKEN_DS_AWAY_TEAM_ID: list(team_average_players_ratings.keys()),
             definition.TOKEN_DS_AWAY_TEAM_Rating: list(
                 team_average_players_ratings.values())})

        data = pd.merge(data, home_team_average_players_ratings, how=definition.TOKEN_INNER_JOIN,
                                 on=definition.TOKEN_DS_HOME_TEAM_ID)
        data = pd.merge(data, away_team_average_players_ratings, how=definition.TOKEN_INNER_JOIN,
                                 on=definition.TOKEN_DS_AWAY_TEAM_ID)
        return data

    def __add_team_stats(self, data: pd.DataFrame, team_data:pd.DataFrame) -> pd.DataFrame:
        """
        The method will be responsible for creating the Team stats features in the Data.
        The team stats features include the buildUpPlaySpeed, chanceCreationShooting and defencePressure of each team
        in each match.
        :param data: DataFrame. The data to which the features will be added.
        :param team_data: DataFrame. The teams data from the db
        :return: DataFrame containing the team stats features
        """
        data = pd.merge(data, team_data, how=definition.TOKEN_INNER_JOIN,
                                 left_on=definition.TOKEN_DS_HOME_TEAM_ID,
                                 right_on=definition.TOKEN_TEAM_ATTR_ID). \
            rename(
            columns={definition.TOKEN_TEAM_SPEED: definition.TOKEN_HOME_TEAM_SPEED,
                     definition.TOKEN_TEAM_CHANES: definition.TOKEN_HOME_TEAM_SHOOT,
                     definition.TOKEN_TEAM_DEF_PRESS: definition.TOKEN_HOME_TEAM_DEF})
        data = pd.merge(data, team_data, how=definition.TOKEN_INNER_JOIN,
                                 left_on=definition.TOKEN_DS_AWAY_TEAM_ID,
                                 right_on=definition.TOKEN_TEAM_ATTR_ID). \
            rename(
            columns={definition.TOKEN_TEAM_SPEED: definition.TOKEN_AWAY_TEAM_SPEED,
                     definition.TOKEN_TEAM_CHANES: definition.TOKEN_AWAY_TEAM_SHOOT,
                     definition.TOKEN_TEAM_DEF_PRESS: definition.TOKEN_AWAY_TEAM_DEF})
        data = data.drop(columns={'team_api_id_x', 'team_api_id_y'})
        return data

    def __add_classification(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        The method will be responsible for creating the labels for each match.
        0 - Draw
        1 - Home team wins
        2 - Home team loose (Away team win)

        :param dataset: The dataset the wins label will be added to
        :return: DataFrame
        """
        win = []
        for l in range(0, len(dataset)):
            if dataset.HomeTeamGoals[l] > dataset.AwayTeamGoals[l]:
                k1 = 1
                win.append(k1)
            elif dataset.HomeTeamGoals[l] == dataset.AwayTeamGoals[l]:
                k1 = 0
                win.append(k1)
            else:
                k1 = 2
                win.append(k1)

        dataset[definition.TOKEN_CLASS_NAME] = win
        return dataset

    def __parse_xml(self, match_data) -> pd.DataFrame:
        """
        The method will be used all the features with xml values and will extract the numeric values from them.
        The numeric values will be added under a new feature name.
        :param match_data: DataFrame. The matches data.
        :return: DataFrame
        """
        match_data[[definition.TOKEN_MATCH_HOME_TEAM_SHOTON, definition.TOKEN_MATCH_AWAY_TEAM_SHOTON]] = \
            match_data[
                [definition.TOKEN_MATCH_SHOTON, definition.TOKEN_MATCH_HOME_TEAM_ID,
                 definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
                lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_SHOTON],
                                                            x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                            x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
                result_type="expand")

        match_data[[definition.TOKEN_MATCH_HOME_TEAM_YELLOWCARD, definition.TOKEN_MATCH_AWAY_TEAM_YELLOWCARD]] = \
            self._match_data[
                [definition.TOKEN_MATCH_CARD, definition.TOKEN_MATCH_HOME_TEAM_ID,
                 definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
                lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CARD],
                                                            x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                            x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
                result_type="expand")
        match_data[[definition.TOKEN_MATCH_HOME_TEAM_REDCARD, definition.TOKEN_MATCH_AWAY_TEAM_REDCARD]] = \
            match_data[[definition.TOKEN_MATCH_CARD, definition.TOKEN_MATCH_HOME_TEAM_ID,
                        definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
                lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CARD],
                                                            x[definition.TOKEN_MATCH_HOME_TEAM_ID], x[
                                                                definition.TOKEN_MATCH_AWAY_TEAM_ID],
                                                            card_type='r'), axis=1, result_type="expand")

        match_data[[definition.TOKEN_MATCH_HOME_TEAM_CROSSES, definition.TOKEN_MATCH_AWAY_TEAM_CROSSES]] = \
            match_data[[definition.TOKEN_MATCH_CROSS, definition.TOKEN_MATCH_HOME_TEAM_ID,
                        definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
                lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CROSS],
                                                            x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                            x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
                result_type="expand")

        match_data[[definition.TOKEN_MATCH_HOME_TEAM_CORNERS, definition.TOKEN_MATCH_AWAY_TEAM_CORNERS]] = \
            match_data[[definition.TOKEN_MATCH_CORNERS,
                        definition.TOKEN_MATCH_HOME_TEAM_ID,
                        definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
                lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_CORNERS],
                                                            x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                            x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
                result_type="expand")

        match_data[[definition.TOKEN_MATCH_HOME_TEAM_POSS, definition.TOKEN_MATCH_AWAY_TEAM_POSS]] = match_data[
            [definition.TOKEN_MATCH_POSS, definition.TOKEN_MATCH_HOME_TEAM_ID,
             definition.TOKEN_MATCH_AWAY_TEAM_ID]].apply(
            lambda x: self.__calculate_stats_both_teams(x[definition.TOKEN_MATCH_POSS],
                                                        x[definition.TOKEN_MATCH_HOME_TEAM_ID],
                                                        x[definition.TOKEN_MATCH_AWAY_TEAM_ID]), axis=1,
            result_type="expand")

        return match_data

    def __fill_with_mean(self, feature, home_or_away, team_data: pd.DataFrame,
                         match_data: pd.DataFrame) -> pd.DataFrame:
        """
        The method will calculate the mean of row in the feature given until that row on theat feature
        :param feature: String. The feature in the df
        :param home_or_away: String. 'home' or 'away' for indicating if the feature is for the home team or away team
        :param team_data: DataFrame. The teams data from the dataset
        :param match_data: DataFrame. The matches data
        :return DataFrame
        """
        for team in team_data[definition.TOKEN_TEAM_ATTR_ID].tolist():
            team_matches = match_data.loc[match_data[f'{home_or_away}_{definition.TOKEN_TEAM_ATTR_ID}'] == team]

            if team_matches.shape[0] != 0:
                not_null = team_matches[~team_matches[feature].isna()]
                #  red card and yellow card feature can be 0 have meaning even with the 0 value. Do not delete rows
                #  with 0 values if its a yellow card or red card feature
                if feature in [definition.TOKEN_MATCH_AWAY_TEAM_YELLOWCARD, definition.TOKEN_MATCH_HOME_TEAM_YELLOWCARD,
                               definition.TOKEN_MATCH_HOME_TEAM_REDCARD, definition.TOKEN_MATCH_AWAY_TEAM_REDCARD]:
                    nulls = team_matches[team_matches[feature].isna()]

                else:
                    nulls = team_matches[team_matches[feature].isna() | team_matches[[feature]].eq(0)]

                matches_indexes = nulls.index.tolist()

                if not_null.shape[0] != 0 and nulls.shape[0] != 0:
                    avg = not_null[feature].mean()
                    if avg == 0:
                        #  If the average is 0 - all the rows have 0 value. Delete them
                        match_data.drop(matches_indexes, inplace=True)
                    else:
                        match_data.at[matches_indexes, feature] = avg

                else:
                    #  If all the values of the feature are null - delete it
                    match_data.drop(matches_indexes, inplace=True)

        return match_data

    def __calculate_stats_both_teams(self, xml_document, home_team, away_team, card_type='y'):
        if not xml_document:
            return None, None

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

    def __add_bets_ods_features(self, data:pd.DataFrame, matches_data: pd.DataFrame) -> pd.DataFrame:
        """
        The method will add the odds features. The odds features are the average of all the bets related
        features for each observation. The new features are calculated for both the home team and away team.
        :param data: DataFrame. The data
        :param matches_data: DataFrame. The matches data
        :return: DataFrame containing the odds features HomeTeamOdds, AwayTeamOdds
        """
        new_df = pd.DataFrame()

        for label, row in data.iterrows():
            away_team, home_team = row.at[definition.TOKEN_DS_HOME_TEAM_ID], row.at[definition.TOKEN_DS_AWAY_TEAM_ID]
            #  Get all the matches of the away_team and the home_team
            match = matches_data.loc[(matches_data[definition.TOKEN_MATCH_HOME_TEAM_ID] == home_team) &
                                         (matches_data[
                                              definition.TOKEN_MATCH_AWAY_TEAM_ID] == away_team)]

            if match.shape[0] == 0:
                remove_row(label, data)
                continue

            betting_ods = match.loc[:, self._bets_columns['all'][0]: self._bets_columns['all'][-1]]

            for bet, column in zip(['h', 'a'],
                                   [definition.TOKEN_DS_HOME_TEAM_ODDS, definition.TOKEN_DS_AWAY_TEAM_ODDS]):
                home_or_away_bets_odds = betting_ods.loc[:, self._bets_columns[bet]]
                #  For each match calculate the mean of all betting ods and that will be the match bet odd.
                betting_odd = home_or_away_bets_odds.fillna(0).values.mean()

                row[column] = betting_odd

            #  Create a new dataframe with the new Odds feature
            new_df = new_df.append(row)

        return new_df

    def __add_team_goals_avg(self, data: pd.DataFrame, team_data: pd.DataFrame) -> pd.DataFrame:
        """
        The method will calculate the mean of each team's goals in the data.
        The mean is calculated separately for the away team and home team.
        The two new features (HomeAvgGoals and AwayAvgGoals) are insereted into the dataframe
        :param data: DataFrame. The data
        :return: DataFrame. Containing the new goals avg features.
        """
        home_new_data = {definition.TOKEN_DS_HOME_TEAM_ID: [], definition.TOKEN_DS_HOME_TEAM_AVG_GOALS: []}
        away_new_data = {definition.TOKEN_DS_AWAY_TEAM_ID: [], definition.TOKEN_DS_AWAY_TEAM_AVG_GOALS: []}
        for label in team_data[definition.TOKEN_TEAM_ATTR_ID].tolist():
            home_team_games = data.loc[(data[definition.TOKEN_DS_HOME_TEAM_ID] == label)]
            home_team_goals_avg = home_team_games[definition.TOKEN_DS_HOME_TEAM_GOALS].mean()
            home_new_data[definition.TOKEN_DS_HOME_TEAM_ID] += [label]
            home_new_data[definition.TOKEN_DS_HOME_TEAM_AVG_GOALS] += [home_team_goals_avg]

            away_team_games = data.loc[(data[definition.TOKEN_DS_AWAY_TEAM_ID] == label)]
            away_team_goals_avg = away_team_games[definition.TOKEN_DS_AWAY_TEAM_GOALS].mean()
            away_new_data[definition.TOKEN_DS_AWAY_TEAM_ID] += [label]
            away_new_data[definition.TOKEN_DS_AWAY_TEAM_AVG_GOALS] += [away_team_goals_avg]

        new_home_df = pd.DataFrame(home_new_data)
        new_away_df = pd.DataFrame(away_new_data)

        data = pd.merge(data, new_home_df, how=definition.TOKEN_LEFT_JOIN,
                                 on=definition.TOKEN_DS_HOME_TEAM_ID)
        data = pd.merge(data, new_away_df, how=definition.TOKEN_LEFT_JOIN,
                                 on=definition.TOKEN_DS_AWAY_TEAM_ID)
        return data

    def __add_goals_difference(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        The method will be responsible for adding the goals difference between teams features.
        For each match the home team and away team will be taken into account in the goals difference
        calculation.
        :param data: DataFrame. The data
        :return: DataFrame containing the new features
        """
        copy_df = deepcopy(data)  # Create a copy of the Data to not change it
        new_data = {definition.TOKEN_DS_HOME_TEAM_ID: [], definition.TOKEN_DS_AWAY_TEAM_ID: [],
                    definition.TOKEN_DS_GOALDIFF: []}

        #  Iterate over the data set until there are no more matches
        while copy_df.shape[0] > 0:
            match = copy_df.iloc[0]  # Take the first match each iteration

            away_team, home_team = match.at[definition.TOKEN_DS_HOME_TEAM_ID], match.at[
                definition.TOKEN_DS_AWAY_TEAM_ID]

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
        data = pd.merge(data, new_data_df, how=definition.TOKEN_INNER_JOIN,
                                 on=[definition.TOKEN_DS_HOME_TEAM_ID, definition.TOKEN_DS_AWAY_TEAM_ID])
        return data




    def __remove_uneeded_features(self, data: pd.DataFrame):
        """
        The method will be responsible for dropping features that are not important for the predictions
        :return:
        """
        data.drop(columns=[definition.TOKEN_MATCH_ID], inplace=True)
        data.drop(columns=[definition.TOKEN_DS_HOME_TEAM_NAME], inplace=True)
        data.drop(columns=[definition.TOKEN_DS_AWAY_TEAM_NAME], inplace=True)
        data.drop(columns=[definition.TOKEN_DS_AWAY_TEAM_ID], inplace=True)
        data.drop(columns=[definition.TOKEN_DS_HOME_TEAM_ID], inplace=True)
        data.drop(columns=[definition.TOKEN_DS_AWAY_TEAM_GOALS], inplace=True)
        data.drop(columns=[definition.TOKEN_DS_HOME_TEAM_GOALS], inplace=True)

# p = FootballPreprocessesor("database.sqlite")
# data = p.preprocess()
# data.to_csv("trainset.csv", index=False)

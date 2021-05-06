import pandas as pd
import definition


def inner_join(left_df: pd.DataFrame, right_df: pd.DataFrame, on: str = None, features: list = None) -> pd.DataFrame:
    """
    The method will preform inner join the two tables based on selected features
    :return:
    """
    if features and on:
        to_join = right_df.loc[:, features[0]: features[-1]]
        ids = right_df.loc[:, definition.TOKEN_MATCH_ID]
        to_join = pd.concat([to_join, ids], axis=1)
        return pd.merge(left_df, to_join, how=definition.TOKEN_INNER_JOIN, on=on)

    else:
        return pd.merge(left_df, right_df, how=definition.TOKEN_INNER_JOIN)

def unique_value_exctraction(df: pd.DataFrame, columns: list) -> set:
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

def dimentions_reduction(df: pd.DataFrame, features: list):
    """
    The method will be responsible for deleting unwanted columns (feature) from the match data.
    :return: DataFrame
    """
    for col in features:
        df.drop(col, axis=1, inplace=True)

    return df


def remove_row(row_index, df:pd.DataFrame):
        """
        The method will delete the row from the Data
        :param row_index: int. The index of the row in the dataframe
        :param df: DataFrame. The dataframe to the delete the row from
        :return: DataFrame without the row
        """
        return df[df != row_index]
"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.18.0
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(
    data: pd.DataFrame, user_column: str, item_column: str, rating_column: str
) -> pd.DataFrame:
    """This node performs the preprocessing of the data.

    Args:
        data (pd.DataFrame): Input data.
        user_column (str): User column.
        item_column (str): Item column.
        rating_column (str): Rating column.

    Returns:
        pd.DataFrame: Preprocessed data
    """
    data[rating_column] = data[rating_column].fillna(0)
    return data


def split_train_test(
    data: pd.DataFrame, user_column: str, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This node split data in train/test datasets

    Args:
        data (pd.DataFrame): Input data.
        user_column (str): User column in data.
        test_size (float): Test size.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
    """
    train, test = train_test_split(
        data, test_size=test_size, stratify=data[user_column]
    )
    return train, test

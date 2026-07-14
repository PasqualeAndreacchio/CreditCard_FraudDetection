from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


class Preprocessing:
    """
    This class is used to preprocess the dataset.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df


    def _remove_duplicates(self):
        """
        This function analyzes the dataset checking for duplicates and removing them.
        It also check for NULL values and removes them.
        Returns:
            pd.DataFrame: The processed dataset.
        """

        # Remove duplicates, null values and reset index
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self.df


    def _split_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        This function splits the dataset into training and test sets using stratified split to make
        sure that fraud and non-fraud transactions are represented proportionally in both sets.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - X_train, X_test, y_train, y_test
        """

        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        return X_train, X_test, y_train, y_test

        
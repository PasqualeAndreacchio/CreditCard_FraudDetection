from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler


class Preprocessing:
    """
    Preprocessing pipeline for the Credit Card Fraud Detection dataset.
 
    On instantiation, the dataset is cleaned by removing duplicate rows and
    rows with missing values. From there, the class exposes methods to obtain
    a train/test split ready for modeling, in three variants:
 
    - get_dataset(): stratified train/test split with 'Time' and 'Amount'
      scaled via RobustScaler (fit on the training set only, to avoid data
      leakage into the test set).
    - get_smote_dataset(): same as get_dataset(), but with SMOTE oversampling
      applied to the training set only, to address the strong class
      imbalance between fraud and non-fraud transactions.
    - get_class_weights(): computes per-class weights (inversely
      proportional to class frequency) to use in the training loop instead
      of oversampling, as an alternative strategy for handling imbalance.
 
    Splits are cached per (test_size, random_state) pair, so calling
    multiple methods with the same parameters reuses the same split and
    scaling instead of recomputing them.
    """


    def __init__(self, df: pd.DataFrame):
        
        # Check of dataset integrity
        required_cols = {'Class', 'Time', 'Amount'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.df = self._remove_duplicates(df)
        # Cache of already-computed splits, keyed by (test_size, random_state)
        self._split_cache: Dict[Tuple[float, int], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = {}


    def get_smote_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        This function applies SMOTE to the training set to balance the dataset.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - X_train_smote, X_test, y_train_smote, y_test
        """

        # Apply the split with the internal method and SMOTE only the training dataset!
        X_train, X_test, y_train, y_test = self._split_dataset(test_size=test_size, random_state=random_state)
        smote = SMOTE(random_state=random_state)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        return X_train_smote, X_test, y_train_smote, y_test
    

    def get_class_weights(self, test_size: float = 0.2, random_state: int = 42, verbose: bool = False) -> Dict[int, float]:
        """
        This function splits the dataset into training and test sets and calculates the class weights.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
            - verbose (bool): An optional parameter, if "True" the function will print the calculated weights
        Returns:
            - class_weights (dict): A dictionary containing the classes and the associated weight.
        """

        # Apply the split with the internal method
        _, _, y_train, _ = self._split_dataset(test_size=test_size, random_state=random_state)
        return self._calculate_class_weights(y_train, verbose=verbose)


    def get_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Returns the preprocessed dataset split into training and test sets, with features already scaled.
        Duplicates and NULL values are removed automatically at class instantiation.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - X_train, X_test, y_train, y_test
        """

        return self._split_dataset(test_size=test_size, random_state=random_state)
    

    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of df with duplicate rows and rows with NULL values
        removed, and the index reset.
        """
        return df.drop_duplicates().dropna().reset_index(drop=True)


    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scales the 'Time' and 'Amount' features using RobustScaler.
        The scaler is fitted strictly on the training set to prevent data leakage.            
        Args:
            X_train (pd.DataFrame): The training set.
            X_test (pd.DataFrame): The testing set.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The scaled training and testing sets.
        """

        # Create copies of the dataframe to avoid pandas warning
        X_train = X_train.copy()
        X_test = X_test.copy()

        # Create the scaler
        rob_scaler_amount = RobustScaler()
        rob_scaler_time = RobustScaler()

        # Fit and transform the training data
        X_train['Amount'] = rob_scaler_amount.fit_transform(X_train[['Amount']])
        X_train['Time'] = rob_scaler_time.fit_transform(X_train[['Time']])

        # Transform the testing data using the fitted scalers
        X_test['Amount'] = rob_scaler_amount.transform(X_test[['Amount']])
        X_test['Time'] = rob_scaler_time.transform(X_test[['Time']])

        return X_train, X_test


    def _split_dataset(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        This function splits the dataset into training and test sets using stratified split to make
        sure that fraud and non-fraud transactions are represented proportionally in both sets.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - X_train, X_test, y_train, y_test with the appropriate scalings.

        Note:
            The result is cached per (test_size, random_state) pair, so repeated calls
            (e.g. get_dataset + get_class_weights with the same parameters) don't redo
            the split and scaling from scratch. Copies are still returned, so that any
            in-place modification by the caller doesn't corrupt the internal cache.
        """

        cache_key = (test_size, random_state)

        if cache_key not in self._split_cache:
            X = self.df.drop('Class', axis=1)
            y = self.df['Class']

            # Split and scale the datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            X_train, X_test = self._scale_features(X_train, X_test)

            self._split_cache[cache_key] = (X_train, X_test, y_train, y_test)

        X_train, X_test, y_train, y_test = self._split_cache[cache_key]

        # Return copies to protect the cache from external in-place modifications
        return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

    
    @staticmethod
    def _calculate_class_weights(y_train: pd.Series, verbose: bool = False) -> Dict[int, float]:
        """
        This function calculates the class weights to be used in the training loop, based on the imbalance
        of the classes in the training set.
        The formula used to calculate the weights for each class is:
            weight = (Total Number of transactions) / (2 * Number of occurrences for that class)
        Same formula implemented in scikit-learn, "2" is the number of classes.
        Args:
            - y_train (pd.Series): The training set.
            - verbose (bool): An optional parameter, if "True" the function will print the calculated weights
        Returns:
            - class_weights (dict): A dictionary containing the classes and the associated weight.
        """

        # Get the count of the transactions (class 1) for each class
        total_transactions = len(y_train)
        fraud_count = int(y_train.sum())
        non_fraud_count = total_transactions - fraud_count

        # Calculate the weights
        fraud_weight = total_transactions / (2. * fraud_count)
        non_fraud_weight = total_transactions / (2. * non_fraud_count)

        # Create a dictionary with the class weights
        class_weights = {
            0: non_fraud_weight,
            1: fraud_weight
        }

        if verbose:
            print(f"Class weights: {class_weights}")

        return class_weights
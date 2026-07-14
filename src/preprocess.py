from typing import Tuple, Dict
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler


class Preprocessing:
    """
    This class is used to preprocess the dataset.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._remove_duplicates()


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
        """

        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        # Split and scale the datasets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = self._scale_features(X_train, X_test)

        return X_train, X_test, y_train, y_test


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


    def get_class_weights(self, test_size: float = 0.2, random_state: int = 42) -> Dict[int, float]:
        """
        This function splits the dataset into training and test sets and calculates the class weights.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - class_weights (dict): A dictionary containing the classes and the associated weight.
        """

        # Apply the split with the internal method
        _, _, y_train, _ = self._split_dataset(test_size=test_size, random_state=random_state)
        return self._calculate_class_weights(y_train)


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
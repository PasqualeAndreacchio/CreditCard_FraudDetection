from typing import Tuple, Dict, Optional, overload
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn.functional as F


class Preprocessing:
    """
    Preprocessing pipeline for the Credit Card Fraud Detection dataset.

    On instantiation, the dataset is cleaned by removing duplicate rows and
    rows with missing values with the possibility of dropping the "Time" column.
    From there, the class exposes methods to obtain a train/test split 
    (optionally train/val/test) ready for modeling, in three variants:

    - get_dataset(): stratified split with 'Time' and 'Amount' scaled via
      RobustScaler (fit on the training set only, to avoid data leakage into
      val/test). By default this returns a train/test split
      (X_train, X_test, y_train, y_test). Passing val_size adds a third,
      held-out validation set, returning (X_train, X_val, X_test,
      y_train, y_val, y_test) instead. Stratification is applied at both
      split points, so class proportions are preserved across all sets.

    - get_smote_dataset(): same splitting/scaling behavior as get_dataset()
      (including the optional val_size), but with SMOTE oversampling applied
      to the training set only, to address the strong class imbalance
      between fraud and non-fraud transactions. Validation and test sets are
      left untouched, since they must reflect the real class distribution.

    - get_class_weights(): alternative to SMOTE. Computes per-class weights
      (inversely proportional to class frequency) from the training labels
      only, to use in the training loop instead of oversampling. Also
      respects val_size when splitting, though the validation set itself
      plays no role in the weight calculation.

    Splits are cached per (test_size, val_size, random_state) pair, so
    calling multiple methods with the same parameters reuses the same split
    and scaling instead of recomputing them. Callers always receive copies,
    so in-place modification of the returned DataFrames/Series never
    corrupts the cache.
    """

    def __init__(self, df: pd.DataFrame, drop_time: bool = False):
        """
        Constructor of the Preprocessing class.
        Args:
            - df (pd.DataFrame): The dataset to be processed.
            - drop_time (bool): Whether to drop the 'Time' column (default=False).
        """
        # Check of dataset integrity
        required_cols = {'Class', 'Time', 'Amount'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        # Remove time (if specified) and duplicates/NaN
        if drop_time: df = df.drop('Time', axis=1)
        self.df = self._remove_duplicates(df)
        
        # Cache of already-computed splits, keyed by (test_size, val_size, random_state)
        self._split_cache: Dict[Tuple[float, Optional[float], int], Tuple] = {}


    @overload
    def get_dataset(
        self, test_size: float = 0.2, val_size: None = None, random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def get_dataset(
        self, test_size: float, val_size: float, random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_dataset(
        self, 
        test_size: float = 0.2, 
        val_size: Optional[float] = None, 
        random_state: int = 42
    ) -> Tuple:
        """
        Returns the preprocessed dataset split into training and test sets, with features already scaled.
        Duplicates and NULL values are removed automatically at class instantiation.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - val_size (float, optional): The proportion of the dataset to include in the validation split.
                                          If None (default), only a train/test split is retreived.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - If val_size is None: X_train, X_test, y_train, y_test
            - If val_size is given: X_train, X_val, X_test, y_train, y_val, y_test
        """

        split =  self._split_dataset(val_size=val_size, test_size=test_size, random_state=random_state)
        X_train, X_test, y_train, y_test = split

        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

        # One-hot configuration for softmax output
        y_train_tensor_onehot = F.one_hot(y_train_tensor, 2).float()
        y_test_tensor_onehot = F.one_hot(y_test_tensor, 2).float()

        return X_train_tensor, X_test_tensor, y_train_tensor_onehot, y_test_tensor_onehot
    

    @overload
    def get_smote_dataset(
        self, test_size: float = 0.2, val_size: None = None, random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def get_smote_dataset(
        self, test_size: float, val_size: float, random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def get_smote_dataset(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        random_state: int = 42,
    ) -> Tuple:
        """
        This function applies SMOTE to the training set to balance the dataset.
        val and test sets are left untouched (they must reflect the real class distribution).
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - val_size (float, optional): The proportion of the dataset to include in the validation split.
                                          If None (default), only a train/test split is returned.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - If val_size is None: X_train_smote, X_test, y_train_smote, y_test
            - If val_size is given: X_train_smote, X_val, X_test, y_train_smote, y_val, y_test
        """
        split = self._split_dataset(val_size=val_size, test_size=test_size, random_state=random_state)
        smote = SMOTE(random_state=random_state)

        if not val_size:
            X_train, X_test, y_train, y_test = split
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
            
            # Add .to_numpy() to the Pandas objects
            X_train_smote_tensor = torch.tensor(X_train_smote.to_numpy(), dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            y_train_smote_tensor = torch.tensor(y_train_smote.to_numpy(), dtype=torch.long)
            y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
            
            # One-hot configuration for softmax output
            y_train_smote_tensor_onehot = F.one_hot(y_train_smote_tensor, 2).float()
            y_test_tensor_onehot = F.one_hot(y_test_tensor, 2).float()

            return X_train_smote_tensor, X_test_tensor, y_train_smote_tensor_onehot, y_test_tensor_onehot
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = split
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

            # Add .to_numpy() to the Pandas objects
            X_train_smote_tensor = torch.tensor(X_train_smote.to_numpy(), dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
            y_train_smote_tensor = torch.tensor(y_train_smote.to_numpy(), dtype=torch.long)
            y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
            y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.long)

            # One-hot configuration for softmax output
            y_train_smote_tensor_onehot = F.one_hot(y_train_smote_tensor, 2).float()
            y_test_tensor_onehot = F.one_hot(y_test_tensor, 2).float()
            y_val_tensor_onehot = F.one_hot(y_val_tensor, 2).float()

            return X_train_smote_tensor, X_test_tensor, X_val_tensor, y_train_smote_tensor_onehot, y_test_tensor_onehot, y_val_tensor_onehot

    def get_class_weights(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        random_state: int = 42,
        verbose: bool = False,
    ) -> Dict[int, float]:
        """
        Splits the dataset (train/test or train/val/test) and calculates
        class weights from the training labels only.
        Args:
            - test_size (float): The proportion of the dataset to include in the test split.
            - val_size (float, optional): The proportion of the dataset to include in the validation split.
            - random_state (int): The seed used by the random number generator.
            - verbose (bool): If True, print the calculated weights.
        Returns:
            - class_weights (dict): A dictionary containing the classes and the associated weight.
        Note:
            Class weights and SMOTE are alternative strategies for handling
            imbalance, not meant to be combined. This method always computes
            weights from the original (non-resampled) training labels for the
            given (test_size, val_size, random_state) — even if get_smote_dataset()
            was previously called with the same parameters. It does NOT reflect
            the balanced distribution produced by SMOTE.
        """
        split = self._split_dataset(test_size=test_size, val_size=val_size, random_state=random_state)
        # y_train sits at index 2 for a 4-tuple split, index 3 for a 6-tuple split
        y_train = split[2] if not val_size else split[3]
        return self._calculate_class_weights(y_train, verbose=verbose)    


    @staticmethod
    def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of df with duplicate rows and rows with NULL values
        removed, and the index reset.
        """
        return df.drop_duplicates().dropna().reset_index(drop=True)

    
    def _scale_features(self, X_train: pd.DataFrame, *X_others: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Scales 'Amount' feature (and 'Time' if present) using RobustScaler.
        The scaler is fitted strictly on the training set to prevent data
        leakage, then applied to any number of other sets (val, test, ...).
        Args:
            X_train (pd.DataFrame): The training set.
            *X_others (pd.DataFrame): Any number of additional sets (e.g. X_val, X_test)
              to transform with the scaler fitted on X_train.
        Returns:
            Tuple[pd.DataFrame, ...]: (X_train, *X_others), all scaled, in the same order given.
        """
        # Make copies to prevent pandas warnings about modifying copies vs views
        X_train = X_train.copy()
        others = [X.copy() for X in X_others]

        rob_scaler_amount = RobustScaler()
        # Fit and transform on the training set only, then transform the other datasets
        X_train['Amount'] = rob_scaler_amount.fit_transform(X_train[['Amount']])
        for X in others:
            X['Amount'] = rob_scaler_amount.transform(X[['Amount']])
        
        # If "Time" column is present, scale it too (always)
        if 'Time' in X_train.columns:
            rob_scaler_time = RobustScaler()
            X_train['Time'] = rob_scaler_time.fit_transform(X_train[['Time']])
            for X in others:
                if 'Time' in X.columns:
                    X['Time'] = rob_scaler_time.transform(X[['Time']])
        return (X_train, *others)


    @overload
    def _split_dataset(
        self, test_size: float = 0.2, val_size: None = None, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...

    @overload
    def _split_dataset(
        self, test_size: float, val_size: float, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: ...

    def _split_dataset(
        self, 
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        random_state: int = 42
    ) -> Tuple:
        """
        This function splits the dataset into training and test sets using stratified split to make
        sure that fraud and non-fraud transactions are represented proportionally in both sets.
        Args:
            - val_size (float): The proportion of the dataset to include in the validation split.
                                If None (default), only a train/test split is retreived.
            - test_size (float): The proportion of the dataset to include in the test split.
            - random_state (int): The seed used by the random number generator.
        Returns:
            - If val_size is None: X_train, X_test, y_train, y_test with the appropriate scalings.
            - If val_size is given: X_train, X_val, X_test, y_train, y_val, y_test with the appropriate scalings.

        Note:
            The result is cached per (test_size, val_size, random_state) pair, so repeated calls
            (e.g. get_dataset + get_class_weights with the same parameters) don't redo
            the split and scaling from scratch. Copies are still returned, so that any
            in-place modification by the caller doesn't corrupt the internal cache.
        """

        # Treating val_size = 0 as val_size = None (no validation set)
        if val_size == 0: val_size = None

        # Sanity checks: test_size and val_size must be within (0, 1)
        # and their sum (including val if present) must be less than 1.
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be in (0, 1), got {test_size}.")
        if val_size is not None:
            if not (0 < val_size < 1):
                raise ValueError(f"val_size must be in (0, 1), got {val_size}.")
            if val_size + test_size >= 1:
                raise ValueError(
                    f"val_size + test_size must be < 1, got {val_size + test_size:.4f}."
                )

        cache_key = (test_size, val_size, random_state)

        if cache_key not in self._split_cache:
            X = self.df.drop('Class', axis=1)
            y = self.df['Class']

            # First, split the dataset into trainval + test sets
            X_trainval, X_test, y_trainval, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            if val_size is None:
                X_train, X_test = self._scale_features(X_trainval, X_test)
                self._split_cache[cache_key] = (X_train, X_test, y_trainval, y_test)
            else:
                # Since val_size is relative to the whole dataset, we have to convert it to be 
                # relative to the training set, which has a size of (1 - test_size)
                relative_val = val_size / (1 - test_size)

                # Split the training set into training and validation sets
                X_train, X_val, y_train, y_val = train_test_split(
                    X_trainval, y_trainval, test_size=relative_val, random_state=random_state, stratify=y_trainval
                )

                # Scale the features and save in the cache 
                X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)
                self._split_cache[cache_key] = (X_train, X_val, X_test, y_train, y_val, y_test)

        # Return copies to protect the cache from external in-place modifications
        return tuple(item.copy() for item in self._split_cache[cache_key])

    
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

        # Sanity checks
        if fraud_count == 0:
            raise ValueError("There are no fraud transactions in the training set.")
        if non_fraud_count == 0:
            raise ValueError("There are no non-fraud transactions in the training set.")

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
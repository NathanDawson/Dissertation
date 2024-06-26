import random
import pandas as pd
import numpy as np  # noqa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from skrebate import ReliefF
from boruta import BorutaPy
from powershap import PowerShap


class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """
        A custom transformer that removes highly correlated features from a dataset.

        This transformer identifies pairs of features that have a correlation higher than
        a specified threshold and removes one feature from each pair to reduce feature redundancy.
        The choice of which feature to remove from each pair is made randomly.
    """

    def __init__(self, threshold=0.80):
        self.threshold = threshold
        self.features_to_drop = []

    def fit(self, X, y=None):
        """
            Fits the transformer to the data by identifying highly correlated feature pairs.

            Parameters:
                - X (pandas DataFrame): The input features to check for correlation.
                - y (ignored): Not used, present here for consistency with the scikit-learn transformer interface.

            Returns:
                - self
        """
        # Calculate the correlation matrix
        corr_matrix = pd.DataFrame(X).corr().abs()
        to_drop = set()

        # Identify pairs of highly correlated features
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= self.threshold:
                    to_drop.add(corr_matrix.columns[random.choice([i, j])])

        self.features_to_drop = list(to_drop)
        return self

    def transform(self, X, y=None):
        """
            Transforms the input dataset by removing the features identified during fitting.

            Parameters:
                - X (pandas DataFrame): The input features to transform.
                - y (ignored): Not used, present here for consistency with the scikit-learn transformer interface.

            Returns:
                - X_transformed: The transformed dataset with highly correlated features removed.
        """
        # Remove the features identified during the fitting process
        X_transformed = pd.DataFrame(X).drop(columns=self.features_to_drop)
        return X_transformed


class ValidateData(BaseEstimator, TransformerMixin):
    """
        A custom transformer for validating data within a pipeline.

        This transformer performs two main validation checks on the data:
            1. Ensures that there are no missing values.
            2. Validates that all categorical features are binary (i.e., have only two unique values).
    """

    def fit(self, X, y=None):
        """
            Placeholder method for compatibility with scikit-learn's transformer interface.

            Parameters:
                - X (pandas DataFrame): The input data to validate.
                - y (ignored): Not used, present here for consistency with the scikit-learn transformer interface.

            Returns:
                - self
        """
        return self

    def transform(self, X, y=None):
        """
            Performs validation checks on the input data and raises a ValueError if any checks fail.

            Parameters:
                - X (pandas DataFrame): The input data to validate.
                - y (ignored): Not used, present here for consistency with the scikit-learn transformer interface.

            Returns:
                - X_transformed (pandas DataFrame): The validated data, returned unchanged if no issues are detected.

            Raises:
                - ValueError: If missing data is detected after imputation or if any categorical feature
                              is found to have more than two unique values (i.e., is not binary).
        """
        X_transformed = pd.DataFrame(X)

        # Check for missing data after imputation
        if X_transformed.isnull().sum().any():
            raise ValueError("Missing data detected.")

        # Check that all categorical features are binary
        for col in X_transformed.select_dtypes(include=['object', 'category']).columns:
            if X_transformed[col].nunique() > 2:
                raise ValueError(f"Non-binary categorical data detected in column: {col}")

        return X_transformed


class FeatureNameTracker(BaseEstimator, TransformerMixin):
    """
        Tracks the names of features selected by a feature selection algorithm.

        This class wraps around a feature selector that implements the `fit` and `transform` methods and
        either provides a `support_` attribute or a `get_support()` method indicating which features are selected.
        It captures the names of the selected features during fitting, allowing for easy retrieval of feature names
        after transformations or selections have been applied.
    """

    def __init__(self, feature_selector, feature_names=None):
        self.feature_selector = feature_selector
        self.feature_names = feature_names
        self.selected_feature_names_ = []

    def fit(self, X, y=None):
        self.feature_selector.fit(X, y)
        if hasattr(self.feature_selector, 'support_'):
            support_mask = self.feature_selector.support_
        elif hasattr(self.feature_selector, 'get_support'):
            support_mask = self.feature_selector.get_support()
        else:
            raise ValueError("The feature selector does not have a 'support_' attribute or a 'get_support()' method.")

        if self.feature_names is None:
            self.feature_names = X.columns if hasattr(X, 'columns') else [str(i) for i in range(X.shape[1])]

        self.selected_feature_names_ = [self.feature_names[i] for i, selected in enumerate(support_mask) if selected]
        return self

    def transform(self, X):
        return self.feature_selector.transform(X)


class ReliefFeatureSelector(BaseEstimator, TransformerMixin):
    """
        Custom transformer that wraps around the ReliefF algorithm to perform feature selection.

        Parameters:
            - n_features_to_select (int): The number of features to select.
            - n_neighbours (int): The number of neighbours to consider for each sample when computing feature importance.
    """

    def __init__(self, n_features_to_select=50, n_neighbors=100):
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.selected_features_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        """
            Fits the ReliefF algorithm on the dataset and selects the top `n_features_to_select` based on their importance scores.

            Parameters:
                - X (DataFrame or numpy array): Feature matrix.
                - y (Series or numpy array): Target variable.

            Returns:
                - self: object
                Fitted instance of the transformer.
        """
        # Ensure X and y are in NumPy array format
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y

        relief = ReliefF(n_neighbors=self.n_neighbors)
        relief.fit(X_np, y_np)

        # Store feature importances
        self.feature_importances_ = relief.feature_importances_

        # Identify indices of top features based on their importances
        self.selected_features_ = sorted(range(len(self.feature_importances_)),
                                         key=lambda i: self.feature_importances_[i],
                                         reverse=True)[:self.n_features_to_select]
        return self

    def transform(self, X):
        """
            Transforms the dataset to include only the selected features.

            Parameters:
                - X (DataFrame or numpy array): The feature matrix to be transformed.

            Returns:
                - X_transformed (DataFrame or numpy array): The transformed feature matrix with only selected features.
        """
        # Ensure X is in NumPy array format
        X_np = X.to_numpy() if hasattr(X, 'to_numpy') else X
        was_dataframe = hasattr(X, 'to_numpy')

        # Select features based on indices identified in fit
        if self.selected_features_ is not None:
            X_transformed = X_np[:, self.selected_features_]
        else:
            raise ValueError("The fit method has not been called or no features were selected.")

        # If the original X was a DataFrame, convert back to DataFrame with selected column names
        if was_dataframe and self.selected_features_ is not None:
            column_names = [X.columns[idx] for idx in self.selected_features_]
            X_transformed = pd.DataFrame(X_transformed, columns=column_names, index=X.index)

        return X_transformed

    def get_support(self):
        """
            Returns a boolean mask indicating which features are selected for retention.

            Returns:
                - support_mask (numpy array): Boolean array of shape [n_features] indicating selected features.
        """
        # Generate a support mask to indicate selected features
        if self.feature_importances_ is None:
            raise ValueError("The fit method has not been called.")

        support_mask = [False] * len(self.feature_importances_)
        for idx in self.selected_features_:
            support_mask[idx] = True
        return support_mask


class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, n_estimators='auto', perc=100, max_iter=100,
                 early_stopping=False, n_iter_no_change=20, random_state=42):
        """
            A custom wrapper for the Boruta feature selection method.

            Parameters:
                - model: The machine learning model/estimator to use for feature selection.
                - n_estimators: Number of estimators to use in Boruta, 'auto' lets Boruta decide.
                - perc: Percentile of importance that features must reach to be considered important.
                - max_iter: Maximum number of iterations to perform in Boruta.
                - early_stopping: Whether to stop early if the feature selection does not improve.
                - n_iter_no_change: Number of iterations with no change in feature rankings to trigger early stopping.
                - random_state: Seed for random number generator.
        """
        self.model = model
        self.n_estimators = n_estimators
        self.perc = perc
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.support_ = None  # To store indices of selected features

    def fit(self, X, y):
        """
            Fit the Boruta feature selector.

            Parameters:
                - X: Training vector, where n_samples is the number of samples and n_features is the number of features.
                - y: Target relative to X for classification.
        """
        # Check if X is a pandas DataFrame or a numpy array
        X_fit = X.values if hasattr(X, 'values') else X

        # Initialise Boruta
        boruta_selector = BorutaPy(
            estimator=self.model,
            n_estimators=self.n_estimators,
            perc=self.perc,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        # Fit Boruta
        boruta_selector.fit(X_fit, y)
        self.support_ = boruta_selector.support_

        return self

    def transform(self, X):
        """
            Reduce X to the selected features.

            Parameters:
                - X: The input samples.

            Returns:
                - The input samples with only the selected features.
        """
        if self.support_ is None:
            raise ValueError("The fit method has not been called.")
        # Ensure consistency with input type (DataFrame or numpy array)
        if hasattr(X, 'iloc'):
            # X is a pandas DataFrame
            return X.iloc[:, self.support_]
        else:
            # X is a numpy array
            return X[:, self.support_]

    def get_support(self):
        """
            Get a mask, or integer index, of the features selected

            Returns:
                - An array of boolean values denoting whether each feature is selected.
        """
        if self.support_ is None:
            raise ValueError("The fit method has not been called.")
        return self.support_


class PowerShapFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, automatic=True, power_alpha=0.01, force_convergence=False,
                 limit_convergence_its=0, stratify=False, cv=None):
        """
            A custom wrapper for the PowerShap feature selection method.

            Parameters:
                - model(estimator object): The machine learning model/estimator to use for feature selection.

                - automatic (bool, default=True): Specifies whether PowerShap should automatically calculate the
                required number of iterations for convergence. If True, PowerShap starts with a predefined number of
                iterations to estimate the necessary iterations for achieving stable feature importance scores.

                - power_alpha (float, default=0.01): The significance level used for the hypothesis testing to determine
                 the importance of features. It represents the probability of incorrectly rejecting the null hypothesis
                 for a feature that it is not important. Lower values of `power_alpha` lead to stricter selection
                 criteria.

                - force_convergence (bool, default=False): If True, PowerShap will iteratively exclude the features
                identified as important and rerun the selection process. This can help in identifying robust sets of
                features but may increase computation time.

                - limit_convergence_its (int, default=0): The maximum number of iterations allowed for
                `force_convergence`. Setting this parameter helps in controlling the runtime of the feature selection
                process, especially in high-dimensional datasets.

                - stratify (bool, default=False): Whether to stratify the splits during the feature selection process.
                Stratification ensures that each split contains approximately the same percentage of samples of each
                target class as the complete set.

                - cv (cross-validation generator, default=None): The cross-validator that determines the
                cross-validation splitting strategy. If None, PowerShap will use its default strategy. It can be any
                scikit-learn cross-validator that yields train/test indices.
        """
        self.model = model
        self.automatic = automatic
        self.power_alpha = power_alpha
        self.force_convergence = force_convergence
        self.limit_convergence_its = limit_convergence_its
        self.stratify = stratify
        self.cv = cv

    def fit(self, X, y):
        """
            Fits the PowerShap feature selector to the data.

            Parameters:
                - X: Feature matrix.
                - y: Target vector.
        """
        # Initialise PowerShap with the provided model and parameters
        self.powershap = PowerShap(
            model=self.model,
            automatic=self.automatic,
            power_alpha=self.power_alpha,
            force_convergence=self.force_convergence,
            limit_convergence_its=self.limit_convergence_its,
            stratify=self.stratify,
            cv=self.cv
        )

        # Fit PowerShap to the data
        self.powershap.fit(X, y)

        return self

    def transform(self, X):
        """
            Transforms the feature matrix to include only the selected features.

            Parameters:
                - X: Feature matrix to transform.

            Returns:
                - Transformed feature matrix with only the selected features.
        """
        # Ensure PowerShap is fitted and has determined the features to keep
        check_is_fitted(self, 'powershap')
        # Use PowerShap's internal method to apply the feature selection
        return self.powershap.transform(X)

    def get_support(self):
        """
            Returns a mask indicating which features are selected, leveraging PowerShap's built-in method.

            Returns:
                - support_mask: A boolean array indicating the selected features.
        """
        if hasattr(self, 'powershap') and hasattr(self.powershap, '_get_support_mask'):
            # Ensure PowerShap is fitted and can provide the support mask
            check_is_fitted(self.powershap, '_get_support_mask')
            # Directly use the get_support_mask method from PowerShap
            return self.powershap._get_support_mask()
        else:
            raise ValueError("The fit method has not been called or PowerShap instance is not properly initialised.")

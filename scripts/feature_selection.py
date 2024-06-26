from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector, RFECV
from transformers import ReliefFeatureSelector, BorutaFeatureSelector, PowerShapFeatureSelector


def select_feature_selection_method():
    """
     Prompts the user to select a feature selection method from the options provided.

     Returns:
         selection_method (int): The integer corresponding to the chosen feature selection method.
            - 1 for Forward Feature Selection
            - 2 for Recursive Feature Elimination with Cross-Validation (RFECV)
            - 3 for Relief-based Feature Selection
            - 4 for Boruta Feature Selection
            - 5 for PowerShap Feature Selection

    Raises:
        ValueError: If the input is not an integer or if it's an integer that doesn't correspond
                    to the available options (1, 2, 3, 4, or 5).
    """
    print("Please choose a feature selection method:")
    print("1: Forward Feature Selection")
    print("2: Recursive Feature Elimination with Cross-Validation (RFECV)")
    print("3: Relief-based Feature Selection")
    print("4: Boruta Feature Selection")
    print("5: PowerShap Feature Selection")

    while True:
        try:
            selection_method = int(input("Enter 1, 2, 3, 4, or 5: "))
            if selection_method in [1, 2, 3, 4, 5]:
                return selection_method
            else:
                print("Invalid input. Please enter 1, 2, 3, 4, or 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_feature_selector(model, method):
    """
    Returns the appropriate feature selector based on user input.

    Parameters:
    - model: The machine learning model/estimator to use for feature selection.
    - method: The method of feature selection chosen by the user.

    Returns:
    - A configured feature selector instance.
    """
    if method == 1:
        # Forward Feature Selection
        return SequentialFeatureSelector(
            model,
            direction="forward",
            scoring="f1",
            n_features_to_select=20,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=8
        )
    elif method == 2:
        # RFECV
        return RFECV(
            estimator=model,
            step=5,
            min_features_to_select=15,
            scoring='f1',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            n_jobs=8
        )
    elif method == 3:
        # ReliefFeatureSelector (custom wrapper for ReliefF)
        return ReliefFeatureSelector(
            n_features_to_select=20,
            n_neighbors=100,
        )
    elif method == 4:
        # BorutaFeatureSelector (custom wrapper for BorutaPy)
        return BorutaFeatureSelector(
            model=model,
            n_estimators='auto',
            perc=80,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=42
        )
    elif method == 5:
        # PowerShapFeatureSelector (custom wrapper for PowerShap)
        return PowerShapFeatureSelector(
            model=model,
            automatic=True,
            power_alpha=0.2,
            force_convergence=True,
            limit_convergence_its=5,
            stratify=True,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        )
    else:
        raise ValueError("Invalid selection method.")

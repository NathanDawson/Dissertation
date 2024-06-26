from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from feature_selection import get_feature_selector
from transformers import RemoveCorrelatedFeatures, ValidateData, FeatureNameTracker


def get_pipeline(data, selection_method):
    """
    Constructs and returns a machine learning pipeline based on the selected feature selection method.

    Parameters:
        - selection_method (int): An integer indicating the chosen method for feature selection.
                                - 1 corresponds to Forward Feature Selection.
                                - 2 corresponds to Recursive Feature Elimination with Cross-Validation (RFECV).
                                - 3 corresponds to Relief-Based Feature Selection.
                                - 4 corresponds to Boruta Feature Selection.
                                - 5 corresponds to PowerShap Feature Selection.

    Returns:
        - pipeline (Pipeline): A scikit-learn Pipeline object configured with the appropriate steps for
                                preprocessing, feature selection, and the RandomForestClassifier.
    """
    feature_names = data.columns.tolist()

    imputer = IterativeImputer(max_iter=10, random_state=42)
    remove_correlated = RemoveCorrelatedFeatures()
    validate_data = ValidateData()
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    feature_selector = get_feature_selector(rf, selection_method)
    feature_selector_with_names = FeatureNameTracker(feature_selector=feature_selector, feature_names=feature_names)

    steps = [imputer, remove_correlated, validate_data, feature_selector_with_names, rf]

    pipeline = make_pipeline(*steps)
    return pipeline

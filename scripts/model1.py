import pandas as pd
import numpy as np # noqa
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pipeline_setup import get_pipeline
from feature_selection import select_feature_selection_method
from results_handler import store_results

data = pd.read_pickle('data/final/merged_data.pkl.bz2', compression='bz2')

X = data.drop(['S'], axis=1)
y = data['S']

# Obtain selection method from user
selection_method = select_feature_selection_method()

# Generate the pipeline based on the user's choice
pipeline = get_pipeline(data, selection_method)

param_grid = {
            'randomforestclassifier__n_estimators': [100, 300, 500],
            'randomforestclassifier__max_depth': [None, 10, 15],
            'randomforestclassifier__min_samples_split': [2, 4, 6]
}

outer_cv = StratifiedKFold(n_splits=5)
inner_cv = StratifiedKFold(n_splits=5)


collect_results = {}
fold_counter = 1

selection_techniques = {
    1: 'Forward Feature Selection',
    2: 'RFECV',
    3: 'Relief',
    4: 'Boruta',
    5: 'PowerShap'
}

# Outer CV loop
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Apply undersampling to the training data
    undersampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
    X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

    # Conditional addition of n_jobs based on the selection method
    if selection_method in [3, 4, 5]:
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='f1', n_jobs=8)
    else:
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='f1')

    grid_search.fit(X_train_under, y_train_under)

    selected_params = grid_search.best_params_
    feature_name_tracker = grid_search.best_estimator_.named_steps.get('featurenametracker')

    if feature_name_tracker:
        selected_features = feature_name_tracker.selected_feature_names_
    else:
        selected_features = X_train.columns.tolist()

    predictions = grid_search.predict(X_test)
    y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]

    # Prepare the dictionary for this fold's results
    fold_results = {
        'Fold': fold_counter,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1 Score': f1_score(y_test, predictions),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'Selected Parameters': selected_params,
        'Selected Features': selected_features,
        'Selection Technique': selection_techniques.get(selection_method, 'Unknown')
    }

    # Show progression when training models
    print(f"Processed fold {fold_counter}")

    collect_results[fold_counter] = fold_results
    fold_counter += 1

store_results(collect_results)
